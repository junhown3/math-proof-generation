"""Compare two judge score JSONL files (before vs after fine-tune) and report uplift statistics.

Usage:
  python -m rl.compare_scores \
      --before data/rl/judge_scores_real.jsonl \
      --after data/rl/judge_scores_adapter.jsonl \
      --paper 2509.22618 --theorems 0-2 \
      --out data/rl/score_comparison.txt

Outputs summary to stdout and (optionally) to --out file:
 - Counts (candidates per theorem)
 - Mean overall score before vs after (+ absolute / % delta)
 - Per-dimension mean deltas
 - Per-theorem table (best before vs best after improvement)
 - Distribution shift (quantiles) if enough samples

Assumptions:
 - Each JSON line follows JudgeScore schema (candidate_id, theorem_index, scores{...}, overall).
 - You may have multiple candidates per theorem per file; we keep all but identify per-theorem best.
"""
from __future__ import annotations
import argparse, json, os, statistics, math
from typing import Dict, List, Tuple, Optional

DIM_ORDER = ["correctness","completeness","cohesion","retrieval_grounding","conciseness"]

class ScoreRow:
    __slots__ = ("candidate_id","paper_id","theorem_index","overall","scores")
    def __init__(self, js: dict):
        self.candidate_id = js.get('candidate_id')
        self.paper_id = js.get('paper_id')
        self.theorem_index = js.get('theorem_index')
        self.overall = float(js.get('overall', 0.0))
        self.scores = js.get('scores', {}) or {}


def read_scores(path: str, paper_filter: Optional[str], theorem_filter: Optional[List[int]]) -> List[ScoreRow]:
    rows: List[ScoreRow] = []
    if not os.path.exists(path):
        raise SystemExit(f"Score file not found: {path}")
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                js=json.loads(line)
            except Exception:
                continue
            if paper_filter and js.get('paper_id')!=paper_filter:
                continue
            if theorem_filter and js.get('theorem_index') not in theorem_filter:
                continue
            rows.append(ScoreRow(js))
    return rows


def agg_mean(rows: List[ScoreRow]) -> float:
    return statistics.mean([r.overall for r in rows]) if rows else float('nan')

def per_dim_mean(rows: List[ScoreRow]) -> Dict[str,float]:
    out={}
    for dim in DIM_ORDER:
        vals=[r.scores.get(dim) for r in rows if isinstance(r.scores.get(dim),(int,float))]
        out[dim]=statistics.mean(vals) if vals else float('nan')
    return out


def best_per_theorem(rows: List[ScoreRow]) -> Dict[int, ScoreRow]:
    best: Dict[int, ScoreRow] = {}
    for r in rows:
        cur=best.get(r.theorem_index)
        if cur is None or r.overall>cur.overall:
            best[r.theorem_index]=r
    return best


def quantiles(values: List[float]) -> Dict[str,float]:
    if not values:
        return {}
    vs=sorted(values)
    def q(p):
        if not vs: return float('nan')
        idx=(len(vs)-1)*p
        lo=math.floor(idx); hi=math.ceil(idx)
        if lo==hi: return vs[lo]
        frac=idx-lo
        return vs[lo]*(1-frac)+vs[hi]*frac
    return {"p10":q(0.10),"p25":q(0.25),"p50":q(0.50),"p75":q(0.75),"p90":q(0.90)}


def format_delta(a: float, b: float) -> str:
    if any(math.isnan(x) for x in (a,b)):
        return 'nan'
    absd = b - a
    pct = (absd / a * 100.0) if a != 0 else float('inf')
    return f"{b:.3f} (Δ {absd:+.3f}, {pct:+.1f}%)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--before', required=True, help='Pre-finetune judge scores JSONL')
    ap.add_argument('--after', required=True, help='Post-finetune (adapter) judge scores JSONL')
    ap.add_argument('--paper', help='Filter by paper id')
    ap.add_argument('--theorems', help='Range/list like 0-3 or 0,2', default=None)
    ap.add_argument('--out', help='Optional output summary file')
    args = ap.parse_args()

    theorem_list=None
    if args.theorems:
        theorem_list=[]
        for part in args.theorems.split(','):
            part=part.strip()
            if not part: continue
            if '-' in part:
                a,b=part.split('-',1)
                theorem_list.extend(range(int(a),int(b)+1))
            else:
                theorem_list.append(int(part))
        theorem_list=sorted(set(theorem_list))

    before_rows=read_scores(args.before, args.paper, theorem_list)
    after_rows=read_scores(args.after, args.paper, theorem_list)

    if not before_rows:
        raise SystemExit('No rows loaded from --before (check filters).')
    if not after_rows:
        raise SystemExit('No rows loaded from --after (check filters).')

    mean_before=agg_mean(before_rows)
    mean_after=agg_mean(after_rows)
    dims_before=per_dim_mean(before_rows)
    dims_after=per_dim_mean(after_rows)

    best_before=best_per_theorem(before_rows)
    best_after=best_per_theorem(after_rows)
    all_theorems=sorted(set(list(best_before.keys())+list(best_after.keys())))

    q_before=quantiles([r.overall for r in before_rows])
    q_after=quantiles([r.overall for r in after_rows])

    lines=[]
    lines.append('=== Score Comparison Summary ===')
    if args.paper: lines.append(f"Paper: {args.paper}")
    if theorem_list: lines.append(f"Theorems: {theorem_list}")
    lines.append(f"Total candidates before: {len(before_rows)}  after: {len(after_rows)}")
    lines.append(f"Mean overall before: {mean_before:.3f}  after: {mean_after:.3f}  Δ={mean_after-mean_before:+.3f} ({((mean_after-mean_before)/mean_before*100) if mean_before else float('inf'):+.1f}%)")

    lines.append('\nPer-dimension means (after vs before):')
    for dim in DIM_ORDER:
        b=dims_before[dim]; a=dims_after[dim]
        if any(math.isnan(x) for x in (a,b)):
            lines.append(f"  {dim:<18} n/a")
        else:
            lines.append(f"  {dim:<18} {a:.3f} (before {b:.3f}, Δ {a-b:+.3f}, {(a-b)/b*100 if b else float('inf'):+.1f}%)")

    lines.append('\nBest-per-theorem uplift:')
    lines.append('theorem | best_before | best_after | Δabs | Δ%')
    for th in all_theorems:
        bb=best_before.get(th); ba=best_after.get(th)
        if not bb or not ba:
            lines.append(f"{th:7d} | missing")
            continue
        absd=ba.overall - bb.overall
        pct=(absd / bb.overall * 100.0) if bb.overall else float('inf')
        lines.append(f"{th:7d} | {bb.overall:11.3f} | {ba.overall:10.3f} | {absd:+.3f} | {pct:+.1f}%")

    if q_before and q_after:
        lines.append('\nOverall distribution quantiles:')
        header='quantile | before | after | Δ'
        lines.append(header)
        for q in ['p10','p25','p50','p75','p90']:
            if q in q_before and q in q_after:
                lines.append(f"{q:8s} | {q_before[q]:6.3f} | {q_after[q]:6.3f} | {q_after[q]-q_before[q]:+6.3f}")

    summary='\n'.join(lines)
    print(summary)
    if args.out:
        with open(args.out,'w',encoding='utf-8') as f:
            f.write(summary+'\n')
        print(f'Wrote summary to {args.out}')

if __name__ == '__main__':
    main()
