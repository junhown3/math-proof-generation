"""Compare judge scores between full-context and minimal-context candidates.

Assumptions:
 - Each JSONL row from judge_scores contains candidate_id, paper_id, theorem_index, scores{}, overall.
 - Candidate meta variant names differ (e.g., fullctx vs minimal) OR we match via candidate_id mapping file.

We join by (paper_id, theorem_index, variant) assuming exactly one fullctx and one minimal per (paper,theorem).
If multiple, we take the highest overall within each mode as a simple best-of (configurable later).
"""
from __future__ import annotations
import argparse, json, statistics, os, sys, pathlib
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from rl.schema import RUBRIC_DIMENSIONS

def load_scores(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                js=json.loads(line)
                rows.append(js)
            except Exception:
                pass
    return rows

def index_by_mode(rows: List[Dict[str,Any]]) -> Dict[Tuple[str,int,str], List[Dict[str,Any]]]:
    out=defaultdict(list)
    for r in rows:
        pid = r.get('paper_id'); th = r.get('theorem_index')
        # variant expected like fullctx / minimal; fallback to judge_model+flag if absent
        variant = r.get('variant') or r.get('meta',{}).get('variant') or 'unknown'
        out[(pid, th, variant)].append(r)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', required=True, help='Judge scores JSONL for full-context mode')
    ap.add_argument('--minimal', required=True, help='Judge scores JSONL for minimal-context mode')
    ap.add_argument('--out', default='data/rl/context_ablation_report.txt')
    ap.add_argument('--best-of', action='store_true', help='If multiple candidates per mode, use best overall score.')
    args = ap.parse_args()

    full_rows = load_scores(args.full)
    min_rows = load_scores(args.minimal)

    # Re-key by (paper,theorem)
    full_best = {}
    for r in full_rows:
        key = (r['paper_id'], r['theorem_index'])
        cur = full_best.get(key)
        if not cur or (args.best_of and r['overall'] > cur['overall']):
            full_best[key] = r
        elif not args.best_of and cur is None:
            full_best[key] = r
    min_best = {}
    for r in min_rows:
        key = (r['paper_id'], r['theorem_index'])
        cur = min_best.get(key)
        if not cur or (args.best_of and r['overall'] > cur['overall']):
            min_best[key] = r
        elif not args.best_of and cur is None:
            min_best[key] = r

    common_keys = sorted(set(full_best.keys()) & set(min_best.keys()))
    if not common_keys:
        print('No overlapping theorem pairs found.')
        return

    deltas_overall = []
    dim_deltas = {d: [] for d in RUBRIC_DIMENSIONS}
    rows_out: List[str] = []
    win_correct = lose_correct = tie_correct = 0

    for key in common_keys:
        f = full_best[key]; m = min_best[key]
        d_overall = f['overall'] - m['overall']
        deltas_overall.append(d_overall)
        line_parts = [key[0], str(key[1]), f"{m['overall']:.3f}", f"{f['overall']:.3f}", f"{d_overall:+.3f}"]
        for d in RUBRIC_DIMENSIONS:
            fv = f['scores'].get(d,0.0); mv = m['scores'].get(d,0.0)
            dd = fv - mv
            dim_deltas[d].append(dd)
            line_parts.append(f"{mv:.2f}/{fv:.2f}/{dd:+.2f}")
        # correctness win stats
        corr_delta = f['scores'].get('correctness',0.0) - m['scores'].get('correctness',0.0)
        if corr_delta>1e-6: win_correct+=1
        elif corr_delta<-1e-6: lose_correct+=1
        else: tie_correct+=1
        rows_out.append('\t'.join(line_parts))

    def stats(arr):
        return {
            'mean': statistics.mean(arr),
            'median': statistics.median(arr),
            'p25': statistics.quantiles(arr, n=4)[0],
            'p75': statistics.quantiles(arr, n=4)[2]
        } if arr else {}

    overall_stats = stats(deltas_overall)
    dim_stats = {d: stats(v) for d,v in dim_deltas.items()}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,'w',encoding='utf-8') as f:
        f.write('Context Ablation Report (full - minimal)\n')
        f.write(f'Theorem pairs: {len(common_keys)}\n')
        f.write('\nSummary Overall Delta:\n')
        f.write(json.dumps(overall_stats, indent=2) + '\n')
        f.write('\nPer-Dimension Delta Stats:\n')
        f.write(json.dumps(dim_stats, indent=2) + '\n')
        f.write(f"\nCorrectness wins/losses/ties: {win_correct}/{lose_correct}/{tie_correct}\n")
        header = ['paper','thm','overall_min','overall_full','Δoverall'] + [f"{d} m/f/Δ" for d in RUBRIC_DIMENSIONS]
        f.write('\n' + '\t'.join(header) + '\n')
        for rline in rows_out:
            f.write(rline + '\n')
    print(f"Wrote context ablation comparison -> {args.out}")

if __name__ == '__main__':
    main()
