"""Build reward dataset by joining candidate metadata with judge scores.

Outputs:
  data/rl/reward_dataset.jsonl  (one row per candidate with reward scalar)

Future extension: add pairwise preference rows for DPO style training.
"""
from __future__ import annotations
import os, json, argparse
from typing import Dict, Any
from rl.schema import RewardRow


def load_jsonl(path: str):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates-dir', default='data/rl/candidates')
    ap.add_argument('--scores-file', default='data/rl/judge_scores.jsonl')
    ap.add_argument('--out', default='data/rl/reward_dataset.jsonl')
    ap.add_argument('--min-overall', type=float, default=0.0, help='Optionally filter out candidates below threshold')
    ap.add_argument('--paper', help='If set, restrict to this paper id')
    ap.add_argument('--theorems', help='Comma/range list filter (e.g. 0-3,5) applied after paper filter')
    args = ap.parse_args()

    # Index candidates
    cand_index: Dict[str, Dict[str, Any]] = {}
    for fname in os.listdir(args.candidates_dir):
        if not fname.endswith('.json'): continue
        path = os.path.join(args.candidates_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            cand_index[data['candidate_id']] = data
        except Exception:
            continue

    scores = load_jsonl(args.scores_file)
    if not scores:
        print('No scores loaded; aborting.')
        return

    written = 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    theorem_filter = None
    if args.theorems:
        def _parse(expr: str):
            vals = []
            for part in expr.split(','):
                part = part.strip()
                if not part:
                    continue
                if '-' in part:
                    a,b = part.split('-',1)
                    vals.extend(range(int(a), int(b)+1))
                else:
                    vals.append(int(part))
            return set(vals)
        theorem_filter = _parse(args.theorems)

    with open(args.out, 'w', encoding='utf-8') as outf:
        for sc in scores:
            cid = sc.get('candidate_id')
            cand = cand_index.get(cid)
            if not cand:
                continue
            meta = cand.get('meta', {})
            if args.paper and meta.get('paper_id') != args.paper:
                continue
            th_i = meta.get('theorem_index')
            if theorem_filter and th_i not in theorem_filter:
                continue
            overall = sc.get('overall', 0.0)
            if overall < args.min_overall:
                continue
            rr = RewardRow(
                candidate_id=cid,
                paper_id=meta.get('paper_id'),
                theorem_index=th_i,
                reward=overall,
                model=meta.get('model'),
                variant=meta.get('variant'),
                meta={
                    'temperature': meta.get('temperature'),
                    'rag_enabled': meta.get('rag_enabled'),
                    'rag_top_k': meta.get('rag_top_k'),
                    'prompt_char_len': meta.get('prompt_char_len'),
                    'context_char_len': meta.get('context_char_len')
                }
            )
            outf.write(rr.to_json() + '\n')
            written += 1
    print(f'Wrote {written} reward rows to {args.out}')

if __name__ == '__main__':
    main()
