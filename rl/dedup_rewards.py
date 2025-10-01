"""Reward dataset cleanup utility.

Purpose:
  - Deduplicate multiple rows for the same candidate_id, keeping the highest reward.
  - Optionally drop rows from specified models (e.g., 'mock-llm').
  - Optionally restrict to a single paper and theorem subset.

Usage examples:
  python -m rl.dedup_rewards \
     --in data/rl/reward_dataset.jsonl \
     --out data/rl/reward_dataset_dedup.jsonl

  python -m rl.dedup_rewards \
     --in data/rl/reward_dataset.jsonl \
     --out data/rl/reward_dataset_dedup_nomock.jsonl \
     --drop-model mock-llm

  python -m rl.dedup_rewards \
     --in data/rl/reward_dataset.jsonl \
     --out data/rl/reward_dataset_filtered.jsonl \
     --paper 2509.22618 --theorems 0-3

Exit codes: 0 on success, >0 on fatal errors.
"""
from __future__ import annotations
import argparse, json, os, sys
from typing import Set


def parse_theorems(expr: str) -> Set[int]:
    out = set()
    for part in expr.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a,b = part.split('-',1)
            out.update(range(int(a), int(b)+1))
        else:
            out.add(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='input', required=True)
    ap.add_argument('--out', dest='output', required=True)
    ap.add_argument('--drop-model', action='append', help='Model name(s) to exclude (can repeat).')
    ap.add_argument('--paper', help='Restrict to this paper id')
    ap.add_argument('--theorems', help='Restrict to these theorem indices (range/list)')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    theorem_filter = parse_theorems(args.theorems) if args.theorems else None
    drop_models = set(args.drop_model or [])

    best = {}
    total = 0
    with open(args.input,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            total += 1
            try:
                row = json.loads(line)
            except Exception:
                continue
            if args.paper and row.get('paper_id') != args.paper:
                continue
            if theorem_filter and row.get('theorem_index') not in theorem_filter:
                continue
            if row.get('model') in drop_models:
                continue
            cid = row.get('candidate_id')
            if not cid:
                continue
            prev = best.get(cid)
            if prev is None or row.get('reward', -1) > prev.get('reward', -1):
                best[cid] = row

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output,'w',encoding='utf-8') as out:
        for row in best.values():
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f"Input rows: {total}  Unique candidates kept: {len(best)}  Output: {args.output}")
    if drop_models:
        print(f"Dropped models: {', '.join(sorted(drop_models))}")
    if args.paper:
        print(f"Filtered paper: {args.paper}")
    if theorem_filter:
        print(f"Filtered theorems: {sorted(theorem_filter)}")


if __name__ == '__main__':
    main()
