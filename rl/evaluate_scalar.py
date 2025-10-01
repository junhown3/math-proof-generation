"""Scalar reward evaluation utility.

Reads a reward dataset JSONL (output of build_reward_dataset.py) and prints:
 - Count per (variant, model)
 - Mean / median / p90 reward per variant
 - Top-k candidates summary
 - Histogram buckets (configurable bucket size)

Usage:
  python -m rl.evaluate_scalar --reward-file data/rl/reward_dataset.jsonl --top-k 5 --bucket 0.5
"""
from __future__ import annotations
import argparse, json, statistics, os
from collections import defaultdict
from typing import List, Dict, Any


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise SystemExit(f"Reward file not found: {path}")
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
    if not rows:
        raise SystemExit("No reward rows parsed.")
    return rows


def summarize(rows: List[Dict[str, Any]], bucket: float, top_k: int):
    by_variant: Dict[str, List[float]] = defaultdict(list)
    by_model: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_variant[r.get('variant','unknown')].append(r['reward'])
        by_model[r.get('model','unknown')].append(r['reward'])

    print("=== Counts by variant ===")
    for v, lst in by_variant.items():
        print(f"{v:15s} {len(lst)}")

    print("\n=== Reward stats by variant ===")
    def stats(lst):
        return {
            'mean': round(sum(lst)/len(lst), 4),
            'median': round(statistics.median(lst), 4),
            'p90': round(sorted(lst)[int(0.9*len(lst))-1], 4)
        }
    for v, lst in by_variant.items():
        s = stats(lst)
        print(f"{v:15s} mean={s['mean']} median={s['median']} p90={s['p90']}")

    print("\n=== Histogram (bucket size = %.2f) ===" % bucket)
    # Build global histogram
    buckets: Dict[float,int] = defaultdict(int)
    for r in rows:
        b = (int(r['reward']/bucket))*bucket
        buckets[b] += 1
    for b in sorted(buckets.keys()):
        print(f"{b:5.2f} - {b+bucket:5.2f}: {buckets[b]}")

    print(f"\n=== Top {top_k} Overall ===")
    top = sorted(rows, key=lambda x: x['reward'], reverse=True)[:top_k]
    for i, row in enumerate(top, 1):
        print(f"#{i} reward={row['reward']:.3f} variant={row.get('variant')} model={row.get('model')} cid={row.get('candidate_id')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reward-file', default='data/rl/reward_dataset.jsonl')
    ap.add_argument('--bucket', type=float, default=0.5)
    ap.add_argument('--top-k', type=int, default=5)
    args = ap.parse_args()
    rows = load_jsonl(args.reward_file)
    summarize(rows, args.bucket, args.top_k)

if __name__ == '__main__':
    main()
