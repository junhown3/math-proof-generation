"""Utility: List saved papers and their parsed theorem counts.

Purpose: Support reuse of previously ingested papers for RLAIF experiments
without re-fetching from arXiv each run.

Usage:
  python -m rl.list_papers_with_theorems
  python -m rl.list_papers_with_theorems --min-theorems 1 --limit 10

Outputs a table-like listing with: arXiv ID, publication date, theorem count, size char count.
"""
from __future__ import annotations
import argparse, os, json
from data_manager import DataManager


def count_theorems(parsed_path: str) -> int:
    try:
        with open(parsed_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return len(data.get('theorems', []))
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--min-theorems', type=int, default=1)
    ap.add_argument('--limit', type=int, default=50)
    args = ap.parse_args()

    dm = DataManager(data_dir=args.data_dir)
    papers = dm.list_saved_papers()
    rows = []
    for p in papers:
        parsed_path = os.path.join(args.data_dir, 'parsed', f"{p['arxiv_id'].replace('/', '_')}_parsed.json")
        tcount = count_theorems(parsed_path)
        if tcount < args.min_theorems:
            continue
        rows.append({
            'id': p['arxiv_id'],
            'date': p['published_date'][:10],
            'theorems': tcount,
            'latex_size': p['latex_size']
        })
        if len(rows) >= args.limit:
            break

    if not rows:
        print('No papers found meeting criteria.')
        return

    print(f"Found {len(rows)} papers with >= {args.min_theorems} theorems (showing up to {args.limit})")
    print("ID            | Date       | Theorems | LaTeX Size")
    print("--------------+------------+----------+-----------")
    for r in rows:
        print(f"{r['id']:<14} {r['date']:<11} {r['theorems']:^9} {r['latex_size']:,}")
    print('\nSelect one of the IDs above and run pipeline without --auto-fetch to reuse.')


if __name__ == '__main__':
    main()
