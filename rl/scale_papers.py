"""Scaling experiment orchestration.

Fetch ~K recent math papers, parse LaTeX to extract theorems & proofs, record proof char lengths,
select up to M theorems per paper, and emit a manifest JSONL usable by generation / judging scripts.

This does NOT yet generate candidates (reuse candidate_generation.py) but centralizes selection so
we can compare baseline vs fine-tuned adapters over a broader set.

Usage (example):
  python -m rl.scale_papers --subjects math.NT math.AG --start-date 2024-08-01 \
      --papers 10 --max-theorems 10 --out data/rl/paper_theorem_manifest.jsonl

Manifest row schema:
  {
    "paper_id": str,
    "title": str,
    "published": str (YYYY-MM-DD),
    "theorem_index": int,
    "theorem_statement": str,
    "proof_char_len": int or null,
    "proof_truncated": bool,
    "proof_char_limit": int or null,
    "subjects": [...],
    "abstract_chars": int,
    "paper_latex_chars": int
  }
"""
from __future__ import annotations
import argparse, json, os
from datetime import datetime
from typing import List
from arxiv_fetcher import ArxivFetcher
from data_manager import DataManager
from latex_parser import LatexParser

GPT_OSS_RELEASE_DATE = '2025-08-05'  # Assumed reference date; adjust if needed.

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', nargs='+', default=['math.NT','math.AG'], help='arXiv subject category codes')
    ap.add_argument('--start-date', type=str, default='2024-08-01', help='Only include papers submitted after this date YYYY-MM-DD (or use alias gpt-oss)')
    ap.add_argument('--papers', type=int, default=10, help='Number of papers to fetch (upper bound)')
    ap.add_argument('--max-theorems', type=int, default=10, help='Maximum theorems per paper to keep')
    ap.add_argument('--proof-char-limit', type=int, default=12000, help='Optional soft limit; if proof exceeds this, mark truncated flag (not hard truncate).')
    ap.add_argument('--out', default='data/rl/paper_theorem_manifest.jsonl')
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Support alias for convenience
    if args.start_date.lower() == 'gpt-oss':
        print(f"[info] Using GPT OSS release date alias -> {GPT_OSS_RELEASE_DATE}")
        args.start_date = GPT_OSS_RELEASE_DATE
    start_dt = datetime.strptime(args.start_date, '%Y-%m-%d')
    fetcher = ArxivFetcher()
    print(f"Searching arXiv for up to {args.papers} papers: subjects={args.subjects} start_date>={args.start_date}")
    papers = fetcher.search_papers(args.subjects, start_dt, max_results=args.papers)
    print(f"Fetched {len(papers)} metadata entries")
    lp = LatexParser()
    dm = DataManager()
    rows: List[str] = []
    for p in papers:
        ok = fetcher.download_latex_source(p)
        if not ok or not p.latex_content:
            print(f"[skip] {p.arxiv_id} no latex content")
            continue
        # Save full paper JSON (DataManager expects an ArxivPaper object)
        dm.save_paper(p)
        parsed = lp.parse(p.latex_content)
        # Persist parsed structure for faster later access
        try:
            dm.save_parsed_content(p.arxiv_id, parsed)
        except Exception as e:
            print(f"[warn] Failed to save parsed content for {p.arxiv_id}: {e}")
        theorems = parsed.get('theorems', [])
        if not theorems:
            print(f"[info] {p.arxiv_id} no theorems parsed")
            continue
        keep = theorems[:args.max_theorems]
        for idx, th in enumerate(keep):
            proof = getattr(th, 'proof', None)
            proof_len = len(proof) if proof else None
            truncated = False
            if proof and args.proof_char_limit > 0 and proof_len and proof_len > args.proof_char_limit:
                truncated = True
            row = {
                'paper_id': p.arxiv_id,
                'title': p.title,
                'published': p.published_date.strftime('%Y-%m-%d'),
                'theorem_index': idx,
                'theorem_statement': getattr(th, 'statement', '') or getattr(th, 'name', ''),
                'proof_char_len': proof_len,
                'proof_truncated': truncated,
                'proof_char_limit': args.proof_char_limit,
                'subjects': p.subjects,
                'abstract_chars': len(p.abstract or ''),
                'paper_latex_chars': len(p.latex_content or ''),
            }
            rows.append(json.dumps(row, ensure_ascii=False))
    with open(args.out,'w',encoding='utf-8') as f:
        for r in rows:
            f.write(r+'\n')
    print(f"Wrote {len(rows)} theorem rows across {len(papers)} papers -> {args.out}")

if __name__ == '__main__':
    main()
