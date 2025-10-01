"""Evaluation harness for RLAIF pipeline.

Computes:
- Average overall judge score per variant (baseline vs rag)
- Pass@k style metric: proportion of theorems where max score among first k candidates >= threshold
- Structural statistics (avg undefined symbols, hallucination risk distribution)

Inputs:
  --scores data/rl/judge_scores.jsonl
  --candidates data/rl/candidates
  (Optionally reward dataset for future correlations)
"""
from __future__ import annotations
import argparse, json, os, math, glob
from collections import defaultdict, Counter
from typing import Dict, List, Any


def load_jsonl(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
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


def load_candidates(candidates_dir: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    for fp in glob.glob(os.path.join(candidates_dir, '*.json')):
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            out[data['candidate_id']] = data
        except Exception:
            continue
    return out


def pass_at_k(scores_for_theorem: List[float], k: int, threshold: float) -> int:
    # Consider first k candidates (as provided order) and see if max >= threshold
    subset = scores_for_theorem[:k]
    return 1 if subset and max(subset) >= threshold else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', default='data/rl/judge_scores.jsonl')
    ap.add_argument('--candidates', default='data/rl/candidates')
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--threshold', type=float, default=6.5, help='Score threshold for pass@k')
    ap.add_argument('--variant-field', default='variant')
    args = ap.parse_args()

    scores = load_jsonl(args.scores)
    cands = load_candidates(args.candidates)
    if not scores:
        print('No scores found.')
        return

    # Group by (paper_id, theorem_index)
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for sc in scores:
        cid = sc.get('candidate_id')
        meta = cands.get(cid, {}).get('meta', {})
        sc['_variant'] = meta.get('variant')
        sc['_paper'] = meta.get('paper_id')
        sc['_theorem'] = meta.get('theorem_index')
        groups[(sc['_paper'], sc['_theorem'])].append(sc)

    variant_scores: Dict[str, List[float]] = defaultdict(list)
    pass_counts: Dict[str, int] = Counter()
    total_theorems: Dict[str, int] = Counter()

    structural_undefined: Dict[str, List[int]] = defaultdict(list)
    hallucination_levels: Dict[str, Counter] = defaultdict(Counter)

    for (paper, thm), rows in groups.items():
        # Stable order: sort by candidate_id for determinism (placeholder until generation order persisted)
        rows_sorted = sorted(rows, key=lambda r: r['candidate_id'])
        # Partition by variant
        by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows_sorted:
            by_variant[r.get('_variant','unknown')].append(r)

        for variant, vr in by_variant.items():
            total_theorems[variant] += 1
            variant_scores[variant].extend([x['overall'] for x in vr])
            # pass@k: use ordering as above
            if pass_at_k([x['overall'] for x in vr], args.k, args.threshold):
                pass_counts[variant] += 1
            # Structural metrics (if original proof file has structural; here not directly stored in judge score, so skip unless later joined)
            # Future improvement: join with proof_result JSONs if needed.

    print('=== Variant Score Summary ===')
    for variant, scores_list in variant_scores.items():
        if not scores_list:
            continue
        avg = sum(scores_list)/len(scores_list)
        print(f"Variant: {variant}\n  Candidates: {len(scores_list)}  Mean Overall: {avg:.3f}")

    print('\n=== pass@k ===')
    for variant, tot in total_theorems.items():
        pk = pass_counts[variant]/tot if tot else 0
        print(f"Variant: {variant}  pass@{args.k} (threshold {args.threshold}): {pk:.2%}  (theorems={tot})")

    # Placeholder for structural since we did not load original quality JSONs here.
    print('\n(Structural metrics integration TODO: requires linking back to proof_results files or embedding structural into candidate metadata.)')

if __name__ == '__main__':
    main()
