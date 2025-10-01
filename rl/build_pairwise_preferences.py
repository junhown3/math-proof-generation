"""Build pairwise preference dataset from scored candidates.

Inputs:
  --candidates-dir  Directory containing candidate *.json (from rl.candidate_generation)
  --judge-scores    Path to judge_scores.jsonl (each line a JudgeScore)
  --out             Output JSONL of pairwise preference rows

Pairwise row schema (one JSON per line):
{
  "paper_id": str,
  "theorem_index": int,
  "prompt": str,          # The proof generation prompt (we reconstruct minimal prompt: theorem statement)
  "chosen": str,          # better proof text
  "rejected": str,        # worse proof text
  "chosen_id": str,
  "rejected_id": str,
  "chosen_score": float,
  "rejected_score": float,
  "score_margin": float,  # chosen_score - rejected_score
  "variant_chosen": str,
  "variant_rejected": str,
  "model_chosen": str,
  "model_rejected": str,
  "paper_theorem_key": "{paper_id}::{theorem_index}",
  "metadata": { ... }      # optional extra provenance
}

Selection strategy:
  * Group by (paper_id, theorem_index)
  * For each group, collect (candidate_id, overall_score)
  * Filter: drop candidates with overall_score missing or generated_proof too short (< 50 chars)
  * Sort descending by score
  * Always produce (best, worst) if >=2
  * Also produce adjacent pairs (i, i+1) if score difference >= min_margin
  * Limit total pairs per theorem with --max-pairs (default 30)

This is a first pass; more advanced strategies (sampling by margin bucket, ensuring variant balance) can be layered later.
"""
from __future__ import annotations
import argparse, os, json, math
from typing import Dict, List, Tuple

MIN_PROOF_CHARS = 50
DEFAULT_MIN_MARGIN = 0.2


def load_candidates(candidates_dir: str) -> Dict[str, Dict]:
    out = {}
    for name in os.listdir(candidates_dir):
        if not name.endswith('.json'):
            continue
        path = os.path.join(candidates_dir, name)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            cid = data.get('candidate_id') or name[:-5]
            out[cid] = data
        except Exception:
            pass
    return out


def load_scores(path: str) -> Dict[str, Dict]:
    out = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                js = json.loads(line)
                cid = js.get('candidate_id')
                if cid:
                    out[cid] = js
            except Exception:
                continue
    return out


def build_pairs(group: List[Dict], min_margin: float, max_pairs: int) -> List[Tuple[Dict, Dict]]:
    pairs: List[Tuple[Dict, Dict]] = []
    if len(group) < 2:
        return pairs
    # Always include (best, worst)
    best = group[0]
    worst = group[-1]
    if best['overall'] > worst['overall']:
        pairs.append((best, worst))
    # Adjacent pairs
    for i in range(len(group) - 1):
        a = group[i]
        b = group[i + 1]
        diff = a['overall'] - b['overall']
        if diff >= min_margin:
            pairs.append((a, b))
        if len(pairs) >= max_pairs:
            break
    # Deduplicate (candidate_id pairs)
    seen = set()
    unique = []
    for a, b in pairs:
        key = (a['candidate_id'], b['candidate_id'])
        if key not in seen:
            seen.add(key)
            unique.append((a, b))
    return unique[:max_pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates-dir', required=True)
    ap.add_argument('--judge-scores', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-margin', type=float, default=DEFAULT_MIN_MARGIN, help='Minimum overall score margin to keep a pair (adjacent)')
    ap.add_argument('--max-pairs', type=int, default=30, help='Max pairs per theorem')
    args = ap.parse_args()

    candidates = load_candidates(args.candidates_dir)
    scores = load_scores(args.judge_scores)

    # Index by paper/theorem
    grouped: Dict[str, List[Dict]] = {}
    for cid, cdata in candidates.items():
        meta = cdata.get('meta', {})
        paper_id = meta.get('paper_id') or meta.get('paper') or cdata.get('paper_id')
        thm_idx = meta.get('theorem_index') if meta.get('theorem_index') is not None else cdata.get('theorem_index')
        if paper_id is None or thm_idx is None:
            continue
        score_entry = scores.get(cid)
        if not score_entry:
            continue
        overall = score_entry.get('overall')
        if overall is None:
            continue
        proof = cdata.get('generated_proof') or ''
        if len(proof.strip()) < MIN_PROOF_CHARS:
            continue
        variant = meta.get('variant', 'unknown')
        key = f"{paper_id}::{thm_idx}"
        grouped.setdefault(key, []).append({
            'candidate_id': cid,
            'overall': float(overall),
            'paper_id': paper_id,
            'theorem_index': thm_idx,
            'variant': variant,
            'model': meta.get('model'),
            'proof': proof,
            'theorem_statement': cdata.get('theorem_statement')
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    total_pairs = 0
    with open(args.out, 'w', encoding='utf-8') as outf:
        for key, rows in grouped.items():
            # Sort descending by overall
            rows.sort(key=lambda r: r['overall'], reverse=True)
            pairs = build_pairs(rows, args.min_margin, args.max_pairs)
            for better, worse in pairs:
                row = {
                    'paper_id': better['paper_id'],
                    'theorem_index': better['theorem_index'],
                    'paper_theorem_key': key,
                    'prompt': better['theorem_statement'],  # minimal prompt; you can expand later
                    'chosen': better['proof'],
                    'rejected': worse['proof'],
                    'chosen_id': better['candidate_id'],
                    'rejected_id': worse['candidate_id'],
                    'chosen_score': better['overall'],
                    'rejected_score': worse['overall'],
                    'score_margin': better['overall'] - worse['overall'],
                    'variant_chosen': better['variant'],
                    'variant_rejected': worse['variant'],
                    'model_chosen': better['model'],
                    'model_rejected': worse['model']
                }
                outf.write(json.dumps(row, ensure_ascii=False) + '\n')
                total_pairs += 1
    print(f"Wrote {total_pairs} preference pairs to {args.out}")

if __name__ == '__main__':
    main()
