"""Build judge input JSONL files (full vs minimal) from individual candidate JSON artifacts.

Purpose:
  Collate generated candidate proofs into two streams for comparative judging:
    - full context variants (e.g., 'rag', 'baseline')
    - minimal context variant ('statement')

Each output line schema (consumed by openai_judge.py):
{
  "id": <stable id string>,
  "paper_id": <paper id>,
  "theorem_index": <int>,
  "statement": <theorem statement>,
  "generation": <candidate proof text>,
  "canonical_proof": <optional canonical proof if attached>
}

We preserve any existing canonical proof fields if earlier stages injected them.

Selection Rules:
  - Minimal file includes ONLY variant == 'statement'
  - Full file includes every OTHER variant (variant != 'statement') unless --restrict-full supplied
  - If --restrict-full is provided, only listed variants (comma list) are included in full (still excluding 'statement').

Usage:
  python -m rl.build_judge_inputs_from_candidates \
     --candidates-dir data/rl/candidates \
     --out-full rl/out/candidates_full.jsonl \
     --out-minimal rl/out/candidates_minimal.jsonl

Optional:
  --restrict-full baseline,rag   (limit full set variants)
  --paper 2509.22618             (filter by a specific paper id)
  --theorems 0-5,7               (restrict theorem indices)
  --require-success              (skip failed generations)
"""
from __future__ import annotations
import argparse, os, json, re
from typing import List, Dict, Any


def parse_range_list(expr: str) -> List[int]:
    out: List[int] = []
    for part in expr.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a,b = part.split('-',1)
            out.extend(range(int(a), int(b)+1))
        else:
            out.append(int(part))
    return sorted(set(out))


def load_candidate_file(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to parse candidate file {path}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates-dir', required=True, help='Directory containing individual candidate *.json')
    ap.add_argument('--out-full', required=True, help='Output JSONL path for full-context variants')
    ap.add_argument('--out-minimal', required=True, help='Output JSONL path for statement-only variants')
    ap.add_argument('--restrict-full', help='Comma list of variants to include in full set (excluding statement)')
    ap.add_argument('--paper', help='Filter by specific paper id')
    ap.add_argument('--theorems', help='Range/list filter for theorem indices (e.g. 0-3,7)')
    ap.add_argument('--force', action='store_true', help='Overwrite existing output files')
    ap.add_argument('--shared-id', action='store_true', help='Drop variant from id so full/minimal rows share the same id per theorem (enables delta computation).')
    ap.add_argument('--require-success', action='store_true', help='Only include candidates with success=true')
    args = ap.parse_args()

    if (os.path.exists(args.out_full) or os.path.exists(args.out_minimal)) and not args.force:
        raise SystemExit('Output file(s) exist; use --force to overwrite.')

    restrict_variants = None
    if args.restrict_full:
        restrict_variants = {v.strip() for v in args.restrict_full.split(',') if v.strip()}

    theorem_filter = None
    if args.theorems:
        theorem_filter = set(parse_range_list(args.theorems))

    files = [os.path.join(args.candidates_dir, f) for f in os.listdir(args.candidates_dir) if f.endswith('.json')]
    if not files:
        raise SystemExit('No candidate *.json files found.')

    # Sort for deterministic ordering
    files.sort()

    minimal_records: List[Dict[str, Any]] = []
    full_records: List[Dict[str, Any]] = []

    for path in files:
        data = load_candidate_file(path)
        paper_id = data.get('paper_id')
        if args.paper and paper_id != args.paper:
            continue
        th_idx = data.get('theorem_index')
        if theorem_filter and th_idx not in theorem_filter:
            continue
        variant = data.get('variant') or 'baseline'
        if args.require_success and not data.get('success'):
            continue
        base_id = data.get('paper_id', 'unknown') + f"_th{th_idx}_" + data.get('model_used','model')
        if not args.shared_id:
            base_id += f"_{variant}"
        record = {
            'paper_id': paper_id,
            'theorem_index': th_idx,
            'statement': data.get('theorem_statement',''),
            'generation': data.get('generated_proof',''),
        }
        # Attach canonical proof if embedded
        canon = None
        quality = data.get('quality') or {}
        # Some pipelines may attach canonical_proof directly; otherwise skip
        if 'canonical_proof' in data:
            canon = data['canonical_proof']
        elif isinstance(quality, dict) and 'canonical_proof' in quality:
            canon = quality['canonical_proof']
        if canon:
            record['canonical_proof'] = canon

        if variant == 'statement':
            minimal_records.append(record)
        else:
            if restrict_variants and variant not in restrict_variants:
                continue
            full_records.append(record)

    # Write outputs
    with open(args.out_full, 'w', encoding='utf-8') as f_full:
        for r in full_records:
            f_full.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(args.out_minimal, 'w', encoding='utf-8') as f_min:
        for r in minimal_records:
            f_min.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"Wrote {len(full_records)} full-context candidate lines -> {args.out_full}")
    print(f"Wrote {len(minimal_records)} statement-only candidate lines -> {args.out_minimal}")
    if not minimal_records:
        print('[warn] No statement-only variants found. Did you include variant=statement in generation?')

if __name__ == '__main__':
    main()
