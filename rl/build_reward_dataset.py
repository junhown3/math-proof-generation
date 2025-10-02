"""Build reward dataset by joining candidate metadata with judge scores.

Outputs (default):
    data/rl/reward_dataset.jsonl  (one row per candidate with reward scalar)

Enhanced Features:
 - Custom reward modes to emphasize specific rubric dimensions (e.g., correctness) or a weighted
     combination of dimensions instead of the pre-computed `overall` field.
 - Optional per-theorem normalization (min-max) to prevent easy theorems from dominating reward scale.

Reward Modes:
    --reward-mode overall            -> use judge `overall` (default)
    --reward-mode dimension --reward-dimension correctness
                                                                        -> use a single rubric dimension score
    --reward-mode weighted --weights correctness=0.6,completeness=0.25,cohesion=0.1,retrieval_grounding=0.05
                                                                        -> weighted sum (missing dims treated as 0)

Normalization:
    --normalize-per-theorem          -> apply min-max over the candidates of each theorem AFTER raw reward computation.
                                                                            Values collapse to midpoint (0.5) if only one candidate.

Future extension: pairwise preference generation lives in separate script.
"""
from __future__ import annotations
import os, json, argparse
from typing import Dict, Any
from rl.schema import RewardRow, RUBRIC_DIMENSIONS


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
    ap.add_argument('--min-overall', type=float, default=0.0, help='(Deprecated name) threshold applied to final reward value')
    ap.add_argument('--paper', help='If set, restrict to this paper id')
    ap.add_argument('--theorems', help='Comma/range list filter (e.g. 0-3,5) applied after paper filter')
    ap.add_argument('--reward-mode', default='overall', choices=['overall','dimension','weighted'],
                    help='How to compute reward scalar from judge scores.')
    ap.add_argument('--reward-dimension', help='Dimension name when --reward-mode=dimension (e.g. correctness)')
    ap.add_argument('--weights', help='Comma list of dim=weight for --reward-mode=weighted (e.g. correctness=0.6,completeness=0.25)')
    ap.add_argument('--normalize-per-theorem', action='store_true', help='Min-max normalize rewards within each theorem to [0,1].')
    ap.add_argument('--min-norm', type=float, default=0.1, help='Floor after normalization (scaled to [min_norm,1]).')
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

    # Pre-parse weights if needed
    weight_map = {}
    if args.reward_mode == 'dimension':
        if not args.reward_dimension:
            raise SystemExit('Must supply --reward-dimension when --reward-mode=dimension')
        if args.reward_dimension not in RUBRIC_DIMENSIONS:
            raise SystemExit(f'Unknown dimension {args.reward_dimension}; valid: {RUBRIC_DIMENSIONS}')
    elif args.reward_mode == 'weighted':
        if not args.weights:
            raise SystemExit('Must supply --weights for weighted mode.')
        for part in args.weights.split(','):
            part = part.strip()
            if not part:
                continue
            if '=' not in part:
                raise SystemExit(f'Invalid weight spec: {part}')
            k,v = part.split('=',1)
            try:
                weight_map[k.strip()] = float(v)
            except ValueError:
                raise SystemExit(f'Invalid float weight for {k}: {v}')
        if not weight_map:
            raise SystemExit('Parsed zero weights.')

    # Collect raw reward rows first (optionally normalized later)
    temp_rows = []  # (RewardRow, raw_reward)
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
        # Derive reward
        if args.reward_mode == 'overall':
            reward_val = float(sc.get('overall', 0.0))
        elif args.reward_mode == 'dimension':
            reward_val = float(sc.get('scores', {}).get(args.reward_dimension, 0.0))
        else:  # weighted
            sdict = sc.get('scores', {}) or {}
            total = 0.0
            wsum = 0.0
            for k,w in weight_map.items():
                total += w * float(sdict.get(k, 0.0))
                wsum += w
            reward_val = total / wsum if wsum else 0.0
        if reward_val < args.min_overall:
            continue
        rr = RewardRow(
            candidate_id=cid,
            paper_id=meta.get('paper_id'),
            theorem_index=th_i,
            reward=reward_val,  # may replace if we normalize later
            model=meta.get('model'),
            variant=meta.get('variant'),
            meta={
                'temperature': meta.get('temperature'),
                'rag_enabled': meta.get('rag_enabled'),
                'rag_top_k': meta.get('rag_top_k'),
                'prompt_char_len': meta.get('prompt_char_len'),
                'context_char_len': meta.get('context_char_len'),
                'reward_mode': args.reward_mode,
                'reward_dimension': args.reward_dimension if args.reward_mode=='dimension' else None,
                'weights': weight_map if args.reward_mode=='weighted' else None,
                'normalized': False
            }
        )
        temp_rows.append(rr)

    # Optional per-theorem normalization
    if args.normalize_per_theorem and temp_rows:
        by_theorem = {}
        for rr in temp_rows:
            by_theorem.setdefault(rr.theorem_index, []).append(rr)
        for th, lst in by_theorem.items():
            vals = [r.reward for r in lst]
            mn, mx = min(vals), max(vals)
            rng = mx - mn
            for r in lst:
                if rng == 0:
                    norm = 0.5
                else:
                    norm = (r.reward - mn) / rng
                # Scale to [min_norm,1]
                if args.min_norm > 0 and args.min_norm < 1:
                    norm = args.min_norm + (1 - args.min_norm) * norm
                r.reward = norm
                r.meta['normalized'] = True

    with open(args.out, 'w', encoding='utf-8') as outf:
        for rr in temp_rows:
            outf.write(rr.to_json() + '\n')
            written += 1

    print(f'Wrote {written} reward rows to {args.out} (mode={args.reward_mode}' +
          (f", dimension={args.reward_dimension}" if args.reward_mode=='dimension' else '') +
          (f", weights={weight_map}" if args.reward_mode=='weighted' else '') +
          (", normalized" if args.normalize_per_theorem else '') + ')')

if __name__ == '__main__':
    main()
