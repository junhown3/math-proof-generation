"""GPT-5 Ablation Runner

Runs baseline (no RAG) and RAG variants for a set of theorem indices in one paper (or multiple papers in future)
using the existing main.py infrastructure, captures summary statistics (prompt/context length deltas, structural
hallucination risk distribution), and writes an ablation report JSON.

Usage:
  python -m rl.gpt5_ablation --paper mock_paper_id --theorems 0,1,2 --model gpt-5 \
      --reasoning-effort medium --rag-chunk-size 400 --rag-overlap 80 --rag-top-k 3

Prerequisites: OPENAI_API_KEY set in environment.
"""
from __future__ import annotations
import argparse, os, json, subprocess, sys, tempfile, time
from typing import List, Dict, Any

RESULTS_DIR = 'proof_results'


def parse_indices(spec: str) -> List[int]:
    out = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a,b = part.split('-',1)
            out.extend(range(int(a), int(b)+1))
        else:
            out.append(int(part))
    return sorted(set(out))


def load_result(paper_id: str, idx: int, variant: str) -> Dict[str, Any] | None:
    fname = f"{paper_id.replace('/', '_')}_theorem_{idx}_{variant}.json"
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_ablation_for_index(paper: str, idx: int, model: str, reasoning_effort: str | None,
                           rag_chunk_size: int, rag_overlap: int, rag_top_k: int,
                           force_rag: bool = False, dry_run: bool = False) -> Dict[str, Any]:
    cmd = [sys.executable, 'main.py', '--paper', paper, '--theorem', str(idx), '--backend', 'openai', '--model', model,
           '--ablate-rag', '--rag', '--rag-chunk-size', str(rag_chunk_size), '--rag-overlap', str(rag_overlap), '--rag-top-k', str(rag_top_k)]
    if force_rag:
        cmd.append('--force-rag')
    if reasoning_effort:
        cmd.extend(['--reasoning-effort', reasoning_effort])
    start = time.time()
    proc = None
    if not dry_run:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    # Always attempt to load whatever artifacts were produced (even on failure)
    baseline = load_result(paper, idx, 'baseline')
    rag = load_result(paper, idx, 'rag')
    elapsed = time.time() - start
    model_mismatch = None
    if baseline and rag:
        bm = baseline.get('model_used')
        rm = rag.get('model_used')
        if bm and rm and bm != rm:
            model_mismatch = f"baseline={bm} rag={rm}"
    return {
        'theorem_index': idx,
        'runtime_sec': elapsed,
        'subprocess_error': (proc.returncode if (proc and proc.returncode != 0) else None),
        'stderr': (proc.stderr[:800] if proc and proc.returncode != 0 else None),
        'stdout': (proc.stdout[:800] if proc and proc.returncode != 0 else None),
        'baseline_found': baseline is not None,
        'rag_found': rag is not None,
        'baseline_prompt_len': baseline.get('quality', {}).get('prompt_char_len') if baseline else None,
        'rag_prompt_len': rag.get('quality', {}).get('prompt_char_len') if rag else None,
        'baseline_context_len': baseline.get('quality', {}).get('context_char_len') if baseline else None,
        'rag_context_len': rag.get('quality', {}).get('context_char_len') if rag else None,
        'baseline_structural': baseline.get('quality', {}).get('structural') if baseline else None,
        'rag_structural': rag.get('quality', {}).get('structural') if rag else None,
        'baseline_success': baseline.get('success') if baseline else False,
        'rag_success': rag.get('success') if rag else False,
        'rag_provenance': rag.get('quality', {}).get('provenance', {}).get('rag') if rag else None,
        'model_mismatch': model_mismatch
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    sums = {
        'count': len(rows),
        'prompt_delta_mean': None,
        'context_delta_mean': None,
        'baseline_success_rate': None,
        'rag_success_rate': None,
        'hallucination_shift': None
    }
    if not rows:
        return sums
    pdeltas = []
    cdeltas = []
    bsucc = 0
    rsucc = 0
    hallo_diff = []
    usable_success_denominator = 0
    for r in rows:
        bpl = r.get('baseline_prompt_len')
        rpl = r.get('rag_prompt_len')
        if bpl is not None and rpl is not None and bpl > 0:
            pdeltas.append((bpl - rpl)/bpl)
        bcl = r.get('baseline_context_len')
        rcl = r.get('rag_context_len')
        if bcl is not None and rcl is not None and bcl > 0:
            cdeltas.append((bcl - rcl)/bcl)
        # Only count success rates for rows where at least baseline was attempted/found
        if r.get('baseline_found'):
            usable_success_denominator += 1
            if r.get('baseline_success'): bsucc += 1
            if r.get('rag_success'): rsucc += 1
        bs = (r.get('baseline_structural') or {}).get('hallucination_risk_level')
        rs = (r.get('rag_structural') or {}).get('hallucination_risk_level')
        if bs and rs:
            hallo_diff.append((bs, rs))
    import statistics
    if pdeltas:
        sums['prompt_delta_mean'] = round(statistics.mean(pdeltas), 4)
    if cdeltas:
        sums['context_delta_mean'] = round(statistics.mean(cdeltas), 4)
    if usable_success_denominator:
        sums['baseline_success_rate'] = bsucc/usable_success_denominator
        sums['rag_success_rate'] = rsucc/usable_success_denominator
    if hallo_diff:
        # Count transitions
        from collections import Counter
        trans = Counter([f"{a}->{b}" for a,b in hallo_diff])
        sums['hallucination_shift'] = dict(trans)
    return sums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper', required=True)
    ap.add_argument('--theorems', required=True, help='Comma-separated list or ranges like 0,1,2-5')
    ap.add_argument('--model', default='gpt-5')
    ap.add_argument('--reasoning-effort', default='medium')
    ap.add_argument('--rag-chunk-size', type=int, default=400)
    ap.add_argument('--rag-overlap', type=int, default=80)
    ap.add_argument('--rag-top-k', type=int, default=3)
    ap.add_argument('--out', default='data/rl/gpt5_ablation_report.json')
    ap.add_argument('--force-rag', action='store_true', help='Force retrieval even if document below small-doc threshold')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    indices = parse_indices(args.theorems)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    for idx in indices:
        print(f"Running ablation for theorem {idx}...")
        row = run_ablation_for_index(args.paper, idx, args.model, args.reasoning_effort,
                                     args.rag_chunk_size, args.rag_overlap, args.rag_top_k,
                                     force_rag=args.force_rag, dry_run=args.dry_run)
        rows.append(row)

    summary = summarize(rows)
    report = {
        'paper_id': args.paper,
        'model': args.model,
        'reasoning_effort': args.reasoning_effort,
        'rag_params': {
            'chunk_size': args.rag_chunk_size,
            'overlap': args.rag_overlap,
            'top_k': args.rag_top_k
        },
        'indices': indices,
        'results': rows,
        'summary': summary
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"Ablation report written to {args.out}")
    print("Summary:")
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
