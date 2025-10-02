"""Batch orchestration for multi-paper evaluation.

Reads a manifest produced by scale_papers.py and for each (paper_id, theorem_index):
 1. Generate baseline candidate (if missing)
 2. Generate adapter candidate (if --adapter-model provided)
 3. Judge both (real or stub) writing to separate score files
After processing, invokes compare_scores (if both before/after provided) to summarize delta.

This is a pragmatic wrapper around existing scripts to avoid manual loops.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, hashlib
from pathlib import Path
from data_manager import DataManager
from latex_parser import LatexParser
from transformers import AutoTokenizer

PYTHON = sys.executable or 'python'

def run(cmd: list[str]):
    print('[cmd]', ' '.join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        print(r.stdout)
        raise SystemExit(f'Command failed: {cmd}')
    else:
        if r.stdout.strip():
            print(r.stdout)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True, help='JSONL from scale_papers')
    ap.add_argument('--baseline-out', default='data/rl/candidates_baseline')
    ap.add_argument('--adapter-out', default='data/rl/candidates_adapter_batch')
    ap.add_argument('--base-model', required=True)
    ap.add_argument('--adapter', help='LoRA adapter directory (if omitted, skip adapter generation)')
    ap.add_argument('--theorems-per-paper', type=int, default=10)
    ap.add_argument('--max-new-tokens', type=int, default=512)
    ap.add_argument('--double-canonical', action='store_true', help='Override per-theorem max_new_tokens to 2x canonical proof token length (fallback to --max-new-tokens if no canonical).')
    ap.add_argument('--temperature', type=float, default=0.3)
    ap.add_argument('--judge-model', default='gpt-5')
    ap.add_argument('--real-judge', action='store_true')
    ap.add_argument('--judge-backend', default='openai')
    ap.add_argument('--canonical-max-chars', type=int, default=4000)
    ap.add_argument('--paper-limit', type=int, default=9999)
    ap.add_argument('--compare-out', default='data/rl/score_comparison_batch.txt')
    ap.add_argument('--force-generate', action='store_true')
    ap.add_argument('--force-rescore', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.baseline_out, exist_ok=True)
    if args.adapter:
        os.makedirs(args.adapter_out, exist_ok=True)

    # Group manifest entries by paper_id
    by_paper = {}
    rows = []
    with open(args.manifest,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            js = json.loads(line)
            by_paper.setdefault(js['paper_id'], []).append(js)
            rows.append(js)

    # Limit per paper theorems
    for pid in list(by_paper.keys()):
        by_paper[pid] = sorted(by_paper[pid], key=lambda r: r['theorem_index'])[:args.theorems_per_paper]
    paper_ids = list(by_paper.keys())[:args.paper_limit]

    dm = DataManager()
    lp = LatexParser()
    tokenizer = None
    if args.double_canonical:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Candidate generation loops (possibly per-theorem if dynamic sizing requested)
    for pid in paper_ids:
        th_indices = [r['theorem_index'] for r in by_paper[pid]]
        if args.double_canonical:
            paper = dm.load_paper(pid)
            theorems = []
            if paper and paper.latex_content:
                parsed = lp.parse(paper.latex_content)
                theorems = parsed.get('theorems', [])
            for th_i in th_indices:
                # compute 2x canonical token length
                max_new = args.max_new_tokens
                if th_i < len(theorems):
                    proof = getattr(theorems[th_i], 'proof', None)
                    if proof:
                        canon_tokens = len(tokenizer(proof, add_special_tokens=False).input_ids)
                        max_new = max(32, 2 * canon_tokens)  # ensure a small lower bound
                # Baseline
                run([PYTHON, '-m', 'rl.generate_with_adapter', '--paper', pid, '--theorems', str(th_i),
                     '--merged-model', args.base_model, '--out-dir', args.baseline_out, '--variant', 'rag',
                     '--max-new-tokens', str(max_new), '--temperature', str(args.temperature)] + (['--force'] if args.force_generate else []))
                if args.adapter:
                    run([PYTHON, '-m', 'rl.generate_with_adapter', '--paper', pid, '--theorems', str(th_i),
                         '--base-model', args.base_model, '--adapter', args.adapter, '--out-dir', args.adapter_out, '--variant', 'rag',
                         '--max-new-tokens', str(max_new), '--temperature', str(args.temperature)] + (['--force'] if args.force_generate else []))
        else:
            range_expr = ','.join(str(i) for i in th_indices)
            run([PYTHON, '-m', 'rl.generate_with_adapter', '--paper', pid, '--theorems', range_expr,
                 '--merged-model', args.base_model, '--out-dir', args.baseline_out, '--variant', 'rag',
                 '--max-new-tokens', str(args.max_new_tokens), '--temperature', str(args.temperature)] + (['--force'] if args.force_generate else []))
            if args.adapter:
                run([PYTHON, '-m', 'rl.generate_with_adapter', '--paper', pid, '--theorems', range_expr,
                     '--base-model', args.base_model, '--adapter', args.adapter, '--out-dir', args.adapter_out, '--variant', 'rag',
                     '--max-new-tokens', str(args.max_new_tokens), '--temperature', str(args.temperature)] + (['--force'] if args.force_generate else []))

    # Judge baseline
    baseline_scores = 'data/rl/judge_scores_batch_baseline.jsonl'
    adapter_scores = 'data/rl/judge_scores_batch_adapter.jsonl'
    for pid in paper_ids:
        th_indices = [r['theorem_index'] for r in by_paper[pid]]
        if not th_indices:
            continue
        min_th, max_th = min(th_indices), max(th_indices)
        range_expr = f"{min_th}-{max_th}" if len(th_indices) > 1 and (max_th - min_th + 1) == len(th_indices) else ','.join(str(i) for i in th_indices)
        judge_base_cmd = [PYTHON, '-m', 'rl.judge_scores', '--paper', pid, '--theorems', range_expr,
                           '--candidates-dir', args.baseline_out, '--out', baseline_scores,
                           '--judge-model', args.judge_model, '--canonical-max-chars', str(args.canonical_max_chars)]
        if args.real_judge:
            judge_base_cmd.append('--real-judge'); judge_base_cmd.extend(['--backend', args.judge_backend])
        if args.force_rescore:
            judge_base_cmd.append('--force-rescore')
        run(judge_base_cmd)
        if args.adapter:
            judge_adp_cmd = [PYTHON, '-m', 'rl.judge_scores', '--paper', pid, '--theorems', range_expr,
                              '--candidates-dir', args.adapter_out, '--out', adapter_scores,
                              '--judge-model', args.judge_model, '--canonical-max-chars', str(args.canonical_max_chars)]
            if args.real_judge:
                judge_adp_cmd.append('--real-judge'); judge_adp_cmd.extend(['--backend', args.judge_backend])
            if args.force_rescore:
                judge_adp_cmd.append('--force-rescore')
            run(judge_adp_cmd)

    if args.adapter:
        # Comparison: we can't rely on consecutive theorem ranges across papers, so just run without per-paper filtering; compare_scores expects a single paper id though -> we will skip and instead produce a simple diff script later if needed.
        # For now we warn user to manually analyze multi-paper JSONL.
        print('[info] Multi-paper comparison: use custom analysis or extend compare_scores for multi-paper.')

    print('Batch orchestration complete.')

if __name__ == '__main__':
    main()
