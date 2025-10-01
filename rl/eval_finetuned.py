"""Evaluation harness for (base or fine-tuned) models.

Steps per theorem:
 1. Load theorem statement (via DataManager / LaTeX parsing like in main pipeline).
 2. Generate N samples with the specified HF model (optionally loading LoRA adapter).
 3. For each sample, build judge prompt and call judge (stub or real) to obtain scores.
 4. Aggregate mean/median overall, print simple table.

This avoids full RAG context assembly for quick alignment eval; you can extend later
by importing the existing context builder if desired.

Usage example:
  python -m rl.eval_finetuned \
    --paper 2509.22618 --theorems 0-2 \
    --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --lora-adapter models/scalar-ft-v1/lora_adapter \
    --samples 3 --max-new-tokens 512 --real-judge --judge-model gpt-4.1-mini

"""
from __future__ import annotations
import argparse, os, json, re, statistics, hashlib, time
from typing import List, Dict, Any

import torch

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
except ImportError:
    raise SystemExit('Missing transformers/peft packages. Install requirements first.')

from data_manager import DataManager
from latex_parser import LatexParser
from rl.judge_scores import build_prompt, judge, _stub_scores  # reuse prompt & judge


def parse_theorem_range(rng: str) -> List[int]:
    out: List[int] = []
    for part in rng.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def sample_proof(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    input_ids = tokenizer(prompt, return_tensors='pt').to(model.device)
    gen = model.generate(**input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=temperature, top_p=0.95, eos_token_id=tokenizer.eos_token_id)
    out = tokenizer.decode(gen[0], skip_special_tokens=True)
    # Return only part after Assistant prefix if present
    if '\nAssistant:\n' in out:
        out = out.split('\nAssistant:\n', 1)[-1].strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper', required=True)
    ap.add_argument('--theorems', required=True, help='e.g. 0-2 or 0,2,5')
    ap.add_argument('--model-name', required=True)
    ap.add_argument('--lora-adapter', help='Path to LoRA adapter directory')
    ap.add_argument('--samples', type=int, default=3)
    ap.add_argument('--temperature', type=float, default=0.3)
    ap.add_argument('--max-new-tokens', type=int, default=512)
    ap.add_argument('--real-judge', action='store_true')
    ap.add_argument('--backend', default='openai')
    ap.add_argument('--judge-model', default='frontier-judge-stub')
    ap.add_argument('--out', default='data/rl/eval_results.jsonl')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print('Loading model/tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto')

    if args.lora_adapter and os.path.isdir(args.lora_adapter):
        print(f'Loading LoRA adapter from {args.lora_adapter}')
        model = PeftModel.from_pretrained(model, args.lora_adapter)

    dm = DataManager()
    paper = dm.load_paper(args.paper)
    if not paper:
        raise SystemExit('Paper not found in data manager.')

    parser_l = LatexParser()
    parsed = parser_l.parse(paper.latex_content or '')
    theorems = parsed.get('theorems', [])

    idxs = parse_theorem_range(args.theorems)
    rows = []
    for idx in idxs:
        if idx >= len(theorems):
            print(f'[skip] theorem index {idx} out of range')
            continue
        th = theorems[idx]
        stmt = getattr(th, 'statement', '') or str(th)
        canonical = getattr(th, 'proof', None)
        print(f'== Theorem {idx} ==')
        for s in range(args.samples):
            # Build simple instruction style prompt consistent with training format
            base_prompt = f"Problem: {stmt}\nProvide a rigorous proof."
            full_prompt = "System: You are an expert mathematician.\nUser:\n" + base_prompt + "\nAssistant:\n"
            proof = sample_proof(model, tokenizer, full_prompt, args.max_new_tokens, args.temperature)
            judge_prompt = build_prompt(stmt, proof, canonical)
            judge_raw, used_fallback = judge(judge_prompt, args.judge_model, real=args.real_judge, backend=args.backend)
            overall = judge_raw.get('overall') if 'overall' in judge_raw else _stub_scores(judge_prompt)['overall']
            row = {
                'paper_id': args.paper,
                'theorem_index': idx,
                'sample_index': s,
                'model_name': args.model_name,
                'lora_adapter': args.lora_adapter or None,
                'proof': proof,
                'overall': overall,
                'scores': judge_raw.get('scores'),
                'fallback': used_fallback,
                'canonical_used': judge_raw.get('canonical_used', False)
            }
            rows.append(row)
            with open(args.out, 'a', encoding='utf-8') as outf:
                outf.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(f" sample {s} overall={overall:.3f}{' (fallback)' if used_fallback else ''}")

    # Aggregate summary
    print('\n=== Aggregate Summary ===')
    by_idx: Dict[int, List[float]] = {}
    for r in rows:
        by_idx.setdefault(r['theorem_index'], []).append(r['overall'])
    for k, vals in sorted(by_idx.items()):
        print(f"Theorem {k}: mean={sum(vals)/len(vals):.3f} median={statistics.median(vals):.3f} n={len(vals)}")
    all_vals = [r['overall'] for r in rows]
    if all_vals:
        print(f"Global mean={sum(all_vals)/len(all_vals):.3f} median={statistics.median(all_vals):.3f} n={len(all_vals)}")

if __name__ == '__main__':
    main()
