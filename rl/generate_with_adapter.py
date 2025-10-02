"""Generate proofs using a base HF model + LoRA adapter (trained via scalar or DPO fine-tune).

This script mirrors a subset of `candidate_generation.py` but loads a local adapter instead of
calling OpenAI / mock backends. It uses the existing context preparation pipeline to produce
RAG-enabled or baseline prompts, then performs local generation.

Usage example:
  python -m rl.generate_with_adapter \
    --paper 2509.22618 --theorems 0-2 \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --adapter models/scalar-ft-tinyllama-realjudge-v1 \
    --out-dir data/rl/candidates_adapter \
    --variant rag --max-new-tokens 512 --temperature 0.3

You can set --variant baseline to disable retrieval. LoRA is kept separate, not merged.

If you want to merge LoRA into a standalone model first, provide the merged model path
in --merged-model instead of --base-model/--adapter pair.
"""
from __future__ import annotations
import argparse, os, json, math, hashlib
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None  # Will raise if adapter path used without peft installed

from data_manager import DataManager
from latex_parser import LatexParser
from context_preparator import ContextPreparator
from rl.schema import Candidate, CandidateMeta

SYSTEM_PREFIX = "System: You are an expert mathematician. Provide a rigorous proof or best-effort reasoning with GAP markers if steps are missing.\nUser:\n"
ASSISTANT_PREFIX = "\nAssistant:\n"

def parse_range(expr: str) -> List[int]:
    out = []
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


def build_prompt(theorem_stmt: str, context: str) -> str:
    return SYSTEM_PREFIX + theorem_stmt.strip() + "\n\nContext:\n" + context.strip() + ASSISTANT_PREFIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper', required=True)
    ap.add_argument('--theorems', required=True, help='Range/list e.g. 0-3 or 0,2,5')
    ap.add_argument('--base-model', help='HF base model id (required unless --merged-model given).')
    ap.add_argument('--adapter', help='Directory containing LoRA adapter (if using base+adapter).')
    ap.add_argument('--merged-model', help='Path to a merged standalone model (skip adapter logic).')
    ap.add_argument('--out-dir', default='data/rl/candidates_adapter')
    ap.add_argument('--variant', default='rag', choices=['rag','baseline'])
    ap.add_argument('--rag-top-k', type=int, default=6)
    ap.add_argument('--rag-chunk-size', type=int, default=900)
    ap.add_argument('--rag-overlap', type=int, default=150)
    ap.add_argument('--max-length', type=int, default=2048)
    ap.add_argument('--max-new-tokens', type=int, default=512)
    ap.add_argument('--temperature', type=float, default=0.3)
    ap.add_argument('--do-sample', action='store_true', help='Enable sampling (default on for temperature>0).')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if not args.merged_model and (not args.base_model or not args.adapter):
        raise SystemExit('Provide either --merged-model OR (--base-model AND --adapter).')

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    dm = DataManager()
    paper = dm.load_paper(args.paper)
    if not paper:
        raise SystemExit(f"Paper {args.paper} not found. Ingest or use an existing one.")
    parser = LatexParser()
    parsed = parser.parse(paper.latex_content or '')
    theorems = parsed.get('theorems', [])
    if not theorems:
        print('Paper has zero theorems; aborting.')
        return
    indices = parse_range(args.theorems)

    # Load model/tokenizer
    model_path_for_tok = args.merged_model or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(model_path_for_tok, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.merged_model:
        model = AutoModelForCausalLM.from_pretrained(args.merged_model, device_map='auto', torch_dtype=torch.bfloat16)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='auto', torch_dtype=torch.bfloat16)
        if not PeftModel:
            raise SystemExit('peft not installed but adapter specified.')
        model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    ctx_prep = ContextPreparator()
    generated = 0
    for ordinal, th_i in enumerate(indices):
        if th_i >= len(theorems):
            print(f'[skip] theorem {th_i} out of range ({len(theorems)})')
            continue
        # Skip if exists and not forcing
        if not args.force:
            already = False
            for fp in Path(args.out_dir).glob('*.json'):
                try:
                    data = json.loads(fp.read_text(encoding='utf-8'))
                    if data.get('meta', {}).get('paper_id') == args.paper and data.get('meta', {}).get('theorem_index') == th_i:
                        already = True; break
                except Exception:
                    pass
            if already:
                print(f'[skip] candidate for theorem {th_i} exists (use --force to regenerate)')
                continue
        rag_enabled = (args.variant == 'rag')
        ctx = ctx_prep.prepare_context(paper, th_i, rag_enabled=rag_enabled,
                                       rag_chunk_size=args.rag_chunk_size, rag_overlap=args.rag_overlap,
                                       rag_top_k=args.rag_top_k)
        prompt = ctx_prep.format_for_llm_prompt(ctx)
        # (Optional) could insert SYSTEM_PREFIX wrappers; currently prompt already structured upstream.
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=args.max_length).to(model.device)
        gen = model.generate(**inputs,
                             max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature,
                             do_sample=True if (args.do_sample or args.temperature>0) else False,
                             pad_token_id=tokenizer.eos_token_id)
        full_text = tokenizer.decode(gen[0], skip_special_tokens=True)
        completion = full_text[len(prompt):].strip() or full_text
        from rl.schema import CandidateMeta, Candidate
        meta = CandidateMeta(paper_id=args.paper, theorem_index=th_i, variant=args.variant,
                             model=(args.merged_model or args.base_model), temperature=args.temperature, seed=args.seed,
                             rag_enabled=rag_enabled, rag_top_k=args.rag_top_k, rag_chunk_size=args.rag_chunk_size,
                             rag_overlap=args.rag_overlap, prompt_char_len=len(prompt),
                             context_char_len=len(ctx.paper_context), provenance=ctx_prep.get_last_provenance())
        cid = Candidate.make_id(args.paper, th_i, args.variant, args.seed, ordinal)
        cand = Candidate(cid, meta, theorems[th_i].statement, completion)
        out_file = os.path.join(args.out_dir, f'{cid}.json')
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(cand.to_json())
        print(f'[gen] theorem {th_i} -> {cid}')
        generated += 1
    print(f'Finished. Generated {generated} new candidates into {args.out_dir}')

if __name__ == '__main__':
    main()
