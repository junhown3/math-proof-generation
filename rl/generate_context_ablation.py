"""Generate proofs for context ablation experiment (full vs minimal context).

This DOES NOT modify existing generation scripts; it's additive.

Modes:
  full    -> uses ContextPreparator (RAG on/off via flag) identical to main pipeline
  minimal -> theorem-only prompt (no metadata, abstract, sections, retrieval, available theorems)

Outputs candidate JSON lines with meta.context_mode = full|minimal and stored in separate directories.
"""
from __future__ import annotations
import argparse, os, json, hashlib
from pathlib import Path
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_manager import DataManager
from latex_parser import LatexParser
from context_preparator import ContextPreparator
from rl.schema import Candidate, CandidateMeta

MIN_PROMPT_TMPL = (
    "System: You are an expert mathematician.\n"
    "User:\nProvide a rigorous proof (or best-effort with clearly labeled GAP lines) of the statement below.\n\n"
    "{theorem}\n\nAssistant:\n"
)

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper', required=True)
    ap.add_argument('--theorems', required=True, help='Range/list e.g. 0-3 or 0,2,5')
    ap.add_argument('--base-model', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--context-mode', choices=['full','minimal'], default='full')
    ap.add_argument('--rag', action='store_true', help='Enable RAG in full mode (ignored in minimal).')
    ap.add_argument('--rag-top-k', type=int, default=6)
    ap.add_argument('--rag-chunk-size', type=int, default=900)
    ap.add_argument('--rag-overlap', type=int, default=150)
    ap.add_argument('--max-length', type=int, default=2048)
    ap.add_argument('--max-new-tokens', type=int, default=768)
    ap.add_argument('--temperature', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--double-canonical', action='store_true', help='If set and full mode: set max_new_tokens = min(2x canonical proof tokens, provided --max-new-tokens ceiling).')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    dm = DataManager()
    paper = dm.load_paper(args.paper)
    if not paper:
        raise SystemExit(f"Paper {args.paper} not found.")
    parser = LatexParser()
    parsed = parser.parse(paper.latex_content or '')
    theorems = parsed.get('theorems', [])
    indices = parse_range(args.theorems)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    ctx_prep = ContextPreparator()
    generated = 0
    for ord_i, th_i in enumerate(indices):
        if th_i >= len(theorems):
            print(f"[skip] theorem {th_i} out of range")
            continue
        # Skip existing
        if not args.force:
            for fp in Path(args.out_dir).glob('*.json'):
                try:
                    js = json.loads(fp.read_text(encoding='utf-8'))
                    if js.get('meta',{}).get('paper_id')==args.paper and js.get('meta',{}).get('theorem_index')==th_i and js.get('meta',{}).get('context_mode')==args.context_mode:
                        print(f"[skip] existing {args.context_mode} candidate theorem {th_i}")
                        break
                except Exception:
                    pass
            else:
                pass  # no break
            # If broken we skip
            else_pass = False
        rag_enabled = (args.context_mode=='full' and args.rag)
        if args.context_mode=='full':
            ctx = ctx_prep.prepare_context(paper, th_i, rag_enabled=rag_enabled,
                                           rag_chunk_size=args.rag_chunk_size, rag_overlap=args.rag_overlap,
                                           rag_top_k=args.rag_top_k)
            prompt = ctx_prep.format_for_llm_prompt(ctx)
            canonical_tokens = 0
            if args.double_canonical:
                proof = getattr(theorems[th_i],'proof',None)
                if proof:
                    canonical_tokens = len(tokenizer(proof, add_special_tokens=False).input_ids)
            dyn_cap = args.max_new_tokens
            if args.double_canonical and canonical_tokens>0:
                dyn_cap = min(args.max_new_tokens, 2 * canonical_tokens)
            max_new = dyn_cap
            context_char_len = len(ctx.paper_context)
        else:
            theorem_stmt = theorems[th_i].statement
            prompt = MIN_PROMPT_TMPL.format(theorem=theorem_stmt)
            max_new = args.max_new_tokens  # minimal prompt: do not shrink
            context_char_len = 0
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=args.max_length).to(model.device)
        gen = model.generate(**inputs, max_new_tokens=max_new, temperature=args.temperature,
                             do_sample=True if args.temperature>0 else False, pad_token_id=tokenizer.eos_token_id)
        full_text = tokenizer.decode(gen[0], skip_special_tokens=True)
        completion = full_text[len(prompt):].strip() or full_text
        variant = 'fullctx' if args.context_mode=='full' else 'minimal'
        meta = CandidateMeta(paper_id=args.paper, theorem_index=th_i, variant=variant, model=args.base_model,
                             temperature=args.temperature, seed=args.seed, rag_enabled=rag_enabled, rag_top_k=args.rag_top_k,
                             rag_chunk_size=args.rag_chunk_size, rag_overlap=args.rag_overlap, prompt_char_len=len(prompt),
                             context_char_len=context_char_len, provenance={'context_mode': args.context_mode})
        cid_base = f"{args.paper}::{th_i}::{variant}::{args.seed}"
        cid = hashlib.sha256(cid_base.encode()).hexdigest()[:16]
        cand = Candidate(cid, meta, theorems[th_i].statement, completion)
        out_path = os.path.join(args.out_dir, f"{cid}.json")
        with open(out_path,'w',encoding='utf-8') as f:
            f.write(cand.to_json())
        print(f"[gen-{args.context_mode}] theorem {th_i} -> {cid} (max_new={max_new})")
        generated += 1
    print(f"Done. Generated {generated} {args.context_mode} candidates -> {args.out_dir}")

if __name__ == '__main__':
    main()
