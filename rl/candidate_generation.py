"""Candidate generation harness.
Generates multiple proof attempts per theorem (with baseline + RAG variants if desired) and stores them for later judging.

Usage (conceptual):
  python -m rl.candidate_generation --paper <id> --theorem 0 \
      --samples 6 --temperatures 0.1,0.3,0.6 --seeds 42,43 \
      --rag-top-k 6 --rag-chunk-size 600

This script wraps the existing MathematicalProofAgent by toggling temperatures / seeds.
For frontier APIs not supporting seed determinism, we just record a pseudo-random seed tag.
"""
from __future__ import annotations
import argparse, os, random, json, time
from typing import List
from proof_agent_setup import ProofAgentConfig
from proof_agent import MathematicalProofAgent
from data_manager import DataManager
from rl.schema import Candidate, CandidateMeta


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(',') if x.strip()]

def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def build_agent(backend: str, model: str, results_dir: str, rag_enabled: bool, rag_chunk_size: int, rag_overlap: int, rag_top_k: int):
    client = None
    if backend == 'openai':
        client = ProofAgentConfig.create_openai_client(model=model)
    elif backend == 'mock':
        client = ProofAgentConfig.create_mock_client()
    else:
        raise SystemExit(f"Unsupported backend for candidate generation: {backend}")
    return MathematicalProofAgent(client, results_dir=results_dir, rag_enabled=rag_enabled,
                                  rag_chunk_size=rag_chunk_size, rag_overlap=rag_overlap, rag_top_k=rag_top_k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper', required=True)
    parser.add_argument('--theorem', type=int, help='Single theorem index (deprecated if --theorems provided)')
    parser.add_argument('--theorems', help='Range/list like 0-3 or 0,2,5 (overrides --theorem)')
    parser.add_argument('--backend', default='mock')
    parser.add_argument('--model', default='gpt-4o')
    parser.add_argument('--samples', type=int, default=4, help='(Currently informational) total target samples; reduce temps/seeds/variants for fewer actual generations.')
    parser.add_argument('--temperatures', default='0.1,0.3')
    parser.add_argument('--seeds', default='42,43')
    parser.add_argument('--with-rag', action='store_true', help='(Legacy) include rag variant; superseded by --variants')
    parser.add_argument('--variants', default='baseline,rag,statement', help='Comma list of variants to generate (subset of baseline,rag,statement). Include "statement" for theorem-statement-only baseline.')
    parser.add_argument('--rag-chunk-size', type=int, default=900)
    parser.add_argument('--rag-overlap', type=int, default=150)
    parser.add_argument('--rag-top-k', type=int, default=8)
    parser.add_argument('--out-dir', default='data/rl/candidates')
    args = parser.parse_args()

    temps = parse_float_list(args.temperatures)
    seeds = parse_int_list(args.seeds)
    variant_list = [v.strip() for v in args.variants.split(',') if v.strip()]
    # Backward compatibility: if --with-rag and rag not in list, add it; ensure baseline included by default
    if args.with_rag and 'rag' not in variant_list:
        variant_list.append('rag')
    if not variant_list:
        variant_list = ['baseline']
    # Validate
    for v in variant_list:
        if v not in ('baseline', 'rag', 'statement'):
            raise SystemExit(f"Unsupported variant: {v}")

    os.makedirs(args.out_dir, exist_ok=True)

    dm = DataManager()
    paper = dm.load_paper(args.paper)
    if not paper:
        raise SystemExit(f"Paper {args.paper} not found; ingest first.")
    # Determine theorem indices
    def parse_range(expr: str):
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

    if args.theorems:
        theorem_indices = parse_range(args.theorems)
    elif args.theorem is not None:
        theorem_indices = [args.theorem]
    else:
        raise SystemExit('Must provide --theorem or --theorems')

    ordinal = 0
    for th_idx in theorem_indices:
        for seed in seeds:
            for temp in temps:
                for variant in variant_list:
                    rag_enabled = (variant == 'rag')  # statement + baseline both disable RAG
                    agent = build_agent(args.backend, args.model, results_dir='proof_results_temp', rag_enabled=rag_enabled,
                                        rag_chunk_size=args.rag_chunk_size, rag_overlap=args.rag_overlap, rag_top_k=args.rag_top_k)
                    if hasattr(agent.llm_client, 'temperature_override'):
                        agent.llm_client.temperature_override = temp
                    result = agent.generate_proof_for_theorem(args.paper, th_idx, variant_tag=variant)
                    cid = Candidate.make_id(args.paper, th_idx, variant, seed, ordinal)
                    meta = CandidateMeta(
                        paper_id=args.paper, theorem_index=th_idx, variant=variant, model=result.model_used,
                        temperature=temp, seed=seed, rag_enabled=rag_enabled, rag_top_k=args.rag_top_k,
                        rag_chunk_size=args.rag_chunk_size, rag_overlap=args.rag_overlap,
                        prompt_char_len=result.quality.get('prompt_char_len', 0) if result.quality else 0,
                        context_char_len=result.quality.get('context_char_len', 0) if result.quality else 0,
                        provenance=(result.quality or {}).get('provenance')
                    )
                    cand = Candidate(cid, meta, result.theorem_statement, result.generated_proof)
                    with open(os.path.join(args.out_dir, f"{cid}.json"), 'w', encoding='utf-8') as f:
                        f.write(cand.to_json())
                    ordinal += 1

    if len(variant_list)==1 and len(temps)==1 and len(seeds)==1 and len(theorem_indices)>1:
        print(f"Generated {ordinal} candidates (one variant '{variant_list[0]}' per theorem).")
    else:
        print(f"Generated {ordinal} candidates across variants={variant_list}, temps={temps}, seeds={seeds}, theorems={theorem_indices}.")

    print(f"Done. Stored candidates in {args.out_dir}")

if __name__ == '__main__':
    main()
