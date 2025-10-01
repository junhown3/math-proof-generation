"""Minimal end-to-end scalar RLAIF pipeline orchestrator.

Focus: TinyLlama (1.1B) sanity run on a small batch of recent math.NT (Number Theory) papers.
This script stitches existing components (fetch -> parse -> candidate generation (single RAG variant)
-> judge (stub or real) -> reward dataset -> optional weighted fine-tune -> scalar stats).

Design goals:
 - Idempotent and incremental: reuses existing artifacts; skips steps if outputs exist unless --force.
 - Minimal external deps beyond those already in repository (uses current scripts via subprocess).
 - Safe for ephemeral RunPod storage: all outputs under data/ and models/ subdirs.

Current intentional limitations:
 - Only supports using already-downloaded papers (assumes separate ingestion step) OR will fetch a
   small set of recent math.NT papers if --auto-fetch is provided.
 - Single paper + theorem range by default to keep token + time low for a smoke test.
 - Generates ONE candidate per theorem (RAG variant) for fast loop; can be expanded later.
 - Judge defaults to stub scoring unless --real-judge is passed.

Typical first smoke run (PowerShell):
  python -m rl.pipeline_scalar_loop `
    --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
    --paper-limit 1 --theorem-range 0-2 --auto-fetch `
    --run-train --epochs 1 --output-dir models/tinyllama-scalar-smoke

Then inspect reward stats:
  python -m rl.evaluate_scalar --reward-file data/rl/reward_dataset.jsonl

Later (real judge step), add:
  --real-judge --judge-model gpt-4o --backend openai

NOTE: Candidate generation presently uses the OpenAI or mock backend via candidate_generation.py.
For a purely local base model (TinyLlama) you typically would not call OpenAI for generation.
This script includes a basic direct HuggingFace generation path (--local-generation) that uses
TinyLlama to create candidate proofs without external APIs, bypassing candidate_generation.py.
The local path implements a simplified single-theorem prompt builder using proof-blind context.

Future extensions:
 - Add caching of retrieval chunks per theorem.
 - Multi-paper batching + held-out split.
 - Structural penalty fusion into reward shaping.
"""
from __future__ import annotations
import argparse, os, json, subprocess, sys, time, textwrap
from pathlib import Path
from typing import List

from data_manager import DataManager
from latex_parser import LatexParser
from context_preparator import ContextPreparator
from proof_agent_setup import ProofAgentConfig
from proof_agent import MathematicalProofAgent, MockLLMClient


def log(msg: str):
    print(f"[pipeline] {msg}")


def run(cmd: List[str]):
    log("EXEC: " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log("STDOUT:\n" + result.stdout)
        log("STDERR:\n" + result.stderr)
        raise SystemExit(f"Command failed: {' '.join(cmd)}")
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            log(line)
    return result


def ensure_papers(math_subjects: List[str], limit: int, force: bool, auto_fetch: bool, reuse_existing: bool) -> List[str]:
    dm = DataManager()
    papers = dm.list_saved_papers()
    if papers and reuse_existing and not force and not auto_fetch:
        # Return first N existing papers directly
        return [p['arxiv_id'] for p in papers[:limit]]
    if not auto_fetch:
        if not papers:
            raise SystemExit("No papers present. Run ingestion or pass --auto-fetch.")
        return [p['arxiv_id'] for p in papers[:limit]]
    # Auto fetch path
    try:
        from arxiv_fetcher import ArxivFetcher
    except ImportError:
        raise SystemExit("arxiv_fetcher module missing; cannot --auto-fetch.")
    fetcher = ArxivFetcher()
    parser = LatexParser()
    log(f"Auto-fetching up to {limit} recent papers for subjects={math_subjects}...")
    from datetime import datetime, timedelta
    start_date = datetime.utcnow() - timedelta(days=14)
    new_papers = fetcher.search_papers(math_subjects, start_date, max_results=limit)
    ids = []
    for p in new_papers:
        if not fetcher.download_latex_source(p):
            continue
        dm.save_paper(p)
        parsed = parser.parse(p.latex_content)
        dm.save_parsed_content(p.arxiv_id, parsed)
        ids.append(p.arxiv_id)
    if not ids:
        raise SystemExit("Failed to fetch any new papers.")
    return ids


def parse_range(expr: str) -> List[int]:
    out = []
    for part in expr.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def local_generate_candidates(paper_id: str, theorem_indices: List[int], model_name: str, out_dir: str, force: bool):
    """Local HuggingFace generation path (TinyLlama) without external API.
    Creates one RAG-enabled candidate per theorem index.
    """
    os.makedirs(out_dir, exist_ok=True)
    dm = DataManager()
    paper = dm.load_paper(paper_id)
    if not paper:
        raise SystemExit(f"Paper {paper_id} not found for local generation.")
    parser = LatexParser()
    parsed = parser.parse(paper.latex_content or '')
    theorems = parsed.get('theorems', [])
    if not theorems:
        log(f"[warn] Paper {paper_id} contains zero parsed theorems. Aborting candidate generation.")
        return
    ctx_prep = ContextPreparator()
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        raise SystemExit("transformers + torch required for --local-generation path.")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')

    for ordinal, th_i in enumerate(theorem_indices):
        # NOTE: We previously attempted a glob with pattern "*::{th_i}::*" to detect
        # existing candidates by embedded theorem index inside hashed IDs. This fails
        # on Windows (colon usage) and is unreliable because IDs are hashes. We now
        # simply scan JSON files and look at stored meta.theorem_index.
        if not force:
            exists = False
            for fp in Path(out_dir).glob('*.json'):
                try:
                    data = json.loads(fp.read_text(encoding='utf-8'))
                    meta = data.get('meta', {})
                    if meta.get('paper_id') == paper_id and meta.get('theorem_index') == th_i:
                        exists = True
                        break
                except Exception:
                    continue
            if exists:
                log(f"Skipping local generation for theorem {th_i} (exists). Use --force to regenerate.")
                continue
        if th_i >= len(theorems):
            log(f"[warn] theorem index {th_i} out of range ({len(theorems)})")
            continue
        # Build context (RAG variant only)
        # For now we force rag_enabled=False to keep minimal; retrieval can be toggled later
        ctx = ctx_prep.prepare_context(paper, th_i, rag_enabled=True, rag_chunk_size=900, rag_overlap=150, rag_top_k=6)
        prompt = ctx_prep.format_for_llm_prompt(ctx)
        input_ids = tok(prompt, return_tensors='pt', truncation=True, max_length=2048).input_ids.to(model.device)
        gen = model.generate(input_ids, max_new_tokens=512, temperature=0.3, do_sample=True)
        text = tok.decode(gen[0], skip_special_tokens=True)
        # Heuristic: keep only completion portion after prompt tail
        generated = text[len(prompt):].strip() or text
        from rl.schema import Candidate, CandidateMeta
        meta = CandidateMeta(
            paper_id=paper_id, theorem_index=th_i, variant='rag', model=model_name,
            temperature=0.3, seed=0, rag_enabled=True, rag_top_k=6, rag_chunk_size=900, rag_overlap=150,
            prompt_char_len=len(prompt), context_char_len=len(ctx.paper_context), provenance=ctx_prep.get_last_provenance()
        )
        cid = Candidate.make_id(paper_id, th_i, 'rag', 0, ordinal)
        cand = Candidate(cid, meta, theorems[th_i].statement, generated)
        with open(os.path.join(out_dir, f"{cid}.json"), 'w', encoding='utf-8') as f:
            f.write(cand.to_json())
        log(f"Local candidate generated for theorem {th_i} -> {cid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', default='math.NT', help='Comma list of arXiv subjects (only used with --auto-fetch).')
    ap.add_argument('--paper-limit', type=int, default=1)
    ap.add_argument('--paper', help='Explicit paper id to reuse (must be already ingested).')
    ap.add_argument('--theorem-range', default='0-2', help='Inclusive range/list notation for theorem indices.')
    ap.add_argument('--model-name', default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    ap.add_argument('--candidates-dir', default='data/rl/candidates')
    ap.add_argument('--scores-file', default='data/rl/judge_scores.jsonl')
    ap.add_argument('--reward-file', default='data/rl/reward_dataset.jsonl')
    ap.add_argument('--output-dir', default='models/scalar-ft-tinyllama')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--run-train', action='store_true')
    ap.add_argument('--real-judge', action='store_true')
    ap.add_argument('--judge-model', default='frontier-judge-stub')
    ap.add_argument('--backend', default='openai')
    ap.add_argument('--auto-fetch', action='store_true')
    ap.add_argument('--force', action='store_true', help='Force regeneration of candidates')
    ap.add_argument('--local-generation', action='store_true', help='Use local HF model instead of candidate_generation script (one RAG candidate per theorem).')
    ap.add_argument('--reuse-existing', action='store_true', help='Prefer already-saved papers over fetching new ones (ignores --subjects unless --auto-fetch).')
    args = ap.parse_args()

    if args.paper:
        # Validate paper presence
        if not DataManager().load_paper(args.paper):
            raise SystemExit(f"Specified --paper {args.paper} not found. Ingest it first or omit --paper.")
        paper_id = args.paper
    else:
        papers = ensure_papers(args.subjects.split(','), args.paper_limit, force=args.force, auto_fetch=args.auto_fetch, reuse_existing=args.reuse_existing)
        if not papers:
            raise SystemExit('No papers available after ensure_papers step.')
        paper_id = papers[0]  # minimal loop currently single paper
    theorem_indices = parse_range(args.theorem_range)
    os.makedirs(args.candidates_dir, exist_ok=True)
    # Quick sanity: if paper has zero theorems, exit gracefully before downstream steps
    dm_check = DataManager()
    p_obj = dm_check.load_paper(paper_id)
    parser_check = LatexParser()
    parsed_check = parser_check.parse(p_obj.latex_content or '') if p_obj else {'theorems': []}
    if not parsed_check.get('theorems'):
        log(f"Paper {paper_id} has zero theorems; skipping pipeline. (Try a different paper or broader subject.)")
        return

    # 1. Candidate generation
    if args.local_generation:
        log('Generating candidates locally with HuggingFace model...')
        local_generate_candidates(paper_id, theorem_indices, args.model_name, args.candidates_dir, force=args.force)
    else:
        # Use existing candidate_generation script for RAG-only single variant if not already present
        for th in theorem_indices:
            existing = False
            if not args.force:
                for fp in Path(args.candidates_dir).glob('*.json'):
                    try:
                        data = json.loads(fp.read_text(encoding='utf-8'))
                        if data.get('meta', {}).get('paper_id') == paper_id and data.get('meta', {}).get('theorem_index') == th:
                            existing = True
                            break
                    except Exception:
                        pass
            if existing:
                log(f"Candidate already exists for theorem {th}; skipping (use --force to regenerate).")
                continue
            cmd = [sys.executable, '-m', 'rl.candidate_generation', '--paper', paper_id, '--theorems', str(th),
                   '--backend', 'mock', '--variants', 'rag', '--temperatures', '0.1', '--seeds', '0', '--out-dir', args.candidates_dir]
            run(cmd)

    # 2. Judge scoring
    log('Scoring candidates...')
    judge_cmd = [sys.executable, '-m', 'rl.judge_scores', '--paper', paper_id, '--theorems', ','.join(map(str, theorem_indices)),
                 '--candidates-dir', args.candidates_dir, '--out', args.scores_file, '--skip-existing']
    if args.real_judge:
        judge_cmd += ['--real-judge', '--judge-model', args.judge_model, '--backend', args.backend]
    run(judge_cmd)

    # 3. Build reward dataset
    log('Building reward dataset...')
    run([sys.executable, '-m', 'rl.build_reward_dataset', '--candidates-dir', args.candidates_dir, '--scores-file', args.scores_file,
         '--paper', paper_id, '--theorems', ','.join(map(str, theorem_indices)), '--out', args.reward_file])

    # Detect empty reward file (no useful data) before training
    reward_count = 0
    try:
        with open(args.reward_file, 'r', encoding='utf-8') as rf:
            for _ in rf:
                reward_count += 1
    except FileNotFoundError:
        pass
    if reward_count == 0:
        log('No reward rows produced; skipping training and evaluation summary.')
        return

    # 4. Optional training
    if args.run_train:
        # Preflight dependency availability
        try:
            import transformers, peft  # noqa: F401
        except ImportError:
            log('transformers or peft not installed; skipping training step. Install requirements first.')
        else:
            log('Running weighted scalar fine-tune...')
            train_cmd = [sys.executable, '-m', 'rl.train_scalar_weighted', '--model-name', args.model_name, '--reward-file', args.reward_file,
                         '--candidates-dir', args.candidates_dir, '--output-dir', args.output_dir, '--epochs', str(args.epochs), '--batch-size', '2', '--gradient-accumulation', '8']
            run(train_cmd)

    # 5. Reward stats summary (lightweight)
    log('Reward dataset summary:')
    run([sys.executable, '-m', 'rl.evaluate_scalar', '--reward-file', args.reward_file, '--bucket', '0.5', '--top-k', '3'])

    log('Pipeline completed successfully.')


if __name__ == '__main__':
    main()
