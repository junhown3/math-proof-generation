"""CLI driver for generating single or batch theorem proofs.

Examples (PowerShell):
  python main.py --paper 2509.22618 --theorem 0 --model gpt-4o
  python main.py --paper 2509.22618 --batch --limit 3 --model gpt-4o
  python main.py --paper 2509.22618 --batch --backend mock

This uses OpenAI by default; set OPENAI_API_KEY beforehand or choose --backend mock.
"""

import argparse
from proof_agent_setup import ProofAgentConfig
from proof_agent import MathematicalProofAgent
from data_manager import DataManager


def build_client(backend: str, model: str, openai_api_key: str | None = None):
	if backend == 'openai':
		# Allow explicit API key override for convenience
		return ProofAgentConfig.create_openai_client(api_key=openai_api_key, model=model)
	elif backend == 'runpod':
		return ProofAgentConfig.create_runpod_client()
	elif backend == 'gemini':
		return ProofAgentConfig.create_gemini_client()
	elif backend == 'mock':
		return ProofAgentConfig.create_mock_client()
	else:
		raise ValueError(f"Unsupported backend: {backend}")


def main():
	parser = argparse.ArgumentParser(description="Mathematical Proof Agent Runner")
	parser.add_argument('--paper', required=True, help='arXiv paper id (e.g. 2509.22618)')
	group = parser.add_mutually_exclusive_group(required=False)
	group.add_argument('--theorem', type=int, help='Single theorem index to generate')
	group.add_argument('--batch', action='store_true', help='Generate multiple theorems (use --limit)')
	group.add_argument('--all', action='store_true', help='Generate proofs for ALL theorems in the parsed paper')
	parser.add_argument('--theorems', type=str, help='Comma-separated list of specific theorem indices to generate (e.g. 0,2,7)')
	parser.add_argument('--first-k', type=int, help='Generate the first K theorem indices (0..K-1)')
	parser.add_argument('--limit', type=int, default=5, help='Max theorems when using --batch')
	parser.add_argument('--backend', default='openai', choices=['openai','runpod','gemini','mock'], help='LLM backend to use')
	parser.add_argument('--model', default='gpt-4o', help='Model name for selected backend')
	parser.add_argument('--reasoning-effort', choices=['minimal','medium','high'], help='Set OpenAI reasoning effort for reasoning-capable models (e.g., gpt-5)')
	parser.add_argument('--results-dir', default='proof_results', help='Directory for JSON results')
	parser.add_argument('--offline', action='store_true', help='Offline mode: skip fetching; require existing saved paper or parsed file')
	parser.add_argument('--dump-context', action='store_true', help='Dump raw context and final prompt to files for inspection')
	parser.add_argument('--clean-dumps', action='store_true', help='Delete previously dumped prompt/context files before running')
	parser.add_argument('--export-pdf', action='store_true', help='After generation, export current (or batch) proofs to PDF using proof_pdf_exporter')
	parser.add_argument('--pdf-out-dir', default='proof_pdfs', help='Output directory for exported PDF/TeX')
	parser.add_argument('--rag', action='store_true', help='Enable retrieval-augmented context (experimental)')
	parser.add_argument('--rag-chunk-size', type=int, default=900, help='RAG chunk size in characters')
	parser.add_argument('--rag-overlap', type=int, default=150, help='Overlap between consecutive RAG chunks')
	parser.add_argument('--rag-top-k', type=int, default=8, help='Top-k retrieved chunks to include')
	parser.add_argument('--force-rag', action='store_true', help='Force RAG retrieval even if document is below small-doc threshold')
	parser.add_argument('--openai-api-key', type=str, help='Optional: supply OpenAI API key directly instead of environment variable')
	parser.add_argument('--ablate-rag', action='store_true', help='Run both baseline (no RAG) and RAG variants for comparison')
	args = parser.parse_args()

	# Manual selection validation: require at least one selection method
	if not any([
		args.theorem is not None,
		args.batch,
		args.all,
		args.theorems is not None,
		args.first_k is not None
	]):
		parser.error('Provide one of --theorem, --batch, --all, --theorems, or --first-k')

	client = build_client(args.backend, args.model, openai_api_key=args.openai_api_key)
	# Apply reasoning effort if OpenAI client and flag provided
	if args.backend == 'openai' and args.reasoning_effort:
		try:
			# Only OpenAIClient has set_reasoning_effort
			client.set_reasoning_effort(args.reasoning_effort)  # type: ignore
			print(f"Using reasoning effort: {args.reasoning_effort}")
		except Exception as e:
			print(f"Could not set reasoning effort: {e}")
	agent = MathematicalProofAgent(
		client,
		results_dir=args.results_dir,
		rag_enabled=args.rag,
		rag_chunk_size=args.rag_chunk_size,
		rag_overlap=args.rag_overlap,
		rag_top_k=args.rag_top_k
	)
	# Stash force_rag flag on agent for downstream use without altering constructor signature
	setattr(agent, 'force_rag', args.force_rag)
	if args.rag:
		print(f"RAG enabled: chunk_size={args.rag_chunk_size}, overlap={args.rag_overlap}, top_k={args.rag_top_k}")

	# Optional cleanup of prior dumps
	if args.clean_dumps:
		import glob, os
		patterns = [
			os.path.join(args.results_dir, '*_prompt.txt'),
			os.path.join(args.results_dir, '*_context.txt')
		]
		removed = 0
		for pat in patterns:
			for p in glob.glob(pat):
				try:
					os.remove(p)
					removed += 1
				except Exception:
					pass
		print(f"Cleaned {removed} previous dump file(s).")

	# In offline mode, ensure a paper object exists; if not saved, synthesize from parsed.
	if args.offline:
		dm = DataManager()
		paper_obj = dm.load_paper(args.paper)
		if not paper_obj:
			paper_obj = dm.load_parsed_only_as_paper(args.paper)
		if not paper_obj:
			raise SystemExit(f"Offline mode: could not find saved paper or parsed file for {args.paper}")
		# Save synthesized paper (optional) so downstream loads succeed uniformly
		if not dm.load_paper(args.paper):
			dm.save_paper(paper_obj)

	# Monkey-patch agent method to support dumping without refactoring whole class
	if args.dump_context:
		from functools import wraps
		orig_generate = agent.generate_proof_for_theorem
		def wrapped(paper_id, theorem_index):
			res = orig_generate(paper_id, theorem_index)
			# Save prompt/context dumps if available
			prompt_file = os.path.join(args.results_dir, f"{paper_id.replace('/', '_')}_theorem_{theorem_index}_prompt.txt")
			context_file = os.path.join(args.results_dir, f"{paper_id.replace('/', '_')}_theorem_{theorem_index}_context.txt")
			# Reconstruct prompt & context similar to generation path
			# NOTE: We re-run context prep to avoid storing giant strings in memory earlier.
			from data_manager import DataManager
			dm2 = DataManager()
			paper_obj2 = dm2.load_paper(paper_id)
			if paper_obj2:
				ctx_obj = agent.context_preparator.prepare_context(paper_obj2, theorem_index)
				prompt_rebuilt = agent.context_preparator.format_for_llm_prompt(ctx_obj)
				with open(prompt_file, 'w', encoding='utf-8') as pf:
					pf.write(prompt_rebuilt)
				with open(context_file, 'w', encoding='utf-8') as cf:
					cf.write(ctx_obj.paper_context)
				print(f"Dumped prompt/context for theorem {theorem_index}.")
			return res
		agent.generate_proof_for_theorem = wrapped  # type: ignore

	# Determine if custom subset run
	selected_indices = None
	if args.theorems:
		selected_indices = []
		for part in args.theorems.split(','):
			part = part.strip()
			if not part:
				continue
			try:
				selected_indices.append(int(part))
			except ValueError:
				raise SystemExit(f"Invalid theorem index in --theorems: '{part}'")
		# Remove duplicates and sort
		selected_indices = sorted(set(selected_indices))
		# We'll treat this like a mini-batch ignoring --theorem/--batch/--all exclusivity if provided
		print(f"Selected theorem indices: {selected_indices}")

	if selected_indices is not None:
		for idx in selected_indices:
			agent.generate_proof_for_theorem(args.paper, idx)
		if args.export_pdf:
			try:
				from proof_pdf_exporter import main as pdf_main
				import sys as _sys
				pdf_args = [
					'proof_pdf_exporter.py',
					'--paper', args.paper,
					'--indices', ','.join(str(i) for i in selected_indices),
					'--out-dir', args.pdf_out_dir
				]
				_sys.argv = pdf_args
				pdf_main()
			except Exception as e:
				print(f"PDF export failed for selection: {e}")
		return

	# Handle --first-k (takes precedence over batch/all/theorem if provided)
	if args.first_k is not None:
		if args.first_k <= 0:
			raise SystemExit('--first-k must be positive')
		# Determine how many theorems exist by reading parsed JSON
		import os, json
		parsed_file = f"data/parsed/{args.paper.replace('/', '_')}_parsed.json"
		if not os.path.exists(parsed_file):
			raise SystemExit('Parsed paper JSON not found; run ingestion first.')
		with open(parsed_file, 'r', encoding='utf-8') as f:
			pdata = json.load(f)
		count = len(pdata.get('theorems', []))
		k = min(args.first_k, count)
		indices_fk = list(range(k))
		print(f'Generating first {k} theorem(s): {indices_fk}')
		for idx in indices_fk:
			agent.generate_proof_for_theorem(args.paper, idx)
		if args.export_pdf:
			try:
				from proof_pdf_exporter import main as pdf_main
				import sys as _sys
				pdf_args = [
					'proof_pdf_exporter.py',
					'--paper', args.paper,
					'--indices', ','.join(str(i) for i in indices_fk),
					'--out-dir', args.pdf_out_dir
				]
				_sys.argv = pdf_args
				pdf_main()
			except Exception as e:
				print(f'PDF export failed for first-k: {e}')
		return

	if args.batch:
		agent.batch_generate_proofs(args.paper, max_theorems=args.limit)
		if args.export_pdf:
			# Export first 'limit' theorems (already generated) to PDF
			try:
				from proof_pdf_exporter import main as pdf_main
				import sys as _sys
				pdf_args = [
					'proof_pdf_exporter.py',
					'--paper', args.paper,
					'--batch',
					'--limit', str(args.limit),
					'--out-dir', args.pdf_out_dir
				]
				_sys.argv = pdf_args
				pdf_main()
			except Exception as e:
				print(f"PDF export failed: {e}")
	elif args.all:
		# Use a very large limit; batch_generate_proofs will cap at actual theorem count
		agent.batch_generate_proofs(args.paper, max_theorems=10**6)
		if args.export_pdf:
			try:
				from proof_pdf_exporter import main as pdf_main
				import sys as _sys
				pdf_args = [
					'proof_pdf_exporter.py',
					'--paper', args.paper,
					'--all',
					'--out-dir', args.pdf_out_dir
				]
				_sys.argv = pdf_args
				pdf_main()
			except Exception as e:
				print(f"PDF export failed: {e}")
	else:
		if args.ablate_rag and args.theorem is not None:
			# Run baseline (rag disabled) then rag variant
			print("Ablation: generating baseline (no RAG) variant...")
			original = (agent.rag_enabled, agent.rag_chunk_size, agent.rag_overlap, agent.rag_top_k)
			agent.rag_enabled = False
			agent.generate_proof_for_theorem(args.paper, args.theorem, variant_tag='baseline')
			# Restore RAG settings and run RAG variant
			agent.rag_enabled, agent.rag_chunk_size, agent.rag_overlap, agent.rag_top_k = original
			print("Ablation: generating RAG variant...")
			agent.generate_proof_for_theorem(args.paper, args.theorem, variant_tag='rag')
		else:
			agent.generate_proof_for_theorem(args.paper, args.theorem)
		if args.export_pdf:
			try:
				from proof_pdf_exporter import main as pdf_main
				import sys as _sys
				pdf_args = [
					'proof_pdf_exporter.py',
					'--paper', args.paper,
					'--theorem', str(args.theorem),
					'--out-dir', args.pdf_out_dir
				]
				_sys.argv = pdf_args
				pdf_main()
			except Exception as e:
				print(f"PDF export failed: {e}")


if __name__ == '__main__':
	main()
