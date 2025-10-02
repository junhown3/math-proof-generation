# Research-Level Mathematical Proof Generation with RAG & Planned RLAIF

## Overview
This project builds an end-to-end pipeline for automated mathematical theorem proof generation directly from arXiv LaTeX sources. It ingests a paper, parses and normalizes its structure, constructs a context (optionally via Retrieval-Augmented Generation), and prompts reasoning-capable LLMs to produce:
1. A **Proof Sketch** (strategic outline)
2. A **Full Proof** (best-effort rigorous derivation) including explicit `GAP:` lines for unresolved steps instead of refusal.

A forthcoming alignment loop (RLAIF: Reinforcement Learning from AI Feedback) will fine‑tune open-source models (e.g. GPT‑OSS variants) using an LLM-as-judge scoring system comparing generated proofs to canonical proofs from the paper.

## Key Features (Implemented Phase)
- **ArXiv LaTeX Ingestion & Parsing**: Extracts abstract, introduction, sections, theorems / lemmas, references; strips original proofs.
- **Statement Normalization**: Enumerated multi-part theorems split; equation environments unified to `\\[ ... \\]` form; stray `$$` sanitized.
- **Context Preparation**: Metadata + introduction + (sections or retrieved chunks) + available results + references; provenance tracked (chars, truncation flags).
- **RAG (Phase 1)**: Overlapping chunker + lightweight lexical (TF–IDF-like) index → top-k relevant chunks; provenance logs chunk ids, spans, scores.
- **GAP-Aware Prompting**: Encourages partial rigorous progress; disallows empty disclaimers; separates Proof Sketch vs Full Proof.
- **Multi-Backend Architecture**: OpenAI (with reasoning effort handling), mock backend, planned GPT‑OSS integration.
- **Robust Output Pipeline**: Structured JSON results (quality metrics, provenance) and optional PDF export of proofs.
- **Resilience Mechanisms**: Retry logic for incomplete reasoning traces; parameter stripping for incompatibilities; success heuristic with GAP allowances.

## Planned / In-Progress (Alignment & Retrieval Enhancements)
| Area | Upcoming Work |
|------|---------------|
| Retrieval | Query normalization, small-doc bypass, adaptive top-k pruning, hybrid lexical + embeddings |
| Alignment (RLAIF) | LLM judge scoring (Rigor, Completeness, Cohesion, Alignment, Hallucination Penalty, GAP Balance) → reward shaping → LoRA fine‑tune |
| Proof Quality | Lemma dependency graph, targeted lemma recall when GAP density high |
| Verification | Symbolic sanity checks (algebraic identities), hallucination detection heuristics |
| Efficiency | Chunk/index caching, prompt length reduction metrics dashboard |
| Curriculum | Progression: lemmas → main theorems; difficulty-aware sampling |

## RLAIF Reward Sketch
Proposed reward (scaled 0–10 per dimension):
```
R = 0.28*Rigor + 0.22*Completeness + 0.18*Cohesion + 0.18*GroundTruthAlignment 
    - 0.08*Hallucination - 0.06*GAP_Overuse
```
Where GAP_Overuse penalizes diffuse or excessive gaps while permitting a small number of well-localized ones.

## Repository Structure (Excerpt)
```
context_preparator.py   # Builds final context (RAG or full sections) + provenance
latex_parser.py         # Extracts structured content from LaTeX
rag_chunker.py          # Overlapping chunk generation
rag_index.py            # Lexical retrieval index
proof_agent.py          # Orchestrates proof generation, evaluation & saving
main.py                 # CLI driver (single theorem / batch)
proof_pdf_exporter.py   # Optional LaTeX/PDF export of proofs
proof_results/          # Generated JSON proof outputs (created at runtime)
```

## Quick Start
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run a single theorem (mock backend):
```bash
python main.py --paper mock_paper_id --theorem 0 --backend mock --rag
```
(If offline real paper parsing already done, use a real arXiv ID.)

3. Batch mode (first 3 theorems):
```bash
python main.py --paper YOUR_ARXIV_ID --batch --limit 3 --backend openai
```

4. Export PDF after generation:
```bash
python main.py --paper YOUR_ARXIV_ID --theorem 0 --export-pdf
```

## Example Output Snippet (Truncated)
```
### Proof Sketch
We reduce to growth of iterated sumsets ...

### Proof
By Lemma 1 we have |A + A| ≥ 2|A| - 1. Next we iterate ...
GAP: Need validated bound linking |2A| to |3A| via Plünnecke inequality.
...
∎
```

## Provenance JSON (Illustrative)
```
"quality": {
  "prompt_char_len": 14230,
  "context_char_len": 11840,
  "provenance": {
    "sections": [
      {"title": "Introduction", "included_chars": 742, "truncated": false}
    ],
    "rag": {
      "enabled": true,
      "total_chunks": 62,
      "top_k": 8,
      "selected": [ {"id":5, "score":0.4123}, {"id":12, "score":0.3710} ]
    }
  }
}
```

## Roadmap (Next Milestones)
- [ ] Query normalization & hybrid retrieval
- [ ] RAG caching layer & coverage metrics
- [ ] LLM judge rubric implementation
- [ ] Reward-weighted fine-tune on GPT‑OSS baseline
- [ ] Curriculum scheduling + difficulty tagging
- [ ] Symbolic verification stubs

## Contributing / Extension Ideas
- Add embedding backend (e.g. e5-small) for hybrid scoring.
- Plug in an SMT solver or CAS for micro-verification of algebraic steps.
- Add UI dashboard (Streamlit) for interactive exploration of proofs + gaps.

## License
MIT (proposed) – confirm before publication.

## Disclaimer
Generated proofs may contain inaccuracies or unjustified steps; GAP lines are explicit placeholders for missing rigor. Alignment phase not yet executed—reward design subject to iteration.

---
*Future updates will document empirical impact once RLAIF loop is deployed.*

## RLAIF (Reinforcement Learning from AI Feedback) – Initial Workflow

This repository now includes a minimal end-to-end alignment loop that converts multi‑candidate proof generations into pairwise preferences and fine‑tunes an open‑source model via Direct Preference Optimization (DPO).

### 1. Generate Candidates
Produce multiple proof attempts (baseline + optional RAG) per theorem:
```
python -m rl.candidate_generation \
  --paper 2509.22618 --theorem 0 \
  --backend openai --model gpt-5 \
  --samples 6 --temperatures 0.1,0.3,0.6 --seeds 42,43 \
  --with-rag
```
Outputs: `data/rl/candidates/*.json`

### 2. Judge Scoring (LLM-as-a-Judge)
Currently uses a stub scorer; replace later with a strong model prompt.
```
python -m rl.judge_scores --candidates-dir data/rl/candidates --out data/rl/judge_scores.jsonl
```

### 3. Build Pairwise Preference Dataset
Converts scalar scores into (chosen, rejected) pairs:
```
python -m rl.build_pairwise_preferences \
  --candidates-dir data/rl/candidates \
  --judge-scores data/rl/judge_scores.jsonl \
  --out data/rl/pairwise.jsonl
```
Each line has: prompt, chosen, rejected, score_margin, variants.

### 4. Train with DPO (LoRA / QLoRA)
Install RL deps first (`pip install -r requirements.txt`). Then:
```
python -m rl.train_dpo \
  --model-name meta-llama/Llama-2-7b-hf \
  --pairwise-jsonl data/rl/pairwise.jsonl \
  --output-dir models/dpo-llama7b-v1 \
  --max-steps 300 --batch-size 2 --gradient-accumulation 8 \
  --lora-r 16 --lora-alpha 32 --beta 0.1 --four-bit --bf16
```

### 5. (Planned) Evaluation Script
Regenerate proofs with the fine‑tuned adapter & re-score to measure uplift in overall judge score and structural metrics (pass@k, hallucination risk shift). Not yet implemented—placeholder.

### Preference Generation Strategy
- Always includes (best, worst) if ≥2 candidates.
- Generates adjacent pairs whose margin ≥ 0.2 (configurable).
- Skips proofs < 50 chars to avoid noise.

### Extending the Judge
Replace stub with frontier model prompt: provide canonical proof (if available), candidate proof, rubric, request JSON. Integrate an automatic JSON parse & fallback.

### Safety / Quality Checks
- Structural validators already attach symbol and lemma signals.
- You can incorporate these into pairwise filtering (e.g., discard candidates with extremely high undefined symbol counts) before DPO.

### Next Improvements
- Tiered token coverage in RAG for more selective fallback.
- Adaptive pair sampling (e.g., stratify by margin buckets).
- Reward shaping: combine scalar (overall) + structural penalty before pairwise derivation.
- Add evaluation harness to compare pretrained vs fine‑tuned variants automatically.

## Scalar Reward Weighted Fine-Tune (Alternative to Pairwise/DPO)

In addition to DPO, you can directly optimize on scalar judge scores:

1. Generate candidates (same as DPO workflow).
2. Judge with `rl/judge_scores.py`.
3. Build reward dataset:
  ```
  python -m rl.build_reward_dataset --candidates-dir data/rl/candidates --scores-file data/rl/judge_scores.jsonl --out data/rl/reward_dataset.jsonl
  ```
4. (Optional) Evaluate distribution:
  ```
  python -m rl.evaluate_scalar --reward-file data/rl/reward_dataset.jsonl --bucket 0.5 --top-k 5
  ```
5. Train weighted SFT (scalar):
  ```
  python -m rl.train_scalar_weighted \
    --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --reward-file data/rl/reward_dataset.jsonl \
    --output-dir models/scalar-ft-v1 \
    --epochs 1 --batch-size 2 --gradient-accumulation 8 --lr 2e-5 --lora-r 16 --lora-alpha 32
  ```

Mechanics: Each example's LM loss is multiplied by a normalized reward weight (mapped to [0.1,1.0]). This approximates importance weighting—higher judged quality steers gradients more strongly without constructing preferences.

When to use:
- Rapid prototype if pairwise set is too small.
- Mixing in with DPO (first weighted SFT to warm start, then DPO for sharper preference shaping).

Limitations:
- Susceptible to reward scale drift; ensure consistent judge model & rubric.
- Does not enforce ordering constraints like DPO; may under-utilize relative quality signal.

Future Extensions:
- Reward normalization per-theorem (to prevent theorem-level reward imbalance).
- Incorporate structural penalty (e.g., undefined symbols) into adjusted reward.
- Two-stage curriculum: weighted SFT -> DPO -> small RLHF-style PPO (optional).

## 🚀 Minimal Scalar Pipeline Orchestrator (TinyLlama Smoke Test)

New script: `rl/pipeline_scalar_loop.py` provides a one-command minimal loop:
1. (Optional) Auto-fetch recent `math.NT` paper(s)
2. Generate one RAG variant candidate per theorem index (local TinyLlama or mock backend)
3. Judge (stub by default or real via OpenAI)
4. Build reward dataset
5. (Optional) Run weighted scalar fine‑tune
6. Summarize reward distribution

### Quick Local Dry Run (Stub Judge, Mock Generation)
```
python -m rl.pipeline_scalar_loop --auto-fetch --paper-limit 1 --theorem-range 0-1 \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-generation --run-train --epochs 1
```

### Using a Real Judge (e.g., OpenAI gpt-4o)
Set environment first (PowerShell):
```
$env:OPENAI_API_KEY="sk-..."
```
Then:
```
python -m rl.pipeline_scalar_loop --auto-fetch --paper-limit 1 --theorem-range 0-2 \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-generation \
  --real-judge --judge-model gpt-4o --backend openai --run-train --epochs 1
```

### RunPod H100 Ephemeral Session Guidelines
When launching a RunPod H100 instance (ephemeral storage):
1. Clone repo inside `/workspace` (or mounted volume) and immediately create a persistent volume if available.
2. Install dependencies:
  ```
  pip install -r requirements.txt --upgrade
  ```
3. (Optional) Cache TinyLlama locally to reduce repeated downloads:
  ```
  python - <<EOF
import transformers, os
transformers.AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
transformers.AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
print('Cached TinyLlama')
EOF
  ```
4. Execute minimal loop (single paper, few theorems):
  ```
  python -m rl.pipeline_scalar_loop --auto-fetch --paper-limit 1 --theorem-range 0-2 \
    --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-generation --run-train --epochs 1
  ```
5. Inspect outputs:
  - Candidates: `data/rl/candidates/*.json`
  - Scores: `data/rl/judge_scores.jsonl`
  - Rewards: `data/rl/reward_dataset.jsonl`
  - Model (LoRA or full): `models/scalar-ft-tinyllama/`

### Dependency & Platform Notes (Windows)
- Ensure you actually install the full training stack before `--run-train`:
  ```powershell
  pip install -r requirements.txt
  ```
- `bitsandbytes` GPU quantization often fails on Windows; if issues arise you can remove it from `requirements.txt` or install a CPU-friendly fork, then rerun without `--four-bit` flags.
- If `transformers` / `peft` are missing, the orchestrator now skips training gracefully and logs a message instead of crashing.

### Zero Theorem Fetches
Some recent `math.NT` papers may parse to zero theorems (e.g., due to unconventional environments). If you see:
```
[pipeline] Paper <id> has zero theorems; skipping pipeline.
```
Try increasing the date window or fetching more papers:
```
python -m rl.pipeline_scalar_loop --auto-fetch --paper-limit 3 --theorem-range 0-2 --local-generation
```
The pipeline will stop early rather than proceeding with empty candidates.

### Flags Summary
| Flag | Purpose |
|------|---------|
| `--auto-fetch` | Pull recent papers (subjects from `--subjects`) |
| `--local-generation` | Use local HF model (TinyLlama) instead of API/mock generator script |
| `--real-judge` | Switch from stub heuristic to real model judging |
| `--run-train` | Perform reward-weighted fine‑tune after reward build |
| `--force` | Regenerate candidates even if they already exist |

### Incremental Expansion Strategy
1. Start: 1 paper × 2–3 theorems × 1 candidate/theorem (RAG) → verify pipeline integrity.
2. Scale theorems: increase `--theorem-range` (e.g., `0-5`).
3. Introduce multiple variants (baseline vs RAG) via `candidate_generation.py` if diversity is low.
4. Move to larger base model (e.g., 3B–8B) before enabling real judge to control costs.
5. Add held-out theorems not seen during fine‑tune for evaluation.

If ephemeral session ends, re-run with `--force` only if you intentionally want fresh generations; otherwise reuse existing JSON artifacts by copying them to persistent storage between sessions.


### Reusing Previously Parsed Papers (No Refetch)
List existing ingested papers and their theorem counts:
```
python -m rl.list_papers_with_theorems --min-theorems 1 --limit 20
```
Pick an arXiv ID (e.g., `2509.12345`) and run the pipeline without auto-fetch:
```
python -m rl.pipeline_scalar_loop --paper 2509.12345 --theorem-range 0-3 --local-generation --reuse-existing
```
If you want both baseline (no RAG) and RAG variants for variance, generate additional candidates using the existing generator before judging:
```
python -m rl.candidate_generation --paper 2509.12345 --theorems 0-3 --backend mock --variants baseline,rag --temperatures 0.1 --seeds 0
python -m rl.judge_scores --paper 2509.12345 --theorems 0-3 --candidates-dir data/rl/candidates --out data/rl/judge_scores.jsonl --skip-existing
python -m rl.build_reward_dataset --paper 2509.12345 --theorems 0-3
```
Then optionally fine‑tune:
```
python -m rl.train_scalar_weighted --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --reward-file data/rl/reward_dataset.jsonl --output-dir models/reuse-ft --epochs 1
```


## Canonical Proof Integration for Judging

The term *canonical proof* refers to the original paper proof text extracted from LaTeX and stored in `data/parsed/<paper_id>_parsed.json` under the `theorems` list (each theorem may include a `proof` string). These are now leveraged directly by the judging script so GPT‑5 (or other judge models) evaluates model outputs against the authentic source proof rather than heuristics or incomplete context.

### How It Works
1. Parse papers with `latex_parser.py` (already done if you see files in `data/parsed/`).
2. Candidate generation produces JSON lines records that *may* include `canonical_proof` (depending on builder script / earlier pipeline stage).
3. During judging, if a candidate lacks `canonical_proof` and you pass `--recover-canonical`, the script loads the parsed JSON and fuzzy-matches the theorem statement to locate the correct proof.

### Updated Judge Usage
```
python rl/openai_judge.py \
  --candidates-full rl/out_openai_ctx/candidates_full.jsonl \
  --candidates-minimal rl/out_openai_ctx/candidates_minimal.jsonl \
  --out-scores-full rl/out_openai_ctx/scores_full.jsonl \
  --out-scores-minimal rl/out_openai_ctx/scores_minimal.jsonl \
  --out-report rl/out_openai_ctx/report.txt \
  --model gpt-5 \
  --recover-canonical \
  --parsed-dir data/parsed
```

### Flags
| Flag | Purpose |
|------|---------|
| `--recover-canonical` | Attempt to load proof text from parsed JSON if missing in candidate record |
| `--allow-missing-canonical` | Judge even if canonical proof cannot be found (default is to skip) |
| `--parsed-dir` | Directory with parsed paper JSON (default `data/parsed`) |

### Matching Heuristic
Simple whitespace normalization + containment / equality scoring of the theorem statement (length-capped for speed). The highest scoring match supplies the canonical proof.

### When Proofs Are Missing
If you see warnings such as:
```
[warn] no canonical proof for 2509.XXXX_th3, skipping (use --recover-canonical or --allow-missing-canonical)
```
Either enable recovery or re-run parsing to ensure proofs were captured. Some LaTeX environments (custom names) may need to be added to `THEOREM_ENVIRONMENTS` or `PROOF_ENVIRONMENTS` in `latex_parser.py`.

### Dataset Construction with Guaranteed Proofs
To produce a dataset containing only theorems with already attached canonical proofs:
```
python rl/build_openai_dataset.py --parsed-dir data/parsed --out-jsonl data/rl/theorems_with_proofs.jsonl --require-canonical
```

This ensures downstream processes (generation / judging) always have reference proofs without on-the-fly recovery.

### Rationale
Using the original proof increases evaluation fidelity, penalizes hallucinated but incorrect reasoning more accurately, and enables dimension-specific metrics (correctness, rigor, completeness) to align with the paper’s actual argument structure.

---


