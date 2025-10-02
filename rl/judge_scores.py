"""Judge scoring script.

Reads candidate JSON files, constructs a judge prompt (using canonical proof if available),
and queries a judge LLM. Two modes:

1. Stub mode (default): deterministic pseudo-random rubric scores for pipeline smoke tests.
2. Real judge mode (--real-judge): expects the external model to return ONLY a JSON object
    following the schema in `judge_prompt_template.md`.

Robust JSON parsing with fallback heuristics:
 - Strips code fences if present.
 - Attempts to locate the first '{' and last '}' to isolate JSON.
 - On parse failure, falls back to stub scoring while annotating rationale.
"""
from __future__ import annotations
import argparse, os, json, glob, hashlib, sys, pathlib
# Ensure project root on path when executed directly
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from typing import Optional, Dict, Any, Tuple, List, Set
import time
from rl.schema import JudgeScore, RUBRIC_DIMENSIONS, aggregate_rubric
from data_manager import DataManager
from latex_parser import LatexParser

def _stub_scores(prompt: str) -> Dict[str, Any]:
    h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
    scores: Dict[str, float] = {}
    for i, dim in enumerate(RUBRIC_DIMENSIONS):
        scores[dim] = 4 + (h >> (i * 8) & 0xF) / 3.0
    overall = aggregate_rubric(scores)
    return {
        "scores": scores,
        "overall": overall,
        "rationale": "stub scoring fallback",
        "canonical_used": False
    }

def _extract_json(text: str) -> Optional[str]:
    if not text:
        return None
    # Remove code fences
    if '```' in text:
        parts = [p for p in text.split('```') if '{' in p and '}' in p]
        if parts:
            text = parts[0]
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end+1]

def call_real_judge(prompt: str, model: str, backend: str, max_retries: int = 3, sleep: float = 2.0,
                    temperature: Optional[float] = None, max_output: Optional[int] = None) -> Dict[str, Any]:
    """Invoke a real judge backend.

    Currently supports:
      backend == 'openai': uses OPENAI_API_KEY env var.
    Returns parsed JSON dict (already validated upstream). If the remote model returns
    raw text, we attempt to parse later.
    """
    backend = backend.lower()
    if backend == 'openai':
        print('[debug] Entering call_real_judge backend=openai', flush=True)
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError('Missing OPENAI_API_KEY environment variable.')
        try:
            from openai import OpenAI
        except ImportError:  # Provide actionable guidance
            raise RuntimeError('openai package not installed. pip install openai')
        client = OpenAI(api_key=api_key)
        last_err = None
        for attempt in range(max_retries):
            try:
                print(f'[debug] openai attempt={attempt+1}', flush=True)
                params = {
                    'model': model,
                    'messages': [{"role": "user", "content": prompt}]
                }
                if temperature is not None:
                    params['temperature'] = temperature
                if max_output is not None:
                    # try common names; server will reject unknown ones
                    params['max_tokens'] = max_output
                try:
                    resp = client.chat.completions.create(**params)
                except Exception as eparam:
                    print(f'[debug] openai param error: {eparam}', flush=True)
                    # Adaptive retry: strip optional params if error references them
                    lower_msg = str(eparam).lower()
                    if any(k in lower_msg for k in ['temperature', 'max_tokens', 'max_output']):
                        params.pop('temperature', None)
                        params.pop('max_tokens', None)
                        resp = client.chat.completions.create(**params)
                    else:
                        raise
                content = resp.choices[0].message.content
                print('[debug] openai success raw length', len(content) if content else 0, flush=True)
                return {"raw": content}
            except Exception as e:  # retry transient errors
                print(f'[debug] openai exception attempt {attempt+1}: {e}', flush=True)
                last_err = e
                time.sleep(sleep * (attempt + 1))
        raise RuntimeError(f'OpenAI judge failure after retries: {last_err}')
    else:
        raise RuntimeError(f'Unsupported backend: {backend}')

def judge(prompt: str, model: str, real: bool, backend: str, temperature: Optional[float], max_output: Optional[int]) -> Tuple[Dict[str, Any], bool]:
    """Return (result_dict, used_fallback)."""
    if not real:
        return _stub_scores(prompt), False
    try:
        raw = call_real_judge(prompt, model, backend=backend, temperature=temperature, max_output=max_output)
        # If backend returns raw text, attempt JSON extraction here
        if 'raw' in raw:
            js_fragment = _extract_json(raw['raw'])
            if js_fragment:
                parsed = json.loads(js_fragment)
                return parsed, False
            raise RuntimeError('Real judge returned non-JSON content.')
        return raw, False
    except Exception as e:
        # Attempt to parse JSON if the exception carried text (e.g., model raw output)
        msg = str(e)
        parsed = None
        js_fragment = _extract_json(msg)
        if js_fragment:
            try:
                parsed = json.loads(js_fragment)
            except Exception:
                parsed = None
        if parsed and 'scores' in parsed and 'overall' in parsed:
            parsed.setdefault('rationale', 'parsed from exception text')
            return parsed, True
        fb = _stub_scores(prompt)
        fb['rationale'] += f"; real judge failure: {type(e).__name__}: {e}"[:240]
        fb['fallback_reason'] = 'real_judge_error'
        return fb, True

JUDGE_PROMPT_HEADER = (
    "System: You are an expert mathematician and meticulous proof reviewer.\n"
    "Return ONLY a JSON object with keys: scores (dimension->1-10), overall (1-10 float), rationale (string), canonical_used (bool).\n"
    "Scoring emphasizes correctness > completeness > cohesion > retrieval_grounding > conciseness.\n"
)

def build_prompt(theorem: str, candidate_proof: str, canonical: Optional[str]) -> str:
    lines = [JUDGE_PROMPT_HEADER]
    lines.append("Theorem Statement:\n" + theorem.strip())
    if canonical:
        lines.append("Canonical Proof (reference):\n" + canonical.strip())
    else:
        lines.append("Canonical Proof: <ABSENT>")
    lines.append("Candidate Proof Attempt:\n" + candidate_proof.strip())
    lines.append("Instructions:\n1. Compare candidate to canonical (if present).\n2. Score each rubric dimension (1-10).\n3. Provide holistic overall (not simple mean).\n4. Output ONLY JSON.")
    return "\n\n".join(lines)

def _parse_theorem_range(expr: str) -> List[int]:
    out: List[int] = []
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates-dir', default='data/rl/candidates')
    parser.add_argument('--out', default='data/rl/judge_scores.jsonl')
    parser.add_argument('--judge-model', default='frontier-judge-stub')
    parser.add_argument('--paper', required=True)
    parser.add_argument('--theorem', type=int, help='Single theorem index (use --theorems for multiple)')
    parser.add_argument('--theorems', help='Range/list e.g. 0-3 or 0,2,5 (overrides --theorem)')
    parser.add_argument('--real-judge', action='store_true', help='Use real judge model invocation (OpenAI backend currently).')
    parser.add_argument('--backend', default='openai', help='Judge backend when --real-judge set (openai).')
    parser.add_argument('--skip-existing', action='store_true', help='Skip candidates already present in scores file.')
    parser.add_argument('--canonical-max-chars', type=int, default=4000, help='Canonical proof char budget. 0 = exclude canonical entirely. -1 = include full canonical with no truncation. >0 = truncate and append [TRUNCATED] marker if exceeded.')
    parser.add_argument('--force-rescore', action='store_true', help='Ignore skip-existing and rescore all matching candidates.')
    parser.add_argument('--judge-temperature', type=float, default=None, help='Optional temperature override for judge model (omit for default).')
    parser.add_argument('--judge-max-output', type=int, default=None, help='Optional upper bound on judge response tokens (best-effort).')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    dm = DataManager()
    paper = dm.load_paper(args.paper)
    canonical_map = {}
    canonical = None
    if paper:
        parser_l = LatexParser()
        parsed = parser_l.parse(paper.latex_content or '')
        theorems = parsed.get('theorems', [])
        # canonical proof determined later per theorem index

    # Build list of theorem indices
    if args.theorems:
        theorem_indices = _parse_theorem_range(args.theorems)
    elif args.theorem is not None:
        theorem_indices = [args.theorem]
    else:
        print('Must specify --theorem or --theorems')
        return

    existing_ids: Set[str] = set()
    if args.skip_existing and not args.force_rescore and os.path.exists(args.out):
        try:
            with open(args.out,'r',encoding='utf-8') as ef:
                for line in ef:
                    line=line.strip()
                    if not line: continue
                    try:
                        js=json.loads(line)
                        existing_ids.add(js.get('candidate_id',''))
                    except Exception:
                        pass
        except Exception:
            pass

    pattern = os.path.join(args.candidates_dir, '*.json')
    cand_files = sorted(glob.glob(pattern))
    # Index candidates by theorem
    by_th: Dict[int, List[str]] = {}
    for cf in cand_files:
        try:
            with open(cf,'r',encoding='utf-8') as f:
                data = json.load(f)
            if data.get('meta',{}).get('paper_id') != args.paper:
                continue
            th_i = data.get('meta',{}).get('theorem_index')
            if th_i in theorem_indices:
                by_th.setdefault(th_i, []).append(cf)
        except Exception:
            continue

    total_candidates = sum(len(v) for v in by_th.values())
    if total_candidates == 0:
        print('No candidates found for requested theorems.')
        return

    written = 0
    with open(args.out, 'a', encoding='utf-8') as outf:
        for th_i in theorem_indices:
            files = by_th.get(th_i, [])
            if not files:
                print(f'[skip] No candidates for theorem {th_i}')
                continue
            # Determine canonical proof for this theorem if available
            canonical_local = None
            canonical_char_len = None
            canonical_included_len = 0
            canonical_truncated = False
            if paper and th_i < len(theorems):
                t = theorems[th_i]
                canon_full = t.proof if getattr(t,'proof',None) else None
                if canon_full:
                    canonical_char_len = len(canon_full)
                if args.canonical_max_chars == 0:
                    canonical_local = None
                elif canon_full:
                    if args.canonical_max_chars == -1:  # full inclusion
                        canonical_local = canon_full
                        canonical_included_len = len(canon_full)
                    else:
                        if len(canon_full) > args.canonical_max_chars:
                            canonical_local = canon_full[:args.canonical_max_chars] + '\n[TRUNCATED]'
                            canonical_included_len = args.canonical_max_chars
                            canonical_truncated = True
                        else:
                            canonical_local = canon_full
                            canonical_included_len = len(canon_full)
                else:
                    canonical_local = None
            for cf in files:
                with open(cf, 'r', encoding='utf-8') as f:
                    cand = json.load(f)
                if args.skip_existing and not args.force_rescore and cand['candidate_id'] in existing_ids:
                    continue
                theorem_stmt = cand.get('theorem_statement', '')
                proof = cand.get('generated_proof', '')
                prompt = build_prompt(theorem_stmt, proof, canonical_local)
                judge_raw, used_fallback = judge(prompt, args.judge_model, real=args.real_judge, backend=args.backend,
                                                 temperature=args.judge_temperature, max_output=args.judge_max_output)
                js = JudgeScore(
                    candidate_id=cand['candidate_id'],
                    paper_id=cand['meta']['paper_id'],
                    theorem_index=cand['meta']['theorem_index'],
                    judge_model=args.judge_model,
                    scores=judge_raw['scores'],
                    overall=judge_raw['overall'],
                    rationale=judge_raw.get('rationale'),
                    canonical_used=bool(canonical_local),
                    prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16]
                )
                row = json.loads(js.to_json())
                # Add canonical metadata for cost / truncation analysis
                if canonical_char_len is not None:
                    row['canonical_char_len'] = canonical_char_len
                row['canonical_included_len'] = canonical_included_len
                row['canonical_truncated'] = canonical_truncated
                row['canonical_max_chars'] = args.canonical_max_chars
                if used_fallback:
                    row['fallback'] = True
                outf.write(json.dumps(row, ensure_ascii=False) + '\n')
                written += 1
    skipped = 0
    if args.skip_existing and not args.force_rescore:
        skipped = sum(1 for th_i in theorem_indices for cf in by_th.get(th_i, []) if json.load(open(cf,'r',encoding='utf-8')).get('candidate_id') in existing_ids)
    print(f"Wrote {written} judge score rows to {args.out} (processed {total_candidates} candidates, skipped {skipped})")
    if written == 0 and skipped == total_candidates and args.skip_existing and not args.force_rescore:
        print("All matching candidates already scored. Use --force-rescore to overwrite or remove --skip-existing.")

if __name__ == '__main__':
    main()
