import os, json, time, argparse, glob
from openai import OpenAI

SYSTEM_JUDGE = "You are an expert mathematical proof reviewer. You will rate a student proof against a canonical reference on multiple dimensions. Be concise and rigorous."

JUDGE_PROMPT_TEMPLATE = """You are given a theorem statement, a canonical reference proof, and a candidate proof produced by a model. Evaluate the candidate compared to the canonical reference.

Theorem Statement:
{statement}

Canonical Proof:
{canonical}

Candidate Proof:
{candidate}

Provide JSON with fields: correctness (0-5), rigor (0-5), clarity (0-5), completeness (0-5), overall (0-5) plus a short 'rationale' string.
JSON:
"""

def call_judge(client: OpenAI, model: str, statement: str, canonical: str, candidate: str):
    prompt = JUDGE_PROMPT_TEMPLATE.format(statement=statement, canonical=canonical, candidate=candidate)
    t0 = time.time()
    try:
        rsp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_JUDGE},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_completion_tokens=400,
        )
    except Exception as e:
        # Fallback without sampling params if necessary (shouldn't trigger with temperature=0)
        if 'temperature' in str(e):
            rsp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_JUDGE},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=400,
            )
        else:
            raise
    dt = time.time() - t0
    text = rsp.choices[0].message.content
    usage = rsp.usage
    return text, dt, usage

def safe_parse_json(block: str):
    import re, json
    m = re.search(r"\{.*\}", block, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found")
    raw = m.group(0)
    return json.loads(raw)

def load_canonical_from_parsed(paper_id: str, statement: str, parsed_dir: str) -> str:
    """Attempt to recover the canonical proof for a theorem statement by matching against parsed JSON.

    Strategy:
      - Open <paper_id>_parsed.json under parsed_dir
      - Iterate its 'theorems' list (if present)
      - Use simple fuzzy containment / equality heuristics on normalized whitespace to locate the best match
      - Return its 'proof' field if non-empty
    This is intentionally lightweight to avoid heavy dependencies.
    """
    parsed_path = os.path.join(parsed_dir, f"{paper_id}_parsed.json")
    if not os.path.exists(parsed_path):
        return ''
    try:
        with open(parsed_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
    except Exception:
        return ''
    theorems = doc.get('theorems') or []
    if not isinstance(theorems, list):
        return ''
    # Normalize target statement (remove spaces & newlines for fuzzy comparison)
    norm_target = ' '.join(statement.split())[:500]  # limit length for speed
    best = ''
    best_score = 0
    for th in theorems:
        stmt = th.get('statement') or ''
        proof = th.get('proof') or ''
        if not proof or not stmt:
            continue
        norm_stmt = ' '.join(stmt.split())[:500]
        # Score = length of overlap of shorter string if one contained; else 0
        score = 0
        if norm_stmt == norm_target:
            score = len(norm_stmt) + 1000  # strong bonus for exact match
        elif norm_stmt in norm_target or norm_target in norm_stmt:
            score = min(len(norm_stmt), len(norm_target))
        # Keep highest scoring
        if score > best_score:
            best_score = score
            best = proof.strip()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates-full', required=True)
    ap.add_argument('--candidates-minimal', required=True)
    ap.add_argument('--out-report', required=True)
    ap.add_argument('--out-scores-full', required=True)
    ap.add_argument('--out-scores-minimal', required=True)
    ap.add_argument('--model', default='gpt-5')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--rate-limit', type=int, default=60)
    ap.add_argument('--parsed-dir', default='data/parsed', help='Directory containing *_parsed.json with canonical proofs')
    ap.add_argument('--allow-missing-canonical', action='store_true', help='If set, judge even when canonical proof cannot be recovered (will skip otherwise).')
    ap.add_argument('--recover-canonical', action='store_true', help='Attempt to recover canonical proof from parsed JSON when not present in candidates records.')
    args = ap.parse_args()

    client = OpenAI()

    def load_existing(path):
        scores = {}
        if os.path.exists(path) and not args.force:
            with open(path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        scores[rec['id']] = rec
                    except Exception:
                        pass
        return scores

    existing_full = load_existing(args.out_scores_full)
    existing_min = load_existing(args.out_scores_minimal)

    def judge_file(candidates_path, existing_map, out_path):
        out_f = open(out_path, 'a')
        per_request_sleep = 60.0 / args.rate_limit if args.rate_limit > 0 else 0.0
        with open(candidates_path) as f:
            for line in f:
                cand = json.loads(line)
                tid = cand['id']
                if tid in existing_map and not args.force:
                    continue
                canonical = cand.get('canonical_proof') or cand.get('canonical') or ''
                if not canonical and args.recover_canonical:
                    canonical = load_canonical_from_parsed(cand.get('paper_id') or '', cand['statement'], args.parsed_dir)
                    if canonical:
                        print(f"[info] recovered canonical proof for {tid} from parsed JSON")
                if not canonical and not args.allow_missing_canonical:
                    print(f"[warn] no canonical proof for {tid}, skipping (use --recover-canonical or --allow-missing-canonical)")
                    continue
                candidate = cand['generation']
                try:
                    judge_text, latency, usage = call_judge(client, args.model, cand['statement'], canonical, candidate)
                except Exception as e:
                    print(f"[error] judge failed {tid}: {e}")
                    continue
                try:
                    parsed = safe_parse_json(judge_text)
                except Exception as e:
                    print(f"[warn] parse failed {tid}: {e}; raw captured")
                    parsed = {"raw": judge_text}
                rec = {
                    'id': tid,
                    'scores': parsed,
                    'latency_s': round(latency,3),
                    'prompt_tokens': usage.prompt_tokens if usage else None,
                    'completion_tokens': usage.completion_tokens if usage else None,
                    'total_tokens': usage.total_tokens if usage else None,
                }
                out_f.write(json.dumps(rec) + '\n')
                out_f.flush()
                if per_request_sleep:
                    time.sleep(per_request_sleep)
        out_f.close()

    judge_file(args.candidates_full, existing_full, args.out_scores_full)
    judge_file(args.candidates_minimal, existing_min, args.out_scores_minimal)

    full_scores = {}
    if os.path.exists(args.out_scores_full):
        with open(args.out_scores_full) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    full_scores[r['id']] = r['scores']
                except Exception:
                    pass
    min_scores = {}
    if os.path.exists(args.out_scores_minimal):
        with open(args.out_scores_minimal) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    min_scores[r['id']] = r['scores']
                except Exception:
                    pass

    dims = ['overall','correctness','rigor','clarity','completeness']
    deltas = {d: [] for d in dims}
    common = set(full_scores.keys()) & set(min_scores.keys())
    for tid in sorted(common):
        fs = full_scores[tid]
        ms = min_scores[tid]
        for d in dims:
            try:
                deltas[d].append((fs.get(d,0) - ms.get(d,0)))
            except Exception:
                pass

    def mean(xs):
        return sum(xs)/len(xs) if xs else 0.0

    with open(args.out_report, 'w') as rep:
        rep.write(f"Theorems compared: {len(common)}\n")
        for d in dims:
            dv = deltas[d]
            rep.write(f"Delta {d} mean={mean(dv):.3f} n={len(dv)}\n")

if __name__ == '__main__':
    main()
