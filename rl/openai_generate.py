import os, time, json, argparse, random
from typing import List, Dict
from openai import OpenAI

SYSTEM_GEN = "You are an expert mathematical proof assistant. Produce a clear, rigorous proof."

GEN_PROMPT_TEMPLATE = """Problem:
{statement}

{context_section}
Provide a rigorous proof. Begin directly with the proof.
"""

CANONICAL_KEY = "canonical_proof"

def build_prompt(statement: str, theorem_context: str | None, mode: str) -> str:
    if mode == 'full' and theorem_context:
        ctx = f"Additional Context:\n{theorem_context}\n"
    else:
        ctx = ""
    return GEN_PROMPT_TEMPLATE.format(statement=statement, context_section=ctx)

def generate_openai(client: OpenAI, model: str, prompt: str, temperature: float, top_p: float, max_new_tokens: int):
    t0 = time.time()
    kwargs = {
        'model': model,
        'messages': [
            {"role": "system", "content": SYSTEM_GEN},
            {"role": "user", "content": prompt}
        ],
        'max_completion_tokens': max_new_tokens,
    }
    try:
        rsp = client.chat.completions.create(temperature=temperature, top_p=top_p, **kwargs)
    except Exception as e:
        if 'temperature' in str(e) or 'top_p' in str(e):
            rsp = client.chat.completions.create(**kwargs)
        else:
            raise
    dt = time.time() - t0
    choice = rsp.choices[0]
    text = choice.message.content
    usage = rsp.usage
    return {
        "text": text,
        "latency_s": round(dt, 3),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-json', required=True, help='JSONL with theorem records (id, statement, optional context, canonical_proof)')
    ap.add_argument('--mode', choices=['full','minimal'], required=True)
    ap.add_argument('--model', default='gpt-5')
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top-p', type=float, default=0.95)
    ap.add_argument('--max-new-tokens', type=int, default=512)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--rate-limit', type=int, default=30, help='Max requests per minute (simple sleep throttle)')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'candidates_{args.mode}.jsonl')

    existing = set()
    if os.path.exists(out_path) and not args.force:
        with open(out_path) as f:
            for line in f:
                try:
                    existing.add(json.loads(line)["id"])
                except Exception:
                    pass

    client = OpenAI()

    per_request_sleep = 60.0 / args.rate_limit if args.rate_limit > 0 else 0.0

    with open(args.input_json) as fin, open(out_path, 'a') as fout:
        for line in fin:
            rec = json.loads(line)
            tid = rec['id']
            if tid in existing:
                continue
            statement = rec['statement']
            theorem_context = rec.get('context')
            prompt = build_prompt(statement, theorem_context, args.mode)
            try:
                gen = generate_openai(client, args.model, prompt, args.temperature, args.top_p, args.max_new_tokens)
            except Exception as e:
                print(f"[error] generation failed for {tid}: {e}")
                continue
            canonical = rec.get('canonical_proof') or rec.get('canonical')
            out_rec = {
                'id': tid,
                'mode': args.mode,
                'statement': statement,
                'prompt': prompt,
                'generation': gen['text'],
                'latency_s': gen['latency_s'],
                'prompt_tokens': gen['prompt_tokens'],
                'completion_tokens': gen['completion_tokens'],
                'total_tokens': gen['total_tokens'],
                'canonical_proof': canonical,
            }
            fout.write(json.dumps(out_rec) + '\n')
            fout.flush()
            print(f"[gen] {tid} mode={args.mode} latency={gen['latency_s']}s tokens={gen['total_tokens']}")
            if per_request_sleep:
                time.sleep(per_request_sleep)

if __name__ == '__main__':
    main()
