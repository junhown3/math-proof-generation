import os, json, argparse, glob

# Build a JSONL of theorem records from parsed paper JSON files.
# Each parsed file is assumed to have structure with optional 'theorems' key OR we heuristically
# extract theorem-like blocks from sections whose content contains 'Theorem[' or '\\begin{theorem}'.
# We do NOT embed canonical proofs here unless clearly delimited; instead we store the raw theorem statement
# and if the parsed JSON already segments proofs we include them.

def build_context(doc_json: dict) -> str:
    abstract = doc_json.get('abstract') or ''
    intro = ''
    for sec in doc_json.get('sections', []):
        title = (sec.get('title') or '').lower()
        if 'introduction' in title:
            intro = sec.get('content', '') or ''
            break
    # Combine and truncate for safety (character-based heuristic)
    combined = (abstract + '\n\n' + intro).strip()
    if len(combined) > 6000:
        combined = combined[:6000] + '...'
    return combined


def extract_from_parsed(doc_json: dict, paper_id: str):
    out = []
    # If upstream pipeline already produced a theorems list (schema unknown here), attempt to use it.
    if 'theorems' in doc_json and isinstance(doc_json['theorems'], list):
        for idx, th in enumerate(doc_json['theorems']):
            stmt = th.get('statement') or th.get('title') or ''
            proof = th.get('proof') or th.get('canonical_proof') or ''
            if not stmt:
                continue
            rec = {
                'id': f'{paper_id}_th{idx+1}',
                'paper_id': paper_id,
                'statement': stmt.strip(),
            }
            if proof:
                rec['canonical_proof'] = proof.strip()
            ctx = build_context(doc_json)
            if ctx:
                rec['context'] = ctx
            out.append(rec)
        if out:
            return out
    # Fallback heuristic: scan sections for theorem environments.
    sections = doc_json.get('sections', [])
    for sec in sections:
        txt = sec.get('content', '') or ''
        # Simple split: look for '\n\\begin{theorem}' markers.
        parts = txt.split('\\begin{theorem}')
        if len(parts) == 1:
            continue
        # Skip the first preamble part.
        accum_index = 0
        for chunk in parts[1:]:
            # Terminate at \end{theorem}
            if '\\end{theorem}' in chunk:
                body, _rest = chunk.split('\\end{theorem}', 1)
            else:
                body = chunk
            body = body.strip()
            if not body:
                continue
            accum_index += 1
            rec = {
                'id': f'{paper_id}_heur{accum_index}',
                'paper_id': paper_id,
                'statement': body,
            }
            ctx = build_context(doc_json)
            if ctx:
                rec['context'] = ctx
            out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parsed-dir', default='data/parsed', help='Directory containing *_parsed.json files')
    ap.add_argument('--limit-papers', type=int, default=5)
    ap.add_argument('--max-theorems-per-paper', type=int, default=5)
    ap.add_argument('--out-jsonl', required=True)
    ap.add_argument('--require-canonical', action='store_true', help='Keep only records with canonical_proof')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.parsed_dir, '*_parsed.json')))
    if args.limit_papers:
        files = files[:args.limit_papers]

    total = 0
    kept = 0
    with open(args.out_jsonl, 'w') as fout:
        for fpath in files:
            paper_id = os.path.basename(fpath).split('_parsed.json')[0]
            try:
                with open(fpath) as f:
                    doc = json.load(f)
            except Exception as e:
                print(f'[warn] failed to load {fpath}: {e}')
                continue
            recs = extract_from_parsed(doc, paper_id)
            if not recs:
                continue
            # Limit per paper
            recs = recs[:args.max_theorems_per_paper]
            for r in recs:
                total += 1
                if args.require_canonical and 'canonical_proof' not in r:
                    continue
                fout.write(json.dumps(r) + '\n')
                kept += 1
    print(f'[summary] scanned={len(files)} total_candidates={total} written={kept} -> {args.out_jsonl}')

if __name__ == '__main__':
    main()
