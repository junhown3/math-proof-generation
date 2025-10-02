"""Run context ablation pipeline:
1. Select theorem pairs (paper_id,theorem_index) either from --pairs file (JSONL with paper_id,theorem_index) or first N from manifest JSONL.
2. Generate minimal + full-context candidates via rl/generate_context_ablation.py
3. Judge both with stub or real judge (two passes) using rl/judge_scores.py
4. Compare with rl/compare_context_modes.py producing report(s).

Isolation: writes to data/rl/context_ablation/*
"""
from __future__ import annotations
import argparse, os, json, subprocess, tempfile, sys, pathlib
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from typing import List, Tuple

THIS_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'rl', 'context_ablation')
DEFAULT_MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'


def read_pairs_from_manifest(manifest_path: str, limit: int) -> List[Tuple[str,int]]:
    pairs = []
    with open(manifest_path,'r',encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            js = json.loads(line)
            pid = js.get('paper_id') or js.get('paper')
            th = js.get('theorem_index') or js.get('idx')
            if pid is None or th is None: continue
            pairs.append((pid, int(th)))
            if len(pairs) >= limit: break
    return pairs


def write_pairs_file(pairs: List[Tuple[str,int]], path: str):
    with open(path,'w',encoding='utf-8') as f:
        for pid, th in pairs:
            f.write(json.dumps({'paper_id': pid, 'theorem_index': th})+'\n')


def run(cmd: List[str]):
    print('\n>> RUN:', ' '.join(cmd), flush=True)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"Command failed with code {res.returncode}", file=sys.stderr)
        sys.exit(res.returncode)


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='data/rl/paper_theorem_manifest.jsonl', help='Manifest JSONL listing theorem entries')
    ap.add_argument('--pairs', help='Optional explicit pairs JSONL (paper_id,theorem_index)')
    ap.add_argument('--num-pairs', type=int, default=10)
    ap.add_argument('--base-model', default=DEFAULT_MODEL)
    ap.add_argument('--temperature', type=float, default=0.3)
    ap.add_argument('--max-new-tokens', type=int, default=1024)
    ap.add_argument('--double-canonical', action='store_true')
    ap.add_argument('--stub-first', action='store_true', help='Run stub judge then real judge second if --real-judge provided')
    ap.add_argument('--real-judge', action='store_true', help='After stub (or only) run, call real judge model')
    ap.add_argument('--judge-model', default='gpt-5')
    ap.add_argument('--canonical-max-chars', type=int, default=-1)
    ap.add_argument('--force-generate', action='store_true')
    ap.add_argument('--skip-generate', action='store_true')
    ap.add_argument('--report-prefix', default='ablation')
    ap.add_argument('--dummy', action='store_true', help='Use dummy generation (no model load) for fast stub pipeline.')
    args = ap.parse_args()

    ensure_dirs()

    # Step 1: Determine pairs
    pairs_path = args.pairs
    if not pairs_path:
        pairs_list = read_pairs_from_manifest(args.manifest, args.num_pairs)
        if not pairs_list:
            print('No pairs found from manifest. Aborting.')
            return
        pairs_path = os.path.join(DATA_DIR, 'pairs_selected.jsonl')
        write_pairs_file(pairs_list, pairs_path)
        print(f'Wrote selected pairs -> {pairs_path}')
    else:
        print(f'Using provided pairs file {pairs_path}')

    # Step 2: Generation (minimal & full)
    minimal_candidates = os.path.join(DATA_DIR, 'candidates_minimal')
    full_candidates = os.path.join(DATA_DIR, 'candidates_full')
    os.makedirs(minimal_candidates, exist_ok=True)
    os.makedirs(full_candidates, exist_ok=True)

    # Generation: our current generate_context_ablation.py expects --paper, --theorems, --context-mode, --out-dir
    gen_script = os.path.join(ROOT, 'rl', 'generate_context_ablation.py')
    if not args.skip_generate:
        # Load pairs (paper_id,theorem_index)
        pair_rows = []
        with open(pairs_path,'r',encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try:
                    js=json.loads(line)
                except Exception:
                    continue
                pid = js.get('paper_id'); th = js.get('theorem_index')
                if pid is None or th is None: continue
                pair_rows.append((pid,int(th)))
        # Group by paper for fewer model loads (the generator loads model per invocation).
        from collections import defaultdict
        by_paper = defaultdict(list)
        for pid, th in pair_rows:
            by_paper[pid].append(th)
        for pid, th_list in by_paper.items():
            th_str = ','.join(str(t) for t in sorted(set(th_list)))
            for mode,outdir in [('minimal', minimal_candidates), ('full', full_candidates)]:
                # Skip if all targets already exist (best-effort):
                existing = 0; needed = len(th_list)
                for t in th_list:
                    # Candidate files hashed: paper::th::variant::seed -> prefix; we cannot precompute without seed.
                    # Instead, scan existing dir for metadata match (lightweight for small counts).
                    for fp in os.listdir(outdir):
                        if not fp.endswith('.json'): continue
                        try:
                            js = json.loads(open(os.path.join(outdir, fp),'r',encoding='utf-8').read())
                        except Exception:
                            continue
                        meta = js.get('meta',{})
                        if meta.get('paper_id')==pid and meta.get('theorem_index')==t and meta.get('provenance',{}).get('context_mode')==mode:
                            existing += 1
                            break
                if existing == needed and not args.force_generate:
                    print(f"[gen-skip] {mode} all {needed} candidates for paper {pid} already exist")
                    continue
                cmd = [sys.executable, gen_script,
                       '--paper', pid,
                       '--theorems', th_str,
                       '--base-model', args.base_model,
                       '--out-dir', outdir,
                       '--context-mode', 'full' if mode=='full' else 'minimal',
                       '--temperature', str(args.temperature),
                       '--max-new-tokens', str(args.max_new_tokens)]
                if mode=='full':
                    cmd.append('--rag')  # enable RAG in full mode by default
                if args.double_canonical and mode=='full':
                    cmd.append('--double-canonical')
                if args.force_generate:
                    cmd.append('--force')
                if args.dummy:
                    cmd.append('--dummy')
                run(cmd)
    else:
        print('Skipping generation as requested.')

    # Step 3: Judging stub first (or only)
    judge_script = os.path.join(ROOT, 'rl', 'judge_scores.py')
    full_stub_scores = os.path.join(DATA_DIR, 'judge_scores_full_stub.jsonl')
    min_stub_scores = os.path.join(DATA_DIR, 'judge_scores_min_stub.jsonl')

    def judge(candidates_dir: str, out_path: str, real: bool):
        # Derive involved papers + theorem range for concise scoring (we reuse manifest pairs).
        # For simplicity, we score all candidates in the directory (paper filter not yet applied here) by iterating per paper.
        # We re-open pairs file to know which (paper,theorem) combos to score.
        pair_list = []
        with open(pairs_path,'r',encoding='utf-8') as f:
            for line in f:
                line=line.strip();
                if not line: continue
                try:
                    js=json.loads(line)
                except Exception:
                    continue
                pid=js.get('paper_id'); th=js.get('theorem_index')
                if pid is None or th is None: continue
                pair_list.append((pid,int(th)))
        from collections import defaultdict
        by_paper = defaultdict(list)
        for pid, th in pair_list:
            by_paper[pid].append(th)
        for pid, ths in by_paper.items():
            th_range = ','.join(str(t) for t in sorted(set(ths)))
            cmd = [sys.executable, judge_script,
                   '--candidates-dir', candidates_dir,
                   '--paper', pid,
                   '--theorems', th_range,
                   '--out', out_path,
                   '--canonical-max-chars', str(args.canonical_max_chars),
                   '--skip-existing']
            if real:
                cmd.append('--real-judge')
                cmd += ['--judge-model', args.judge_model]
            run(cmd)

    # Stub run
    judge(full_candidates, full_stub_scores, real=False)
    judge(minimal_candidates, min_stub_scores, real=False)

    compare_script = os.path.join(ROOT, 'rl', 'compare_context_modes.py')
    report_stub = os.path.join(DATA_DIR, f'{args.report_prefix}_stub.txt')
    run([sys.executable, compare_script, '--full', full_stub_scores, '--minimal', min_stub_scores, '--out', report_stub, '--best-of'])

    if args.real_judge:
        # Real judge pass (reuse same candidates)
        full_real_scores = os.path.join(DATA_DIR, 'judge_scores_full_real.jsonl')
        min_real_scores = os.path.join(DATA_DIR, 'judge_scores_min_real.jsonl')
        if args.stub_first:
            print('Stub-first completed; starting real judge scoring...')
        judge(full_candidates, full_real_scores, real=True)
        judge(minimal_candidates, min_real_scores, real=True)
        report_real = os.path.join(DATA_DIR, f'{args.report_prefix}_real.txt')
        run([sys.executable, compare_script, '--full', full_real_scores, '--minimal', min_real_scores, '--out', report_real, '--best-of'])

    print('Context ablation pipeline complete.')

if __name__ == '__main__':
    main()
