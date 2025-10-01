"""
Proof PDF Exporter
Generates human-readable PDF (via LaTeX) aggregating one or more generated proof attempts.

Usage examples:
  python proof_pdf_exporter.py --paper=2509.22618 --theorem=0
  python proof_pdf_exporter.py --paper=2509.22618 --batch --limit=5
  python proof_pdf_exporter.py --paper=2509.22618 --all
  python proof_pdf_exporter.py --paper=2509.22618 --theorem=0 --no-compile  # just produce .tex

If pdflatex is not installed or not found, use --no-compile to still generate the .tex file.
"""
import os
import json
import re
import argparse
import subprocess
import tempfile
from datetime import datetime
from typing import List, Dict, Optional

PDLATEX_CANDIDATES = [
    r"C:\\Program Files\\MiKTeX\\miktex\\bin\\x64\\pdflatex.exe",
    r"C:\\texlive\\2024\\bin\\windows\\pdflatex.exe",
    r"C:\\texlive\\2023\\bin\\win32\\pdflatex.exe",
    "pdflatex",
]

BASE_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage{amsmath, amssymb, amsthm, mathtools}
\usepackage{hyperref}
\usepackage{geometry}
\usepackage{enumitem}
\geometry{margin=1in}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\Q}{\mathbb{Q}}
\begin{document}
"""

FOOTER = "\\end{document}\n"


def find_pdflatex() -> Optional[str]:
    for cand in PDLATEX_CANDIDATES:
        if cand == "pdflatex":
            # optimistic path import; let subprocess raise if missing
            return cand
        if os.path.exists(cand):
            return cand
    return None


def sanitize_latex_basic(segment: str) -> str:
    """Escape LaTeX specials in a segment that is known to be outside math.

    We are conservative: keep existing double backslashes, but escape &, %, #, _, and unescaped $.
    We avoid escaping braces or caret/tilde to not over-noise plain text. If braces become an
    issue we can extend this later.
    """
    # First escape backslash-dollar patterns only for solitary dollars
    # Replace any $ not already escaped
    segment = re.sub(r'(?<!\\)\$', r'\\$', segment)
    replacements = {
        '&': r'\&',
        '#': r'\#',
        '%': r'\%',
        '_': r'\_',
    }
    out = []
    for ch in segment:
        out.append(replacements.get(ch, ch))
    return ''.join(out)


def sanitize_preserving_math(text: str) -> str:
    """Escape LaTeX specials outside math mode while preserving inline math $...$.

    Heuristic approach: identify simple inline math spans delimited by single $ ... $ (no newlines)
    and leave their interior untouched; everything else has specials escaped.
    If an odd number of $ occurs, the trailing unmatched $ is escaped to avoid entering math mode.
    """
    # Find non-overlapping inline math spans
    math_spans = []
    for m in re.finditer(r'(?<!\\)\$[^\n$]+?(?<!\\)\$', text):
        math_spans.append((m.start(), m.end()))
    if not math_spans:
        return sanitize_latex_basic(text)
    pieces = []
    last = 0
    for s, e in math_spans:
        if s > last:
            outside = text[last:s]
            pieces.append(sanitize_latex_basic(outside))
        pieces.append(text[s:e])  # keep math intact
        last = e
    if last < len(text):
        tail = text[last:]
        pieces.append(sanitize_latex_basic(tail))
    return ''.join(pieces)


def load_proof_jsons(paper_id: str, proof_results_dir: str, theorem_indices: List[int]) -> List[Dict]:
    results = []
    for idx in theorem_indices:
        path = os.path.join(proof_results_dir, f"{paper_id.replace('/', '_')}_theorem_{idx}.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    results.append(json.load(f))
                except json.JSONDecodeError:
                    print(f"Warning: could not parse {path}")
        else:
            print(f"Missing proof JSON: {path}")
    return results


def detect_available_indices(paper_id: str, proof_results_dir: str) -> List[int]:
    prefix = f"{paper_id.replace('/', '_')}_theorem_"
    indices = []
    for name in os.listdir(proof_results_dir):
        if name.startswith(prefix) and name.endswith('.json'):
            try:
                part = name[len(prefix):-5]
                indices.append(int(part))
            except ValueError:
                continue
    return sorted(indices)


def quality_block(q: Dict) -> str:
    if not q:
        return ""
    rows = []
    for k,v in q.items():
        rows.append(f"\\texttt{{{k}}}: {sanitize_latex_basic(str(v))}\\\\")
    return (
        "\\paragraph{Quality Heuristic Summary.}\\ "+
        "\\begin{small}" + "\\begin{flushleft}" + '\n'.join(rows) + "\\end{flushleft}\\end{small}\n"
    )


def theorem_block(entry: Dict) -> str:
    thm = (entry.get('theorem_statement') or '').strip()
    base_proof = (entry.get('generated_proof') or '').strip()
    quality = entry.get('quality') or {}
    model = entry.get('model_used', 'unknown')
    idx = entry.get('theorem_index', '?')
    tok = entry.get('token_count', '')
    proof_sections = entry.get('proof_sections') if isinstance(entry.get('proof_sections'), dict) else {}
    provenance = quality.get('provenance') if isinstance(quality, dict) else None

    # Clean residual artifacts
    base_proof = re.sub(r"Response\(id='[^']+'[^\n]*\n", '', base_proof)

    sketch = ''
    full = ''
    if isinstance(proof_sections, dict) and ('proof_sketch' in proof_sections or 'full_proof' in proof_sections):
        sketch = proof_sections.get('proof_sketch', '').strip()
        full = proof_sections.get('full_proof', '').strip()

    # Fallback: if split not available, treat whole thing as full proof
    if not full:
        full = base_proof

    # Protect GAP lines minimally (escape a few specials) without touching math heavily
    def _protect_gaps(txt: str) -> str:
        out_lines = []
        for line in txt.splitlines():
            if line.startswith('GAP:'):
                line = line.replace('&', '\\&').replace('%','\\%')
            out_lines.append(line)
        return '\n'.join(out_lines)
    full = _protect_gaps(full)
    # Sanitize outside math to avoid accidental italics from stray $ starting math mode
    full = sanitize_preserving_math(full)

    # Build provenance snippet (lightweight)
    prov_tex = ''
    if provenance and isinstance(provenance, dict):
        sec_meta = provenance.get('sections', [])
        lines = []
        for s in sec_meta[:8]:  # limit to first 8 sections to avoid bloat
            try:
                lines.append(f"{sanitize_latex_basic(s.get('title',''))}: {s.get('included_chars')} chars" + (' (trunc)' if s.get('truncated') else ''))
            except Exception:
                continue
        if lines:
            prov_tex = ("\\paragraph{Context Provenance.}\\\n" + "\\begin{small}" + ", ".join(lines) + "\\end{small}\n")

    sketch_block = ''
    if sketch:
        sketch_block = f"\\paragraph{{Proof Sketch.}}\\\\\n{sanitize_preserving_math(sketch)}\n"

    return rf"""\section*{{Theorem {idx}}}
\begin{{theorem}}[{sanitize_latex_basic(entry.get('paper_id',''))}]
{sanitize_preserving_math(thm)}
\end{{theorem}}

\paragraph{{Model Used}} {sanitize_latex_basic(model)}\\
\paragraph{{Token Count}} {tok}\\
{quality_block(quality)}
{prov_tex}
{sketch_block}
\paragraph{{Proof.}}\ 
{full}\ \qedhere
"""


def build_document(paper_id: str, entries: List[Dict], title: Optional[str]) -> str:
    title_line = f"\\title{{Proof Attempts for {sanitize_latex_basic(paper_id)} }}\n\\date{{Generated: {datetime.utcnow().isoformat()} UTC}}\n\\maketitle\n"
    body = ''.join(theorem_block(e) + '\n' for e in entries)
    return BASE_PREAMBLE + title_line + body + FOOTER


def write_tex(tex_content: str, out_dir: str, basename: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    tex_path = os.path.join(out_dir, f"{basename}.tex")
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_content)
    print(f"Written LaTeX file: {tex_path}")
    return tex_path


def compile_pdf(tex_path: str, pdflatex_path: Optional[str]) -> Optional[str]:
    if not pdflatex_path:
        print("pdflatex not found; skipping compilation (use --no-compile to silence).")
        return None
    work_dir = os.path.dirname(tex_path)
    name = os.path.basename(tex_path)
    log_tails = []
    for run in range(2):  # run twice for references
        try:
            r = subprocess.run([pdflatex_path, "-interaction=nonstopmode", name], cwd=work_dir,
                               capture_output=True, timeout=90)
            # Decode robustly (avoid codepage issues)
            try:
                stdout_txt = r.stdout.decode('utf-8', errors='replace') if isinstance(r.stdout, (bytes, bytearray)) else r.stdout
            except Exception:
                stdout_txt = str(r.stdout)
            if r.returncode != 0 and run == 0:
                print("First pass warnings/errors (continuing):")
                print(stdout_txt[-1200:])
            log_tails.append(stdout_txt[-400:])
        except subprocess.TimeoutExpired:
            print("pdflatex timed out.")
            return None
    pdf_path = tex_path[:-4] + '.pdf'
    if os.path.exists(pdf_path):
        print(f"PDF generated: {pdf_path}")
        return pdf_path
    print("PDF not produced; tail of log:")
    for tail in log_tails:
        print(tail)
    return None


def main():
    parser = argparse.ArgumentParser(description="Export generated proofs to PDF")
    parser.add_argument('--paper', required=True, help='arXiv paper id')
    parser.add_argument('--theorem', type=int, help='Single theorem index')
    parser.add_argument('--batch', action='store_true', help='Batch mode with --limit')
    parser.add_argument('--limit', type=int, default=5, help='Max theorems in batch export')
    parser.add_argument('--all', action='store_true', help='Export all available proofs for paper')
    parser.add_argument('--indices', type=str, help='Comma-separated theorem indices to export (overrides other selection flags except --theorem)')
    parser.add_argument('--proof-dir', default='proof_results', help='Directory containing proof JSON files')
    parser.add_argument('--out-dir', default='proof_pdfs', help='Output directory for TeX/PDF')
    parser.add_argument('--basename', default=None, help='Basename for output files')
    parser.add_argument('--no-compile', action='store_true', help='Only write .tex (skip PDF)')
    args = parser.parse_args()

    indices: List[int] = []
    if args.indices:
        try:
            indices = sorted({int(x.strip()) for x in args.indices.split(',') if x.strip()})
        except ValueError:
            parser.error('Invalid integer in --indices')
    elif args.all:
        indices = detect_available_indices(args.paper, args.proof_dir)
    elif args.batch:
        avail = detect_available_indices(args.paper, args.proof_dir)
        indices = avail[:args.limit]
    elif args.theorem is not None:
        indices = [args.theorem]
    else:
        parser.error('Specify one of --theorem, --batch, --all, or --indices')

    if not indices:
        print('No theorem indices resolved. Nothing to export.')
        return

    entries = load_proof_jsons(args.paper, args.proof_dir, indices)
    if not entries:
        print('No proof JSON files loaded. Aborting.')
        return

    title = f"arXiv {args.paper} Proof Attempts"
    tex_content = build_document(args.paper, entries, title)

    base = args.basename or (f"{args.paper.replace('/', '_')}_proofs")
    tex_path = write_tex(tex_content, args.out_dir, base)

    if args.no_compile:
        return

    pdflatex_path = find_pdflatex()
    compile_pdf(tex_path, pdflatex_path)

if __name__ == '__main__':
    main()
