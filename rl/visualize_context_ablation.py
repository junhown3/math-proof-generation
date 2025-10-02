"""Visualization utility for context ablation (full vs minimal context generation).

Reads two judge score JSONL files (one for full-context, one for minimal-context
mode) and produces a small set of static plots to `--out-dir`:

 1. boxplot_overall.png       – Distribution of overall scores per mode
 2. scatter_min_vs_full.png   – Scatter plot of per-theorem best (overall) minimal vs full
 3. delta_hist.png            – Histogram of (full - minimal) overall best scores
 4. per_dim_delta_bar.png     – Bar plot of mean per-dimension deltas (full - minimal)

Assumptions:
 - Each JSONL row has: paper_id, theorem_index, overall, scores{dimension:val}
 - Multiple candidates per (paper,theorem) may exist; we reduce to the BEST (max overall)
   within each mode before computing scatter / delta metrics.

Usage:
  python -m rl.visualize_context_ablation \
      --full data/rl/context_ablation/judge_scores_full_real.jsonl \
      --minimal data/rl/context_ablation/judge_scores_min_real.jsonl \
      --out-dir data/rl/context_ablation/figures

The script intentionally has NO heavy dependencies beyond matplotlib & stdlib.
"""
from __future__ import annotations
import argparse, os, json, math
from typing import Dict, List, Tuple, Any
import statistics
import matplotlib
matplotlib.use('Agg')  # headless safe
import matplotlib.pyplot as plt

RUBRIC_DIMS = ["correctness","completeness","cohesion","retrieval_grounding","conciseness"]


def load_scores(path: str) -> List[Dict[str, Any]]:
    rows = []
    if not os.path.exists(path):
        raise SystemExit(f"Score file not found: {path}")
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                js=json.loads(line)
            except Exception:
                continue
            if 'overall' in js and 'theorem_index' in js:
                rows.append(js)
    if not rows:
        raise SystemExit(f"No valid rows parsed from {path}")
    return rows


def best_per_theorem(rows: List[Dict[str,Any]]) -> Dict[Tuple[str,int], Dict[str,Any]]:
    best: Dict[Tuple[str,int], Dict[str,Any]] = {}
    for r in rows:
        key = (r.get('paper_id'), r.get('theorem_index'))
        cur = best.get(key)
        if cur is None or r['overall'] > cur['overall']:
            best[key] = r
    return best


def boxplot(scores_full: List[float], scores_min: List[float], out_path: str):
    plt.figure(figsize=(5,4))
    plt.boxplot([scores_min, scores_full], labels=['minimal','full'], showfliers=False)
    plt.ylabel('Overall Score')
    plt.title('Overall Score Distribution')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def scatter(best_min: Dict[Tuple[str,int], Dict[str,Any]], best_full: Dict[Tuple[str,int], Dict[str,Any]], out_path: str):
    xs=[]; ys=[]
    for key in sorted(set(best_min.keys()) & set(best_full.keys())):
        xs.append(best_min[key]['overall'])
        ys.append(best_full[key]['overall'])
    plt.figure(figsize=(5,5))
    plt.scatter(xs, ys, alpha=0.7, edgecolors='k', linewidths=0.5)
    max_axis = max(xs+ys) if xs and ys else 10
    plt.plot([0,max_axis],[0,max_axis], 'r--', linewidth=1)
    plt.xlabel('Minimal (best per theorem)')
    plt.ylabel('Full (best per theorem)')
    plt.title('Per-Theorem Best Overall: Minimal vs Full')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def delta_hist(best_min: Dict[Tuple[str,int], Dict[str,Any]], best_full: Dict[Tuple[str,int], Dict[str,Any]], out_path: str):
    deltas=[]
    for key in sorted(set(best_min.keys()) & set(best_full.keys())):
        deltas.append(best_full[key]['overall'] - best_min[key]['overall'])
    plt.figure(figsize=(5,4))
    plt.hist(deltas, bins=20, color='#4C72B0', edgecolor='k')
    plt.axvline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Δ overall (full - minimal)')
    plt.ylabel('Count')
    plt.title('Distribution of Overall Score Delta')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def per_dim_bar(best_min: Dict[Tuple[str,int], Dict[str,Any]], best_full: Dict[Tuple[str,int], Dict[str,Any]], out_path: str):
    dim_means = []
    labels = []
    for dim in RUBRIC_DIMS:
        diffs=[]
        for key in sorted(set(best_min.keys()) & set(best_full.keys())):
            mv = best_min[key].get('scores',{}).get(dim)
            fv = best_full[key].get('scores',{}).get(dim)
            if isinstance(mv,(int,float)) and isinstance(fv,(int,float)):
                diffs.append(fv-mv)
        if diffs:
            labels.append(dim)
            dim_means.append(statistics.mean(diffs))
    plt.figure(figsize=(6,4))
    plt.bar(labels, dim_means, color='#55A868', edgecolor='k')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel('Mean Δ (full - minimal)')
    plt.title('Per-Dimension Mean Improvement')
    for i,v in enumerate(dim_means):
        plt.text(i, v + (0.02 if v>=0 else -0.02), f"{v:+.2f}", ha='center', va='bottom' if v>=0 else 'top', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', required=True, help='Judge scores JSONL for full-context mode')
    ap.add_argument('--minimal', required=True, help='Judge scores JSONL for minimal-context mode')
    ap.add_argument('--out-dir', required=True, help='Directory to write figures')
    ap.add_argument('--prefix', default='')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    full_rows = load_scores(args.full)
    min_rows = load_scores(args.minimal)

    best_full = best_per_theorem(full_rows)
    best_min = best_per_theorem(min_rows)

    # Boxplot uses ALL rows (not just best)
    boxplot([r['overall'] for r in full_rows], [r['overall'] for r in min_rows], os.path.join(args.out_dir, f"{args.prefix}boxplot_overall.png"))
    scatter(best_min, best_full, os.path.join(args.out_dir, f"{args.prefix}scatter_min_vs_full.png"))
    delta_hist(best_min, best_full, os.path.join(args.out_dir, f"{args.prefix}delta_hist.png"))
    per_dim_bar(best_min, best_full, os.path.join(args.out_dir, f"{args.prefix}per_dim_delta_bar.png"))

    # Small textual summary for quick glance
    common = sorted(set(best_min.keys()) & set(best_full.keys()))
    deltas = [best_full[k]['overall'] - best_min[k]['overall'] for k in common]
    summary_path = os.path.join(args.out_dir, f"{args.prefix}summary.txt")
    with open(summary_path,'w',encoding='utf-8') as f:
        f.write(f"Pairs compared: {len(common)}\n")
        if deltas:
            f.write(f"Mean Δ overall (full - minimal): {statistics.mean(deltas):+.3f}\n")
            f.write(f"Median Δ overall: {statistics.median(deltas):+.3f}\n")
            pos = sum(1 for d in deltas if d>1e-6); neg=sum(1 for d in deltas if d<-1e-6); tie=len(deltas)-pos-neg
            f.write(f"Wins/Losses/Ties (full vs minimal): {pos}/{neg}/{tie}\n")
    print(f"Wrote figures + summary to {args.out_dir}")

if __name__ == '__main__':
    main()
