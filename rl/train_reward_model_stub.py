"""Baseline reward model stub.

This stub trains a trivial scaler mapping rubric overall scores to a normalized reward and
writes a tiny 'model' artifact (JSON) recording mean/std for later standardization.

In a real setup, you'd replace with a transformer fine-tune or DPO style trainer.
"""
from __future__ import annotations
import argparse, json, os, math
from statistics import mean, pstdev


def load_reward_rows(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='data/rl/reward_dataset.jsonl')
    ap.add_argument('--out', default='data/rl/reward_model_stub.json')
    args = ap.parse_args()

    rows = load_reward_rows(args.dataset)
    if not rows:
        print('No reward rows found; aborting.')
        return

    rewards = [r['reward'] for r in rows if 'reward' in r]
    mu = mean(rewards)
    sigma = pstdev(rewards) if len(rewards) > 1 else 1.0

    # Simple calibration: z-score then squash to (0,1) via logistic
    def calibrate(x: float):
        z = (x - mu) / (sigma if sigma > 1e-6 else 1.0)
        return 1 / (1 + math.exp(-z))

    preview = [calibrate(r) for r in rewards[:5]]

    artifact = {
        'type': 'stub_reward_model',
        'mean': mu,
        'std': sigma,
        'count': len(rewards),
        'preview_calibrated_first5': preview
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(artifact, f, indent=2)
    print(f"Saved stub reward model to {args.out}")

if __name__ == '__main__':
    main()
