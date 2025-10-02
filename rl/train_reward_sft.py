"""Reward-Weighted Supervised Fine-Tuning (alternative to train_scalar_weighted).

This script trains a causal LM with LoRA adapters using a JSONL reward dataset where each line has:
  {"candidate_id": ..., "paper_id": ..., "theorem_index": ..., "reward": float, "model": base_model_name, "variant": str, "meta": {...}}

Differences vs train_scalar_weighted:
- Supports optional per-example temperature-style scaling of the loss: loss * (reward ** reward_power)
- Optionally normalizes rewards to [min_scale,1.0] using global or per-theorem min-max
- Can filter by variant(s) or model substrings

Usage:
  python -m rl.train_reward_sft \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --data data/rl/reward_dataset_correctness_norm.jsonl \
    --output-dir models/reward-sft-corrnorm-v1 \
    --lora-r 64 --lora-alpha 128 --batch-size 8 --epochs 1
"""
from __future__ import annotations
import argparse, json, os, math, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    raise SystemExit('peft package required. pip install peft')

@dataclass
class RewardExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    reward: float

class RewardDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]

SYSTEM_PREFIX = "System: You are an expert mathematician. Provide a rigorous proof or best-effort reasoning with GAP markers if steps are missing.\nUser:\n"
ASSISTANT_PREFIX = "\nAssistant:\n"

def build_prompt(theorem_statement: str) -> str:
    return SYSTEM_PREFIX + theorem_statement.strip() + ASSISTANT_PREFIX

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-model', required=True)
    ap.add_argument('--data', required=True, help='Reward dataset JSONL produced by build_reward_dataset.py')
    ap.add_argument('--candidate-dirs', nargs='+', default=['data/rl/candidates','data/rl/candidates_adapter','data/rl/candidates_baseline','data/rl/candidates_adapter_batch'],
                   help='Directories to search for candidate JSON files (searched in order).')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--max-length', type=int, default=2048)
    ap.add_argument('--lora-r', type=int, default=64)
    ap.add_argument('--lora-alpha', type=int, default=128)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--grad-accum', type=int, default=4)
    ap.add_argument('--warmup-ratio', type=float, default=0.05)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--reward-power', type=float, default=1.0, help='Scales weight = reward ** power (after normalization).')
    ap.add_argument('--min-scale', type=float, default=0.1, help='Lower bound after normalization.')
    ap.add_argument('--per-theorem-norm', action='store_true', help='Apply min-max normalization per theorem index rather than global.')
    ap.add_argument('--filter-variant', nargs='*', help='Only include examples whose variant is in this set.')
    ap.add_argument('--filter-model-substr', nargs='*', help='Only include examples where base model substring matches one of these.')
    return ap.parse_args()


def load_reward_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                js = json.loads(line)
                rows.append(js)
            except Exception:
                continue
    return rows


def normalize_rewards(rows: List[Dict[str, Any]], per_theorem: bool, min_scale: float) -> None:
    if per_theorem:
        by_th: Dict[int, List[float]] = {}
        for r in rows:
            by_th.setdefault(r['theorem_index'], []).append(r['reward'])
        stats = {k: (min(v), max(v)) for k,v in by_th.items()}
        for r in rows:
            mn, mx = stats[r['theorem_index']]
            if mx > mn:
                scaled = (r['reward'] - mn) / (mx - mn)
            else:
                scaled = 0.5
            r['reward_norm'] = min_scale + (1-min_scale)*scaled
    else:
        rewards = [r['reward'] for r in rows]
        mn, mx = min(rewards), max(rewards)
        for r in rows:
            if mx > mn:
                scaled = (r['reward'] - mn) / (mx - mn)
            else:
                scaled = 0.5
            r['reward_norm'] = min_scale + (1-min_scale)*scaled


def collate(batch: List[Dict[str, Any]]):
    input_ids = torch.nn.utils.rnn.pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention = torch.nn.utils.rnn.pad_sequence([b['attention_mask'] for b in batch], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([b['labels'] for b in batch], batch_first=True, padding_value=-100)
    weights = torch.tensor([b['weight'] for b in batch], dtype=torch.float32)
    return {'input_ids': input_ids, 'attention_mask': attention, 'labels': labels, 'weights': weights}


def seed_all(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def prepare_rows(rows: List[Dict[str, Any]], tokenizer, max_length: int, reward_power: float, filter_variant, filter_model_substr,
                 cand_index: Dict[str, Dict[str, Any]]):
    out = []
    missing = 0
    for r in rows:
        if filter_variant and r.get('variant') not in filter_variant:
            continue
        if filter_model_substr and not any(s in r.get('model', '') for s in filter_model_substr):
            continue
        meta = r.get('meta', {})
        theorem_stmt = meta.get('theorem_statement') or meta.get('theorem') or ''
        generated = meta.get('generated_proof') or meta.get('proof') or ''
        if (not theorem_stmt or not generated) and r.get('candidate_id') in cand_index:
            c = cand_index[r['candidate_id']]
            theorem_stmt = c.get('theorem_statement', theorem_stmt)
            generated = c.get('generated_proof', generated)
        if not theorem_stmt or not generated:
            missing += 1
            continue
        prompt = build_prompt(theorem_stmt)
        full = prompt + generated
        toks = tokenizer(full, truncation=True, max_length=max_length, return_tensors='pt')
        input_ids = toks.input_ids[0]
        attn = toks.attention_mask[0]
        labels = input_ids.clone()
        prompt_len = len(tokenizer(prompt).input_ids)
        labels[:prompt_len] = -100
        weight = float(r.get('reward_norm', r['reward'])) ** reward_power
        out.append({'input_ids': input_ids, 'attention_mask': attn, 'labels': labels, 'weight': weight})
    if missing:
        print(f"[info] Skipped {missing} rows lacking candidate reconstruction data.")
    return out

def index_candidates(candidate_dirs: List[str]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for d in candidate_dirs:
        if not os.path.isdir(d):
            continue
        for fp in os.scandir(d):
            if not fp.name.endswith('.json'):
                continue
            try:
                with open(fp.path,'r',encoding='utf-8') as f:
                    js = json.load(f)
                cid = js.get('candidate_id')
                if cid and cid not in idx:
                    idx[cid] = js
            except Exception:
                continue
    print(f"[index] Loaded {len(idx)} candidates from {len(candidate_dirs)} dirs.")
    return idx

if __name__ == '__main__':
    args = parse_args()
    seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    model = get_peft_model(model, lora_cfg)
    model.to(device)
    rows = load_reward_rows(args.data)
    normalize_rewards(rows, per_theorem=args.per_theorem_norm, min_scale=args.min_scale)
    cand_index = index_candidates(args.candidate_dirs)
    prepped = prepare_rows(rows, tokenizer, args.max_length, args.reward_power, args.filter_variant, args.filter_model_substr, cand_index)
    if not prepped:
        raise SystemExit('No training rows after filtering.')
    ds = RewardDataset(prepped)
    steps_per_epoch = math.ceil(len(ds)/args.batch_size/args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, int(total_steps*args.warmup_ratio), total_steps)
    model.train()
    os.makedirs(args.output_dir, exist_ok=True)
    step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            batch = {k: v.to(device) for k,v in batch.items()}
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            base_loss = out.loss
            w = batch['weights']
            scaled_loss = base_loss * w.mean()
            scaled_loss.backward()
            if (step+1) % args.grad_accum == 0:
                opt.step(); sched.step(); opt.zero_grad(); model.zero_grad(set_to_none=True)
            if (step+1) % 50 == 0:
                print(f"epoch {epoch} step {step+1} loss {float(base_loss):.4f} mean_w {float(w.mean()):.3f}")
            step += 1
        ckpt_dir = os.path.join(args.output_dir, f'epoch_{epoch}')
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"Saved epoch checkpoint -> {ckpt_dir}")
    print('Training complete.')
