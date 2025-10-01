"""Weighted scalar fine-tuning.

This script performs a simple supervised fine-tune where each candidate proof
is treated as a target completion conditioned on the theorem statement.
Each example's loss is multiplied by its (normalized) reward score.

Intended to reflect your specified RL approach: generate -> judge -> optimize
using scalar overall scores directly (no pairwise preference construction).

Usage example:
  python -m rl.train_scalar_weighted \
    --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --reward-file data/rl/reward_dataset.jsonl \
    --output-dir models/scalar-ft-v1 \
    --epochs 1 --lr 2e-5 --batch-size 2 --gradient-accumulation 8

Supports optional LoRA and 4-bit quantization similar to DPO script.
"""
from __future__ import annotations
import argparse, json, os, math
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                              get_linear_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    raise SystemExit("Missing packages: transformers, peft, (optional bitsandbytes). Install requirements first.")

SYSTEM_PREFIX = "System: You are an expert mathematician. Provide a rigorous proof.\nUser:\n"
ASSISTANT_PREFIX = "\nAssistant:\n"

def load_reward_rows(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise SystemExit(f"Reward file not found: {path}")
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        raise SystemExit("No reward rows parsed")
    return rows

class ScalarRewardDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], candidates_dir: str, tokenizer, max_length: int):
        self.examples: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Build idx by candidate id to open candidate json for prompt+completion
        by_id = {}
        for fname in os.listdir(candidates_dir):
            if not fname.endswith('.json'):
                continue
            path = os.path.join(candidates_dir, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                by_id[data['candidate_id']] = data
            except Exception:
                continue
        # Collect examples
        for r in rows:
            cid = r['candidate_id']
            reward = float(r['reward'])
            cand = by_id.get(cid)
            if not cand:
                continue
            theorem = cand.get('theorem_statement', '').strip()
            proof = cand.get('generated_proof', '').strip()
            if len(proof) < 30:
                continue
            self.examples.append({
                'theorem': theorem,
                'proof': proof,
                'reward': reward
            })
        if not self.examples:
            raise SystemExit("No usable examples after filtering")
        # Normalize rewards to [0.1, 1.0] (avoid zero weight) via min-max
        vals = [ex['reward'] for ex in self.examples]
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx > mn else 1.0
        for ex in self.examples:
            norm = (ex['reward'] - mn) / rng
            ex['weight'] = 0.1 + 0.9 * norm

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = SYSTEM_PREFIX + ex['theorem'] + ASSISTANT_PREFIX + ex['proof']
        tokens = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = tokens['input_ids'][0]
        attn = tokens['attention_mask'][0]
        return {
            'input_ids': input_ids,
            'attention_mask': attn,
            'weight': torch.tensor(ex['weight'], dtype=torch.float32)
        }

def collate(batch, pad_token_id: int):
    max_len = max(item['input_ids'].size(0) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    weights = torch.stack([b['weight'] for b in batch])
    for i, item in enumerate(batch):
        l = item['input_ids'].size(0)
        input_ids[i, :l] = item['input_ids']
        attn[i, :l] = item['attention_mask']
    # Labels = input_ids shifted; typical LM training uses same tokens with ignore for pads
    labels = input_ids.clone()
    labels[input_ids == pad_token_id] = -100
    return {
        'input_ids': input_ids,
        'attention_mask': attn,
        'labels': labels,
        'weights': weights
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-name', required=True)
    ap.add_argument('--reward-file', default='data/rl/reward_dataset.jsonl')
    ap.add_argument('--candidates-dir', default='data/rl/candidates')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--gradient-accumulation', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--warmup-ratio', type=float, default=0.03)
    ap.add_argument('--max-length', type=int, default=2048)
    ap.add_argument('--four-bit', action='store_true')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--lora-r', type=int, default=16)
    ap.add_argument('--lora-alpha', type=int, default=32)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument('--no-lora', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading rewards...', flush=True)
    reward_rows = load_reward_rows(args.reward_file)

    print('Loading tokenizer/model...', flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.four_bit:
        quant_config = BitsAndBytesConfig(load_in_4bit=True,
                                          bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
                                          bnb_4bit_quant_type='nf4',
                                          bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map='auto',
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        quantization_config=quant_config
    )

    if not args.no_lora:
        if args.four_bit:
            model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                              bias='none', task_type='CAUSAL_LM')
        model = get_peft_model(model, lora_cfg)

    dataset = ScalarRewardDataset(reward_rows, args.candidates_dir, tokenizer, args.max_length)
    print(f"Dataset examples: {len(dataset)}", flush=True)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, tokenizer.pad_token_id))

    total_steps = math.ceil(len(dataset) / (args.batch_size * args.gradient_accumulation)) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    device = next(model.parameters()).device

    step = 0
    if len(dataset) == 0:
        print('No training examples after filtering; exiting.', flush=True)
        return
    est_steps_per_epoch = (len(dataset) + (args.batch_size - 1)) // args.batch_size
    print(f"Estimated optimizer steps per epoch (before accumulation): {est_steps_per_epoch}", flush=True)
    model.train()
    try:
        for epoch in range(args.epochs):
            print(f"Starting epoch {epoch+1}/{args.epochs}", flush=True)
            for batch in dl:
                input_ids = batch['input_ids'].to(device)
                attn = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                weights = batch['weights'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                loss = outputs.loss
                weighted_loss = loss * weights.mean()
                weighted_loss.backward()
                if (step + 1) % args.gradient_accumulation == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                if step % 20 == 0:
                    print(f"epoch={epoch} step={step} loss={loss.item():.4f} w_loss={weighted_loss.item():.4f} mean_w={weights.mean().item():.3f}", flush=True)
                step += 1
        print('Training loop finished; saving model...', flush=True)
    except Exception as e:
        import traceback
        print('ERROR during training loop:', e, flush=True)
        traceback.print_exc()
        raise
    print('Saving model...')
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if not args.no_lora:
        model.save_pretrained(os.path.join(args.output_dir, 'lora_adapter'))
    print('Done.')

if __name__ == '__main__':
    main()
