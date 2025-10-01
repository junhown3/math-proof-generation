"""Train a model with Direct Preference Optimization (DPO) on pairwise proof preferences.

Usage example:
  python -m rl.train_dpo \
      --model-name meta-llama/Llama-2-7b-hf \
      --pairwise-jsonl data/rl/pairwise.jsonl \
      --output-dir models/dpo-llama7b-v1 \
      --max-steps 300 --batch-size 2 --gradient-accumulation 8 \
      --lora-r 16 --lora-alpha 32 --lr 2e-5 --beta 0.1 --max-length 2048

This minimal script uses TRL's DPOTrainer. For Windows / no GPU, remove bitsandbytes args and 4-bit loading.
"""
from __future__ import annotations
import argparse, json, os
from dataclasses import dataclass
from typing import List, Dict

import torch

# Lazy imports (so that basic operations work even if transformers not installed yet)
try:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import DPOTrainer, DPOConfig
except ImportError as e:
    raise SystemExit("Missing required RL packages. Install transformers, datasets, accelerate, peft, trl, bitsandbytes (optional).")


def load_pairwise_dataset(path: str):
    """Create a HuggingFace dataset from JSONL of pairwise rows.
    Each line must contain: prompt, chosen, rejected.
    We wrap into a dict for DPOTrainer which expects fields matching config keys.
    """
    # We build an in-memory dataset instead of streaming for simplicity.
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            js = json.loads(line)
            prompt = js['prompt']
            chosen = js['chosen']
            rejected = js['rejected']
            # Basic sanity filtering
            if len(chosen) < 30 or len(rejected) < 30:
                continue
            records.append({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected
            })
    # Use datasets from dict
    from datasets import Dataset
    if not records:
        raise SystemExit("No valid pairwise records; aborting.")
    return Dataset.from_list(records)


SYSTEM_PREFIX = "System: You are an expert mathematician. Provide a rigorous proof or best-effort reasoning with GAP markers when steps are missing.\nUser:\n"
ASSISTANT_PREFIX = "\nAssistant:\n"


def format_example(prompt: str, answer: str, max_length: int, tokenizer) -> str:
    text = SYSTEM_PREFIX + prompt.strip() + ASSISTANT_PREFIX + answer.strip()
    # (Optional) could truncate by tokens later; here just return full string
    return text


def tokenize_batch(features: Dict[str, List[str]], tokenizer, max_length: int):
    # DPOTrainer handles tokenization internally if we just supply columns; we keep this placeholder
    return features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-name', required=True)
    ap.add_argument('--pairwise-jsonl', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--max-length', type=int, default=2048)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--gradient-accumulation', type=int, default=8)
    ap.add_argument('--max-steps', type=int, default=300)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--lora-r', type=int, default=16)
    ap.add_argument('--lora-alpha', type=int, default=32)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument('--beta', type=float, default=0.1, help='DPO beta (KL temperature)')
    ap.add_argument('--four-bit', action='store_true', help='Enable 4-bit quantization for memory savings')
    ap.add_argument('--bf16', action='store_true', help='Use bfloat16 (if supported)')
    ap.add_argument('--no-lora', action='store_true', help='Train full model (not recommended for large models)')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading pairwise dataset...")
    dset = load_pairwise_dataset(args.pairwise_jsonl)

    print(f"Loaded {len(dset)} preference rows")

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.four_bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map='auto',
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        quantization_config=quant_config
    )

    # Format dataset columns for DPOTrainer: must have prompt, chosen, rejected
    def map_row(row):
        return {
            'prompt': SYSTEM_PREFIX + row['prompt'].strip() + ASSISTANT_PREFIX,  # condition part
            'chosen': row['chosen'].strip(),
            'rejected': row['rejected'].strip()
        }

    dset_formatted = dset.map(map_row, remove_columns=dset.column_names)

    if not args.no_lora:
        print("Applying LoRA...")
        if args.four_bit:
            model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type='CAUSAL_LM'
        )
        model = get_peft_model(model, lora_cfg)

    trainer_cfg = DPOConfig(
        beta=args.beta,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        warmup_ratio=0.03,
        max_length=args.max_length,
        max_target_length=args.max_length,
        remove_unused_columns=False
    )

    print("Starting DPO training...")
    trainer = DPOTrainer(
        model=model,
        args=trainer_cfg,
        train_dataset=dset_formatted,
        tokenizer=tokenizer
    )
    trainer.train()

    print("Saving final model (and adapters if LoRA)...")
    trainer.save_model()
    if not args.no_lora:
        # Save adapter separately
        model.save_pretrained(os.path.join(args.output_dir, 'lora_adapter'))
    print("Done.")

if __name__ == '__main__':
    main()
