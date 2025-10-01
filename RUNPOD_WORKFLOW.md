# RunPod + GitHub Workflow Guide

This document provides step-by-step instructions for working with this RLAIF math proof generation project on RunPod H100 instances.

## 🚀 Quick Start (Every New Pod)

### Step 1: Setup Git & SSH Authentication

**Generate SSH key on RunPod:**
```bash
ssh-keygen -t ed25519 -C "junhown3@gmail.com"
# Press Enter for all prompts (use defaults)
```

**Add public key to GitHub:**
```bash
cat ~/.ssh/id_ed25519.pub
```
- Copy the entire output (starts with `ssh-ed25519`)
- Go to GitHub.com → Settings → SSH and GPG keys → New SSH key
- Paste the key and save

### Step 2: Clone Repository and Setup Environment

**Clone project:**
```bash
git clone git@github.com:junhown3/math-proof-generation.git
cd math-proof-generation
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Verify GPU access:**
```bash
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name()}')"
```

### Step 3: Training and Development

**Run scalar weighted fine-tuning:**
```bash
python -m rl.train_scalar_weighted \
    --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --reward-file data/rl/reward_dataset_dedup_nomock.jsonl \
    --candidates-dir data/rl/candidates \
    --output-dir models/scalar-ft-runpod-v1 \
    --epochs 1 \
    --batch-size 8 \
    --gradient-accumulation 4
```

**Generate more candidates (if needed):**
```bash
python -m rl.candidate_generation --paper 2509.22618 --theorem-start 0 --theorem-end 3 --variants baseline,rag
```

**Evaluate results:**
```bash
python -m rl.evaluate_scalar --model-dir models/scalar-ft-runpod-v1 --test-file data/rl/reward_dataset_dedup_nomock.jsonl
```

### Step 4: Save Work Back to GitHub

**Add and commit changes:**
```bash
git add .
git commit -m "Training results from RunPod H100 - $(date)"
git push origin master
```

## 🔧 Alternative: HTTPS Authentication

If SSH setup is problematic, use HTTPS with Personal Access Token:

1. **Create GitHub Personal Access Token:**
   - GitHub.com → Settings → Developer settings → Personal access tokens → Generate new token
   - Select scopes: `repo` (full repository access)

2. **Clone with HTTPS:**
   ```bash
   git clone https://github.com/junhown3/math-proof-generation.git
   ```

3. **When prompted for credentials:**
   - Username: `junhown3`
   - Password: `your_personal_access_token`

## 📁 Project Structure Overview

```
├── rl/                           # RLAIF pipeline
│   ├── train_scalar_weighted.py  # Main training script
│   ├── candidate_generation.py   # Generate proof candidates
│   ├── judge_scores.py          # Score candidates with judge
│   └── evaluate_scalar.py       # Evaluation utilities
├── data/
│   ├── rl/
│   │   ├── reward_dataset_dedup_nomock.jsonl  # Clean reward dataset
│   │   └── candidates/          # Generated candidate proofs
│   └── papers/                  # ArXiv paper data
├── models/                      # Training outputs (gitignored)
└── requirements.txt            # Python dependencies
```

## ⚡ Performance Tips

**Memory optimization:**
- Use `--gradient-accumulation 4` or higher if OOM
- Consider `--batch-size 4` for large models
- Monitor GPU memory: `watch -n 1 nvidia-smi`

**Training monitoring:**
```bash
# Tail training logs
tail -f training.log

# Check training progress
ls -la models/scalar-ft-runpod-v1/
```

## 🐛 Troubleshooting

**SSH authentication fails:**
- Verify key was added to GitHub: `ssh -T git@github.com`
- Should see: "Hi junhown3! You've successfully authenticated"

**Git push fails:**
```bash
git config user.name "junhown3"
git config user.email "junhown3@gmail.com"
```

**CUDA/GPU issues:**
- Restart pod if GPU not detected
- Verify CUDA version: `nvcc --version`

**Training crashes:**
- Check logs in training output directory
- Reduce batch size if OOM
- Verify dataset format with: `head -n 5 data/rl/reward_dataset_dedup_nomock.jsonl`

## 🎯 Typical Session Workflow

1. **Start RunPod H100 instance**
2. **SSH setup** (steps 1-2 above)
3. **Run training** (step 3)
4. **Monitor progress** (`nvidia-smi`, check logs)
5. **Save results** (step 4)
6. **Terminate pod** (all work saved to GitHub)

## 📊 Next Steps After Training

1. **Download trained model locally** (via git pull)
2. **Run evaluation scripts** to compare baseline vs fine-tuned
3. **Scale up**: Add more papers/theorems to dataset
4. **Iterate**: Adjust hyperparameters based on results

---

*Last updated: October 2025*
*Repository: https://github.com/junhown3/math-proof-generation*