---
aliases: [speedrun, training pipeline, end to end, runcpu]
tags: [section, pipeline]
source: runs/speedrun.sh
---

# 13 — Speedrun Walkthrough

<span class="phase-tag">PIPELINE</span> *The complete end-to-end pipeline — from raw text to chatbot*

> **Source:** `runs/speedrun.sh` (~98 lines)
> **Cost:** ~$30-50 on spot 8xH100, ~$0 on CPU/MPS (tiny model)

---

## What the speedrun does

The speedrun takes a blank GPU node and produces a GPT-2-grade chatbot in ~3 hours. Five stages, fully automated:

```
 STAGE 1          STAGE 2            STAGE 3           STAGE 4        STAGE 5
 Setup        →   Tokenizer      →   Pretraining   →   Fine-tune  →   Chat
 (2 min)          (10 min)           (90-120 min)      (20-30 min)    (interactive)
 install deps     train BPE          8×H100 DDP        SFT on tasks   web UI
 create venv      vocab=32768        depth=24, FP8     MMLU/GSM8K     talk to it
```

---

## Stage-by-stage breakdown

### Stage 1 — Environment setup (~2 minutes)

```bash
# Install uv package manager (if needed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# Create venv and install all dependencies
uv venv && uv sync --extra gpu
source .venv/bin/activate
```

Nothing interesting — just environment prep. The `--extra gpu` flag installs CUDA-specific packages (flash-attn, kernels, etc.)

### Stage 2 — Tokenizer (~10 minutes)

```bash
# Download first 8 shards (~800MB) for tokenizer training
python -m nanochat.dataset -n 8

# Start downloading 170 shards in background (for pretraining later)
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!

# Train BPE tokenizer: vocab_size = 2^15 = 32768
python -m scripts.tok_train

# Evaluate compression ratio
python -m scripts.tok_eval
```

> [!keyinsight] Parallelism trick
> Tokenizer training only needs 8 shards (~2B chars). But pretraining needs 170 shards. The script kicks off the 170-shard download **in the background** while the tokenizer trains on the first 8. By the time pretraining starts, most shards are already downloaded.

**What this produces:**
- A trained BPE tokenizer with 32,768 tokens
- Saved to `~/.cache/nanochat/`
- This is the tokenizer your `GPTConfig.vocab_size = 32768` was designed for

### Stage 3 — Pretraining (~90-120 minutes) — THE EXPENSIVE PART

```bash
# Wait for all data shards to finish downloading
wait $DATASET_DOWNLOAD_PID

# Train the base model on 8 GPUs
torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
    -- --depth=24 \
       --target-param-data-ratio=8 \
       --device-batch-size=16 \
       --fp8 \
       --run=$WANDB_RUN

# Evaluate: CORE metric, BPB, sample generation
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval \
    -- --device-batch-size=16
```

**Key flags explained:**

| Flag | Value | What it does |
|------|-------|-------------|
| `--nproc_per_node=8` | 8 | Use all 8 H100s via DDP (Distributed Data Parallel) |
| `--depth=24` | 24 | 24 transformer blocks (your copywork used 12) |
| `--target-param-data-ratio=8` | 8 | Slightly undertrained (Chinchilla optimal ~10-12) |
| `--device-batch-size=16` | 16 | 16 sequences per GPU per step |
| `--fp8` | — | FP8 precision (2x speedup, H100/Hopper only) |
| `--run` | name | Wandb experiment name (optional logging) |

> [!nanochat] The `--depth` dial
> `--depth=24` auto-computes all other hyperparameters:
> - `model_dim = depth × aspect_ratio` → wider model
> - `n_heads = model_dim // head_dim` → more heads
> - `learning_rate` schedule adjusted automatically
> - `num_iterations` from data:param ratio
>
> The architecture is **identical** to what you wrote — just scaled up. Your code handles any depth.

> [!shape] Model size at depth=24
> ```
> depth=12 (your copywork):  ~286M params
> depth=24 (speedrun):       ~800M+ params (auto-computed width)
> ```

**What this produces:**
- A base model checkpoint in `~/.cache/nanochat/`
- CORE score ≥ 0.2565 (GPT-2 parity)
- The model can predict next tokens but doesn't know how to chat yet

### Stage 4 — Supervised Fine-Tuning (~20-30 minutes)

```bash
# Download personality conversations (2.3MB)
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# Fine-tune on conversation tasks
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft \
    -- --device-batch-size=16 --run=$WANDB_RUN

# Evaluate chat performance
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

SFT trains on a mixture of tasks:
- **MMLU** (3 epochs) — multiple choice broad knowledge
- **GSM8K** (4 epochs) — grade-school math with tool use
- **SmolTalk** — conversational data
- **Identity conversations** — custom personality ("I am nanochat, created by...")

**What this produces:**
- A fine-tuned checkpoint that can have conversations
- ChatCORE evaluation score

### Stage 5 — Chat with your model

```bash
# CLI chat (single prompt)
python -m scripts.chat_cli -p "Why is the sky blue?"

# Interactive CLI chat
python -m scripts.chat_cli

# Web UI (ChatGPT-like interface on localhost:8000)
python -m scripts.chat_web
```

---

## Testing locally first (free)

Before spending money on H100s, test the full pipeline on your Mac:

```bash
bash runs/runcpu.sh
```

This runs the same stages with a **tiny model** (small depth, CPU/MPS, no FP8). The text output will be garbage, but it verifies every script works end-to-end.

---

## GPU rental options

| Provider | 8xH100 spot | On-demand | Setup difficulty |
|----------|-------------|-----------|-----------------|
| **Lambda Labs** | ~$20-24/hr | ~$28/hr | Easy — pre-built PyTorch images |
| **RunPod** | ~$22/hr | ~$32/hr | Easy — web UI, spot available |
| **Vast.ai** | ~$15-20/hr | varies | Cheapest — peer-to-peer, less reliable |
| **AWS p5.48xlarge** | ~$25/hr spot | ~$98/hr | Most reliable, hardest capacity |
| **GCP a3-highgpu-8g** | ~$25/hr spot | ~$98/hr | Similar to AWS |
| **CoreWeave** | ~$24/hr | ~$36/hr | ML-focused, good availability |

**Total speedrun cost:**
- Spot instance: **~$30-50** (~2 hours × $15-24/hr)
- On-demand: **~$56-72**

**Recommended:** Lambda Labs or RunPod for simplicity.

### Running on a rented node

```bash
# SSH into your node, then:
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# Option A: Full speedrun with wandb logging
WANDB_RUN=my-run screen -L -Logfile speedrun.log -S speedrun bash runs/speedrun.sh

# Option B: Full speedrun without wandb
screen -L -Logfile speedrun.log -S speedrun bash runs/speedrun.sh

# Detach from screen: Ctrl+A, then D
# Reattach later: screen -r speedrun
# Watch logs: tail -f speedrun.log
```

> [!keyinsight] Use `screen` or `tmux`
> The speedrun takes ~3 hours. If your SSH connection drops, the run dies. `screen` keeps it alive in the background. Always use it for long training runs.

---

*Back to [[Index]]*
