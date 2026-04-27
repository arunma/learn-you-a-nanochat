# Learn You a Nanochat

A hands-on tutorial for learning LLM internals through **copywork** — reading, understanding, and rewriting [Karpathy's nanochat](https://github.com/karpathy/nanochat) from memory.

## What is this?

This repo contains:

1. **Karpathy's nanochat** — the original GPT implementation (`nanochat/`) as a read-only reference
2. **An Obsidian vault** — a structured tutorial that walks through every line of the model (`vault/`)
3. **Your code** — a blank `gpt.py` that you build piece by piece through copywork (`my_nanochat/`)

The core model is **512 lines** in `nanochat/gpt.py`. By the end, you'll have written your own drop-in replacement.

## The copywork method

Copywork is how musicians learn solos (transcribe by ear), how artists learn technique (master copies), and how Benjamin Franklin learned to write (reconstructing essays from memory).

For code:

1. **Read** a section of the tutorial — understand *why* each line exists
2. **Close** the reference
3. **Write** the code from memory
4. **Diff** against the original — gaps reveal what you don't actually understand
5. **Repeat** until it sticks, then move on

## Getting started

### Prerequisites

- Python 3.10+
- [Obsidian](https://obsidian.md/) (free) for the tutorial vault
- Basic familiarity with Python and PyTorch

### Setup

```bash
# Clone this repo
git clone https://github.com/arunma/learn-you-a-nanochat.git
cd learn-you-a-nanochat

# Set up Python environment
python -m venv .venv
source .venv/bin/activate
pip install torch

# Open the vault in Obsidian
# File → Open Vault → select the vault/ folder
```

### Enable the color theme

In Obsidian: **Settings → Appearance → CSS Snippets → enable `nanochat-theme`**

This activates the color-coded callouts and badges used throughout the tutorial.

## Repo structure

```
learn-you-a-nanochat/
│
├── nanochat/                ← Karpathy's code (READ-ONLY reference)
│   ├── gpt.py               ← THE file — 512 lines, the entire model
│   ├── tokenizer.py          ← BPE tokenizer
│   ├── dataloader.py         ← data loading
│   ├── optim.py              ← AdamW + Muon optimizer
│   ├── engine.py             ← inference with KV cache
│   └── ...
│
├── my_nanochat/             ← YOUR code (write through copywork)
│   ├── __init__.py
│   └── gpt.py               ← starts empty, grows section by section
│
├── vault/                   ← Obsidian tutorial vault
│   ├── Index.md              ← start here
│   ├── sections/
│   │   ├── 00 - The Big Picture.md    ← overview: 5 stages of LLM building
│   │   ├── 01 - Config and Imports.md ← first copywork section
│   │   └── ...                        ← one file per code chunk
│   ├── reference/
│   │   ├── Shape Cheatsheet.md        ← every tensor dimension
│   │   ├── Glossary.md               ← all terms defined
│   │   ├── nanoGPT vs nanochat.md     ← what changed and why
│   │   └── PyTorch API Reference.md   ← every PT built-in used
│   └── copywork/
│       └── (scratch practice files)
│
├── scripts/                 ← Karpathy's training/eval scripts
├── tasks/                   ← evaluation tasks
└── NANOCHAT_README.md       ← Karpathy's original README
```

## Tutorial sections

The vault walks through `nanochat/gpt.py` in order:

| Section | Lines | What you learn |
|---------|-------|----------------|
| **00 — The Big Picture** | — | The 5 stages of LLM building, all 7 building blocks |
| **01 — Config and Imports** | 1–40 | GPTConfig, imports, the model's DNA |
| **02 — Building Blocks** | 42–63 | `norm()`, custom `Linear`, rotary embeddings helper |
| **03 — CausalSelfAttention** | 65–127 | Q/K/V, RoPE, flash attention, GQA |
| **04 — MLP and Block** | 129–152 | relu², residual connections, the two-half block |
| **05 — GPT Init** | 154–199 | wte, lm_head, per-layer scalars, value embeddings |
| **06 — Weight Initialization** | 201–267 | How every parameter starts |
| **07 — Rotary Embeddings Deep Dive** | 268–283 | RoPE from scratch |
| **08 — Sliding Window Attention** | 285–312 | Per-layer window patterns |
| **09 — Forward Pass** | 416–481 | The complete data flow |
| **10 — Smear Gate** | 433–449 | Cheap bigram trick |
| **11 — Optimizer Setup** | 374–415 | AdamW + Muon, parameter groups |
| **12 — Generation** | 483–513 | Autoregressive decoding, sampling |

## Color coding

The tutorial uses consistent color coding:

| Badge | Color | Meaning |
|-------|-------|---------|
| `[PT]` | Blue | PyTorch built-in — learn the API |
| `[NC]` | Green | nanochat custom — understand every line |
| `[SHAPE]` | Orange | Tensor dimensions at this point |
| `[NEW]` | Purple | New in nanochat vs the simpler nanoGPT |

## Key numbers

| Symbol | Name | Value |
|--------|------|-------|
| `T` | sequence_len | 2048 |
| `V` | vocab_size | 32,768 |
| `N` | n_layer | 12 |
| `C` | n_embd | 768 |
| `n_head` | query heads | 6 |
| `d_h` | head_dim | 128 |

## Running your model

Once `my_nanochat/gpt.py` is complete, test it:

```bash
# Verify your model matches the original architecture
python -c "
from my_nanochat.gpt import GPT, GPTConfig
model = GPT(GPTConfig())
params = sum(p.numel() for p in model.parameters())
print(f'Parameters: {params:,}')
"
```

## Credits

- **nanochat** by [Andrej Karpathy](https://github.com/karpathy/nanochat) — the model we're studying
- Tutorial structure and vault by this repo's contributors
- The copywork method draws from traditions in music, art, and stenography
