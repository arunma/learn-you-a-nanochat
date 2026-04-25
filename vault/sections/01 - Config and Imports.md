---
aliases: [config, GPTConfig, imports]
tags: [section, phase-1]
source: nanochat/gpt.py lines 1–40
---

# 01 — Config and Imports

<span class="phase-tag">SECTION 1</span> *The model's DNA — what the transformer is made of before any computation happens*

> **Source:** `nanochat/gpt.py` lines 1–40
> **Copywork target:** ~40 lines (the docstring, imports, and GPTConfig dataclass)

---

## What this code does

Before any tensor flows through any layer, the model needs to know its own dimensions — how wide, how deep, how many heads, how long the context window is. `GPTConfig` is that blueprint. Every other class in the file reads from it.

The imports pull in PyTorch and three nanochat-specific modules. The docstring at the top is worth reading because it lists every architectural decision that makes nanochat different from a textbook transformer.

---

## The docstring — a roadmap of design choices

```python
"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""
```

> [!keyinsight] Every line in this docstring is a departure from the textbook
> Your earlier notes described a model with learned positional embeddings, GELU, weight tying, LayerNorm with learnable params, and biases. nanochat flips **every single one** of these. Each change is individually small but the cumulative effect is a significantly different (and better) model.

Let's map each bullet to what you already know:

| Docstring line | Your notes said | nanochat does instead |
|---------------|----------------|----------------------|
| rotary embeddings | `wpe = nn.Embedding(T, C)` — learned position vectors | RoPE — rotation applied to Q and K, no parameters |
| QK norm | not mentioned | `norm(q)`, `norm(k)` before attention scores |
| untied weights | `lm_head.weight = wte.weight` (shared) | separate parameters |
| relu^2 | `nn.GELU()` | `F.relu(x).square()` |
| norm after embedding | norm before attn and MLP only | additional `norm(x)` right after `wte(idx)` |
| no learnable params in rmsnorm | `nn.LayerNorm` has `gamma` and `beta` | `F.rms_norm` — just divide, no learned rescaling |
| no bias | `bias=True` default | `bias=False` everywhere |
| GQA | all heads are equal | fewer KV heads than Q heads |
| Flash Attention 3 | manual `q @ k.T` → mask → softmax → `@ v` | one fused kernel call |

---

## The imports

```python
from functools import partial                          # NC — Python stdlib
from dataclasses import dataclass                      # NC — Python stdlib

import torch                                           # PT
import torch.nn as nn                                  # PT
import torch.nn.functional as F                        # PT

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE   # NC
from nanochat.optim import MuonAdamW, DistMuonAdamW                # NC
from nanochat.flash_attention import flash_attn                     # NC
```

> [!pytorch] The three PyTorch imports
> These three lines appear at the top of virtually every PyTorch file:
> - `torch` — tensor creation, device management, utilities
> - `torch.nn` — layer classes (`nn.Linear`, `nn.Embedding`, `nn.Module`)
> - `torch.nn.functional` — stateless operations (`F.relu`, `F.cross_entropy`, `F.rms_norm`)
>
> Rule of thumb: `nn.Something` = has parameters (learned). `F.something` = pure function (no state).

> [!nanochat] The three nanochat imports
> - `get_dist_info` — returns `(ddp, rank, local_rank, world_size)` for distributed training
> - `print0` — prints only on rank 0 (avoids duplicate output across GPUs)
> - `COMPUTE_DTYPE` — auto-detected: `bfloat16` on H100/A100, `float32` on CPU/MPS
> - `MuonAdamW` / `DistMuonAdamW` — hybrid optimizer (AdamW for embeddings, Muon for matrices)
> - `flash_attn` — wrapper that picks Flash Attention 3 (Hopper GPUs) or PyTorch SDPA (fallback)

---

## GPTConfig — the model's blueprint

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048        # T — context window (tokens per sequence)
    vocab_size: int = 32768         # V — number of tokens in vocabulary
    n_layer: int = 12               # N — how many transformer blocks
    n_head: int = 6                 # number of query heads
    n_kv_head: int = 6              # number of key/value heads (GQA)
    n_embd: int = 768               # C — embedding dimension (channels)
    # Sliding window attention pattern string, tiled across layers.
    # Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
```

> [!shape] Derived dimensions (not in the config, but computed from it)
> - `head_dim = n_embd // n_head` = `768 // 6` = **128** dimensions per head
> - `4 * n_embd` = **3072** — MLP hidden dimension (the 4x expansion)
> - `n_kv_head * head_dim` = `6 * 128` = **768** — total KV dimension

### Field-by-field breakdown

#### `sequence_len: int = 2048`

The context window — how many tokens the model can see at once. Your notes used `T = 1024` (nanoGPT). nanochat doubles it to 2048.

Every token at position `t` can attend to tokens at positions `0` through `t` (causal mask). Longer context = more information available per prediction, but attention is quadratic in T, so doubling context roughly quadruples attention memory.

#### `vocab_size: int = 32768`

Your notes used 50,257 (GPT-2's tokenizer). nanochat trains its own BPE tokenizer with 32,768 tokens. Smaller vocabulary means:
- Faster softmax at the output (32K scores vs 50K)
- Smaller embedding table (32K rows vs 50K rows)
- Slightly longer token sequences (fewer chars per token on average)

> [!keyinsight] 32768 = 2^15
> Powers of 2 are preferred for vocab sizes because GPU tensor cores operate on tiles of 8, 16, 32, 64, etc. A vocab that's a power of 2 means no wasted compute in the final matrix multiply. nanochat also pads the vocab to the nearest multiple of 64 at runtime for this reason.

#### `n_layer: int = 12`

Depth — number of transformer blocks stacked. Each block is one round of attention + MLP. The shape `(B, T, C)` is preserved through every block. More layers = more rounds of "thinking" = better quality but more compute.

#### `n_head: int = 6` and `n_kv_head: int = 6`

> [!qkv] Group Query Attention (GQA)
> In your notes, every head had its own Q, K, and V — `n_head` copies of each.
>
> GQA separates this: **Q gets `n_head` copies, but K and V get only `n_kv_head` copies.** When `n_kv_head < n_head`, multiple Q heads share the same K/V head.
>
> **Why?** K and V are the expensive part during inference (they're cached in the KV cache). Fewer KV heads = smaller cache = faster inference. Quality barely drops.
>
> With the defaults (`n_head=6`, `n_kv_head=6`), it's standard MHA — no sharing. But the code supports GQA if you set `n_kv_head` lower.

#### `n_embd: int = 768`

The embedding dimension — `C` in shape traces. Every token is represented as a 768-dimensional float vector. This is the "width" of the model. Your notes used 384.

`n_embd` must be divisible by `n_head` (so head_dim is an integer): `768 / 6 = 128`.

#### `window_pattern: str = "SSSL"`

> [!nanochat] Sliding window attention — new concept
> Not every layer needs to see the full 2048-token context. `"SSSL"` means:
> - Layer 0: **S**hort window (quarter context = 512 tokens)
> - Layer 1: **S**hort
> - Layer 2: **S**hort
> - Layer 3: **L**ong (full 2048)
> - Layer 4: **S**hort (pattern repeats)
> - ...
> - Layer 11 (final): always **L**ong regardless of pattern
>
> Short-window layers save memory (attention is quadratic). The model learns to use short-range attention for local patterns and reserves full context for the layers that need it.

---

## Versus your earlier notes

| Your notes (nanoGPT) | nanochat | What changed |
|----------------------|----------|-------------|
| `vocab_size = 50257` | `vocab_size = 32768` | Custom tokenizer, power-of-2 |
| `block_size = 1024` | `sequence_len = 2048` | Doubled context window |
| `n_embd = 384` | `n_embd = 768` | Doubled width |
| `n_head = 6` only | `n_head = 6` + `n_kv_head = 6` | GQA support (separate Q vs KV heads) |
| No window pattern | `window_pattern = "SSSL"` | Per-layer sliding window attention |
| `@dataclass` with `block_size`, `dropout` | No `dropout` field, uses `sequence_len` | Simpler config, dropout removed |

---

> [!copywork] Copywork checkpoint
> Close this note. From memory, write:
> 1. The docstring (all 10 bullet points — what makes nanochat different)
> 2. The imports (3 stdlib/PT, 3 nanochat)
> 3. The `GPTConfig` dataclass with all 7 fields and the window_pattern comment
>
> **Check yourself:** Did you remember `n_kv_head`? Did you get `32768` not `50257`? Did you write `sequence_len` not `block_size`?

---

*Next: [[02 - Building Blocks]] — `norm()`, the custom `Linear`, and `apply_rotary_emb()`*

*Back to [[Index]]*
