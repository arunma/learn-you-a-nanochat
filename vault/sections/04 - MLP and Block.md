---
aliases: [MLP, Block, feed-forward, FFN, residual]
tags: [section, phase-4]
source: nanochat/gpt.py lines 129–152
---

# 04 — MLP and Block

<span class="phase-tag">SECTION 4</span> *The feed-forward network and the two-line block that is the entire transformer layer*

> **Source:** `nanochat/gpt.py` lines 129–152
> **Copywork target:** ~24 lines (two small classes)

---

## What this code does

After attention gathers information from other tokens, the MLP processes it **per token independently**. Attention = "what should I pay attention to?" MLP = "what should I do with what I gathered?"

Then `Block` composes attention + MLP into one transformer layer with residual connections. The block's forward pass is **two lines** — the simplest and most elegant code in the entire model.

---

## Part 1: `MLP` — expand, activate, compress

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)    # 768 → 3072
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)  # 3072 → 768

    def forward(self, x):
        x = self.c_fc(x)           # expand
        x = F.relu(x).square()     # activate
        x = self.c_proj(x)         # compress
        return x
```

> [!shape] Shape trace — MLP forward
> ```
> INPUT                    OPERATION                         OUTPUT
> ────────────────────────────────────────────────────────────────────
> x: (B, T, 768)          c_fc: Linear(768, 3072)           (B, T, 3072)   expand 4×
>    (B, T, 3072)          F.relu(x)                         (B, T, 3072)   zero negatives
>    (B, T, 3072)          .square()                         (B, T, 3072)   square positives
>    (B, T, 3072)          c_proj: Linear(3072, 768)         (B, T, 768)    compress back
>
> OUTPUT: (B, T, 768) — same shape as input ✓
> ```

### The 4× expansion — why expand then compress?

768 dimensions is enough to *represent* a token, but not enough working space to *transform* its meaning. The MLP temporarily expands to 3072 dimensions — think of it as scratch paper. More room to compute intermediate features, then compress the result back to 768.

The 4× ratio is a convention from the original transformer paper. It works well empirically. nanochat follows it: `4 * config.n_embd`.

### relu² — what it is and why

```python
x = F.relu(x).square()
```

This is two operations chained:

```
relu:    max(0, x)     — zero out all negatives
square:  x²            — amplify positives, further suppress near-zero
```

Numeric example with a vector `[-2.0, 0.5, -0.1, 3.0, 0.01]`:

```
After relu:    [ 0.0,  0.5,  0.0,  3.0,  0.01]    negatives → 0
After square:  [ 0.0,  0.25, 0.0,  9.0,  0.0001]   small values crushed, large amplified
```

The squaring creates a **sharper gate** than relu alone. Values near zero (0.01 → 0.0001) are effectively killed. Strong activations (3.0 → 9.0) are amplified. The MLP becomes more selective about what information passes through.

> [!versus] nanoGPT vs nanochat — activation function
>
> | nanoGPT (your notes) | nanochat |
> |---------------------|----------|
> | `nn.GELU()` | `F.relu(x).square()` |
> | Smooth curve, gradual transition near 0 | Hard cutoff at 0, then squared |
> | Separate `self.gelu` module | Inline, no stored module |
> | Slightly more compute (exp, erf) | Cheaper (max, multiply) |

> [!keyinsight] Why no separate activation module?
> nanoGPT stored `self.gelu = nn.GELU()` as a module attribute. nanochat just calls `F.relu(x).square()` inline — no module needed. `F.relu` is stateless (no learnable parameters), so there's nothing to store. Simpler.

### The two Linear layers — naming convention

| Layer | Name | Shape | Role |
|-------|------|-------|------|
| `c_fc` | "connected, fully connected" | (768, 3072) | Expand to working space |
| `c_proj` | "connected, projection" | (3072, 768) | Project back to residual stream |

The `c_` prefix is a GPT-2 naming convention Karpathy kept. `c_proj` appears in both MLP and Attention — both times it means "project back to the residual dimension."

---

## Part 2: `Block` — the two-line transformer layer

```python
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x
```

> [!shape] Shape trace — Block forward
> ```
> INPUT                    OPERATION                              OUTPUT
> ──────────────────────────────────────────────────────────────────────────
> x: (B, T, 768)          norm(x)                                (B, T, 768)
>                          self.attn(norm(x), ve, ...)            (B, T, 768)
>                          x + attn_output                        (B, T, 768)   residual ①
>
>    (B, T, 768)           norm(x)                                (B, T, 768)
>                          self.mlp(norm(x))                      (B, T, 768)
>                          x + mlp_output                         (B, T, 768)   residual ②
>
> OUTPUT: (B, T, 768) — same shape as input ✓
> ```

### The two-line pattern — pre-norm residual

Each line follows the same pattern: `x = x + sublayer(norm(x))`

Breaking it apart:

```
1. norm(x)              — normalize to consistent scale
2. sublayer(norm(x))    — attention or MLP processes the normalized input
3. x + ...              — add the result BACK to the original x (residual)
```

The `+ x` is the **residual connection**. It means:
- The sublayer only needs to learn the **delta** (what to add), not the full transformation
- Gradients flow directly backward through the `+` (no vanishing gradient)
- If a sublayer learns nothing useful, the `+ 0` pass-through does no harm

> [!keyinsight] Pre-norm vs post-norm
> "Pre-norm" means `norm` is applied **before** the sublayer: `x + sublayer(norm(x))`.
> The original 2017 transformer used "post-norm": `norm(x + sublayer(x))`.
>
> Pre-norm is what GPT-2, GPT-3, LLaMA, and nanochat all use. It's more stable during training because the residual stream (the `x` that flows through) is never normalized — it accumulates freely. Only the *input to each sublayer* is normalized.

### Why `forward` takes 5 arguments

```python
def forward(self, x, ve, cos_sin, window_size, kv_cache):
```

The Block passes these through to attention. MLP doesn't need them — it only takes `x`:

| Arg | Used by attention? | Used by MLP? | What it is |
|-----|-------------------|-------------|------------|
| `x` | Yes | Yes | The tensor |
| `ve` | Yes | No | Value embedding for this layer |
| `cos_sin` | Yes | No | Rotary embeddings |
| `window_size` | Yes | No | Sliding window size |
| `kv_cache` | Yes | No | Inference cache |

The MLP is beautifully simple — it just sees `norm(x)` and returns a tensor. All the complexity lives in attention.

> [!versus] nanoGPT vs nanochat — Block structure
>
> | nanoGPT (your notes) | nanochat |
> |---------------------|----------|
> | `self.ln_1 = nn.LayerNorm(C)` | `norm(x)` — function call, no stored module |
> | `self.ln_2 = nn.LayerNorm(C)` | `norm(x)` — same function, no second module |
> | `x = x + self.attn(self.ln_1(x))` | `x = x + self.attn(norm(x), ve, ...)` |
> | `x = x + self.mlp(self.ln_2(x))` | `x = x + self.mlp(norm(x))` |
> | Block stores `ln_1`, `ln_2` (with learned params) | Block stores nothing for norm (no params) |
> | `forward(self, x)` — 1 arg | `forward(self, x, ve, cos_sin, ...)` — 5 args |

---

## Both classes together — the dependency chain

```
Block
├── self.attn = CausalSelfAttention    ← Section 03
│   ├── c_q, c_k, c_v, c_proj         ← Linear (Section 02)
│   ├── ve_gate                        ← Linear (Section 02)
│   ├── apply_rotary_emb()             ← Section 02
│   ├── norm()                         ← Section 02
│   └── flash_attn                     ← imported
│
├── self.mlp = MLP
│   ├── c_fc                           ← Linear (Section 02)
│   └── c_proj                         ← Linear (Section 02)
│
└── norm()                             ← Section 02 (called in forward, not stored)
```

Everything you've written so far composes into this. The Block is the top-level unit that gets repeated 12 times.

---

## The complete shape trace — one block, input to output

```
ONE COMPLETE TRANSFORMER BLOCK

INPUT
x                          (B, T, 768)              from previous block (or embeddings)
ve                         (B, T, 768) or None      value embedding for this layer
cos_sin                    ((1,T,1,64), (1,T,1,64)) precomputed rotary embeddings

ATTENTION HALF
norm(x)                    (B, T, 768)              normalize for attention
c_q(norm(x))               (B, T, 768)              Q projection
.view → q                  (B, T, 6, 128)           split into heads
c_k(norm(x))               (B, T, 768)              K projection
.view → k                  (B, T, 6, 128)           split into KV heads
c_v(norm(x))               (B, T, 768)              V projection
.view → v                  (B, T, 6, 128)           split into KV heads
[ve gating]                (B, T, 6, 128)           V enriched (if ve not None)
apply_rotary_emb(q, k)     (B, T, 6, 128)           position encoded
norm(q), norm(k) * 1.2     (B, T, 6, 128)           QK norm + sharpen
flash_attn(q, k, v)        (B, T, 6, 128)           attended values
.view(B, T, -1)            (B, T, 768)              heads reassembled
c_proj(y)                  (B, T, 768)              projected to residual
x = x + attn_out           (B, T, 768)              RESIDUAL CONNECTION ①

MLP HALF
norm(x)                    (B, T, 768)              normalize for MLP
c_fc(norm(x))              (B, T, 3072)             expand 4×
relu().square()            (B, T, 3072)             activate (relu²)
c_proj(...)                (B, T, 768)              compress back
x = x + mlp_out            (B, T, 768)              RESIDUAL CONNECTION ②

OUTPUT                     (B, T, 768)              same shape as input ✓
```

---

> [!copywork] Copywork checkpoint
> Close this note. From memory, write both classes (~24 lines):
>
> **MLP:**
> 1. `__init__`: two Linear layers — `c_fc` (expand) and `c_proj` (compress)
> 2. `forward`: three lines — expand, activate, compress
>
> **Block:**
> 1. `__init__`: store `attn` and `mlp`
> 2. `forward`: two lines — attention half, MLP half
>
> **Common traps:**
> - Did you write `4 * config.n_embd` (not a hardcoded 3072)?
> - Did you write `F.relu(x).square()` (not `F.relu(x ** 2)` or `F.relu(x) ** 2`)?
>   - `.square()` and `** 2` are equivalent, but match the original
> - Did you pass `bias=False` to both MLP Linears?
> - Did Block's `__init__` take `layer_idx` and pass it to `CausalSelfAttention`?
> - Did you write `norm(x)` not `self.norm(x)`? (It's a function, not a method)
> - Did you pass all 5 args to `self.attn(norm(x), ve, cos_sin, window_size, kv_cache)`?
> - Did you write `self.mlp(norm(x))` with only 1 arg? (MLP doesn't need the rest)

---

*Previous: [[03 - CausalSelfAttention]]*
*Next: [[05 - GPT Init]] — wte, lm_head, per-layer scalars, value embeddings, rotary buffers*

*Back to [[Index]]*
