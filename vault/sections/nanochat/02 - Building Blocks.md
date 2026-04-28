---
aliases: [building blocks, norm, linear, rotary, RoPE, has_ve]
tags: [section, phase-2]
source: nanochat/gpt.py lines 42–63
---

# 02 — Building Blocks

<span class="phase-tag">SECTION 2</span> *Four helper pieces that every other class depends on*

> **Source:** `nanochat/gpt.py` lines 42–63
> **Copywork target:** ~22 lines (4 small functions/classes)

---

## What this code does

Before the main model classes, nanochat defines four small building blocks. They're used everywhere — `norm()` is called 15+ times, `Linear` replaces every `nn.Linear`, `apply_rotary_emb` runs on every Q and K tensor. Understanding these four pieces means understanding the ingredients before the recipe.

---

## Piece 1: `norm()` — RMSNorm in one line

```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

> [!shape] Shape trace
> ```
> INPUT                    OPERATION                         OUTPUT
> ────────────────────────────────────────────────────────────────────
> x: (B, T, 768)          F.rms_norm along last dim         (B, T, 768)
>                          ↑ for each 768-dim vector independently:
>                          ↑   rms = sqrt(mean(x²) + eps)
>                          ↑   output = x / rms
>                          ↑ shape unchanged, values rescaled
> ```

### What RMSNorm actually computes

Given a vector `[200, -400, 100, 300]`:

```
Step 1: square each value     → [40000, 160000, 10000, 90000]
Step 2: take the mean         → 75000
Step 3: take the square root  → 273.9
Step 4: divide original by it → [0.73, -1.46, 0.37, 1.10]
```

The wild range `[-400, 300]` becomes a consistent `[-1.46, 1.10]`. Every vector entering the next layer has roughly unit scale.

### How it differs from your notes

> [!versus] nanoGPT vs nanochat
>
> | nanoGPT (your notes) | nanochat |
> |---------------------|----------|
> | `nn.LayerNorm(384)` | `F.rms_norm(x, (x.size(-1),))` |
> | Subtracts mean, divides by std | Only divides by RMS (no mean subtraction) |
> | Learnable `gamma` and `beta` parameters | No learnable parameters at all |
> | Called as `self.ln_1(x)` (module) | Called as `norm(x)` (plain function) |
> | Two parameter tensors per norm | Zero parameter tensors per norm |

> [!keyinsight] Why the trailing comma in `(x.size(-1),)`?
> `F.rms_norm` expects a **tuple** for the normalized shape. `(768)` is just the number 768 in parentheses. `(768,)` is a one-element tuple. The comma makes it a tuple. Miss it and you get an error.

---

## Piece 2: `Linear` — custom weight casting

```python
class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))
```

> [!shape] Shape trace
> ```
> INPUT                    OPERATION                              OUTPUT
> ──────────────────────────────────────────────────────────────────────────
> x: (B, T, in)  bf16     self.weight: (out, in)  fp32
>                          .to(dtype=x.dtype) → (out, in) bf16    cast only
>                          F.linear(x, weight) = x @ W.T          matmul
>                          result: (B, T, out) bf16
>
> Example: Linear(768, 3072, bias=False)
> x: (B, T, 768) bf16     weight: (3072, 768) fp32 → bf16
>                          (B, T, 768) @ (768, 3072) = (B, T, 3072)  bf16
> ```

### Why not just use `nn.Linear`?

The optimizer needs full float32 precision to track tiny gradient updates. But matrix multiplies are 2x faster in bfloat16 on modern GPUs. This class bridges both needs:

- **Weights stored in fp32** — optimizer gets full precision
- **Weights cast to bf16 at forward time** — GPU gets fast matmuls
- **No autocast needed** — explicit is better than implicit

> [!pytorch] `F.linear(x, weight)` vs `nn.Linear`
> `F.linear` is the raw function: `output = x @ weight.T` (plus optional bias).
> `nn.Linear` is the module wrapper that stores the weight and calls `F.linear`.
> This custom class inherits `nn.Linear`'s weight storage but overrides the forward pass.

> [!keyinsight] No bias — anywhere
> Every `Linear(...)` in nanochat is called with `bias=False`. The custom class doesn't even handle bias — the `F.linear` call has no third argument. One less thing per layer, cleaner gradients.

### Deep dive: where does the fp32 weight actually live?

`Linear` inherits from `nn.Linear`, so `self.weight` is a standard `nn.Parameter` — created by `nn.Linear.__init__()`, stored in fp32 by default.

The fp32 copy **never leaves the module**. What happens each forward pass:

```python
self.weight                        # fp32, (out, in) — the permanent master copy
self.weight.to(dtype=x.dtype)      # bf16, NEW temporary tensor — discarded after matmul
```

`.to()` creates a **new tensor**. It does not modify `self.weight`. Once the matmul finishes, the temporary bf16 copy is garbage-collected. The fp32 master sits untouched, ready for the optimizer.

```
GPU VRAM
┌──────────────────────────────────────────────────┐
│  Linear module                                    │
│  ┌──────────────────────────────────────┐         │
│  │ self.weight  (fp32, persistent)      │ ◄────── optimizer holds reference
│  └──────────────────────────────────────┘         │  and updates in-place
│                                                    │
│  During forward():                                 │
│  ┌──────────────────────────────────────┐         │
│  │ temp = self.weight.to(bf16)          │ ← born  │
│  │ result = x @ temp.T                  │         │
│  │ (temp is garbage-collected)          │ ← dies  │
│  └──────────────────────────────────────┘         │
└──────────────────────────────────────────────────┘
```

**Memory cost per parameter:**
- `self.weight` in fp32 → 4 bytes (persistent)
- `temp_bf16` during forward → 2 bytes (briefly, then freed)
- AdamW optimizer: `m` (momentum) + `v` (variance) → 8 more bytes (both fp32)
- **Total: ~14 bytes per parameter during training**

This is why training is so memory-hungry compared to inference (which only needs the weights).

For a 1B-parameter model: ~14 GB just for weights + optimizer state, before counting activations and gradients.

nanochat at ~124M params: comfortably under 2 GB.

---

## Piece 3: `has_ve()` — which layers get value embeddings

```python
def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding
    (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2
```

### What are value embeddings?

Standard attention computes V (values) from the previous layer's hidden states:

```python
V = c_v(x)    # x is activations from the previous layer — dynamic, context-dependent
```

**Value embeddings** add a second source for V — a **learned lookup table indexed directly by token ID**, just like the input embedding table:

```python
# Standard attention:
V_standard = c_v(x)                 # dynamic — depends on context

# With value embeddings:
V_static = value_embed[token_ids]   # static — depends only on token identity
V_combined = V_standard + V_static  # mix both sources
```

For each token ID, there's a learned "what content does this token contribute to attention" vector that's the **same regardless of context or layer depth**.

### The key intuition: shortcut path for token identity

In a deep transformer, by layer 6 or 8, the activations have been transformed so much that the original "this is the token 'cat'" signal has become diffused into abstract representations. Value embeddings give attention a **direct path back to raw token identity** at any layer that has them.

Similar in spirit to residual connections — a way to preserve original information that would otherwise get lost in deep stacks.

### Where does this come from? — ResFormer (2024)

This technique is from **"Value Residual Learning For Alleviating Attention Concentration In Transformers"** (Zhou et al., 2024) — sometimes called **ResFormer**.

Their findings:
1. In deep transformers, attention tends to **concentrate on too few tokens** in later layers (loss of diversity)
2. Adding direct value paths from input embeddings **improves attention diversity** and overall quality
3. You don't need it on every layer — alternating works almost as well at half the cost

This was popularized by the **modded-nanoGPT speedrun community** (Keller Jordan and others). nanochat inherits it as a modern best practice.

> [!keyinsight] This is a 2024 technique
> You haven't seen this before because most tutorials still teach GPT-2/LLaMA-style architecture without it. nanochat is intentionally on the bleeding edge of "modern minimalism."

### Where VEs sit in the architecture

```
Token IDs ──► Token Embed ──► Layer 0  (no VE)
                               Layer 1  (VE) ◄── Value Embed Table 1 [vocab × kv_dim]
                               Layer 2  (no VE)
                               Layer 3  (VE) ◄── Value Embed Table 3 [vocab × kv_dim]
                               Layer 4  (no VE)
                               Layer 5  (VE) ◄── Value Embed Table 5 [vocab × kv_dim]
                               ...
                               Layer 11 (VE) ◄── Value Embed Table 11 [vocab × kv_dim]
                               ──► lm_head ──► logits
```

Each VE-enabled layer has its **own** embedding table — they're not shared. So at layer 3, attention can "look up" what token 'cat' contributes specifically *at that depth*, which differs from what it contributes at layer 11.

### The `has_ve` pattern — with `n_layer = 12`:

```
(n_layer - 1) % 2 = 11 % 2 = 1    ← the "target" parity

Layer  0:   0 % 2 = 0   ≠ 1  →  False
Layer  1:   1 % 2 = 1   = 1  →  True   ✓ value embedding
Layer  2:   2 % 2 = 0   ≠ 1  →  False
Layer  3:   3 % 2 = 1   = 1  →  True   ✓ value embedding
Layer  4:   4 % 2 = 0   ≠ 1  →  False
Layer  5:   5 % 2 = 1   = 1  →  True   ✓ value embedding
Layer  6:   6 % 2 = 0   ≠ 1  →  False
Layer  7:   7 % 2 = 1   = 1  →  True   ✓ value embedding
Layer  8:   8 % 2 = 0   ≠ 1  →  False
Layer  9:   9 % 2 = 1   = 1  →  True   ✓ value embedding
Layer 10:  10 % 2 = 0   ≠ 1  →  False
Layer 11:  11 % 2 = 1   = 1  →  True   ✓ value embedding (last layer — always)
```

**6 out of 12 layers** get value embeddings. The formula guarantees the **last layer always gets one** regardless of whether `n_layer` is odd or even.

### Why the formula generalizes

```
n_layer = 12 (even): target = 11 % 2 = 1 → odd layers get VE → last (11) is odd  ✓
n_layer = 13 (odd):  target = 12 % 2 = 0 → even layers get VE → last (12) is even ✓
```

A simpler `layer_idx % 2 == 1` would only work for even `n_layer`. The `(n_layer - 1) % 2` trick generalizes.

> [!nanochat] Why alternating, not every layer?
> **Cost:** Each VE table is `vocab_size × kv_dim` parameters. For nanochat (32768 × 768): ~25M params per table. Adding one to all 12 layers = ~300M extra parameters — more than doubling the model.
>
> **The alternating compromise:** Half the layers get them → half the parameter overhead. Empirically, performance is nearly identical to every layer. The last layer is guaranteed because it feeds directly into `lm_head` — having a clean token-identity signal there is most valuable.

---

## Piece 4: `apply_rotary_emb()` — RoPE in 6 lines

This replaces learned positional embeddings (`wpe`) from your earlier notes. It's the biggest new concept in this section.

```python
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention: (B, T, n_head, d_h)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
```

> [!shape] Complete shape trace
> ```
> INPUT
> x:   (B, T, n_head, 128)     ← the Q or K tensor (4D)
> cos: (1, T, 1, 64)           ← precomputed cosines (broadcasts)
> sin: (1, T, 1, 64)           ← precomputed sines (broadcasts)
>
> SPLIT LAST DIM IN HALF
> d = 128 // 2 = 64
> x1 = x[..., :64]             (B, T, n_head, 64)    first half
> x2 = x[..., 64:]             (B, T, n_head, 64)    second half
>
> ROTATE (element-wise multiply, cos/sin broadcast across B and n_head)
> y1 = x1 * cos + x2 * sin     (B, T, n_head, 64)    rotated first half
> y2 = x1 * (-sin) + x2 * cos  (B, T, n_head, 64)    rotated second half
>
> REASSEMBLE
> torch.cat([y1, y2], dim=3)    (B, T, n_head, 128)   back to full head_dim
>
> OUTPUT: same shape as input — only the VALUES changed, not the dimensions
> ```

### The intuition: rotation as position encoding

Imagine each pair of numbers in the head dimension as a 2D point on a circle. RoPE **rotates** that point by an angle that depends on position in the sequence.

```
Position 0: rotate by 0θ     (no rotation)
Position 1: rotate by 1θ
Position 2: rotate by 2θ
Position 5: rotate by 5θ
Position 100: rotate by 100θ
```

The rotation angle increases with position. Different dimension pairs rotate at different speeds (θ varies per pair — low dimensions rotate fast, high dimensions rotate slowly).

### Why rotation gives RELATIVE position

When attention computes `Q · K` (dot product), both Q and K have been rotated:
- Q at position 5: rotated by 5θ
- K at position 3: rotated by 3θ

The dot product of two rotated vectors depends on the **difference** in their angles: `5θ - 3θ = 2θ`. The absolute positions (5 and 3) disappear. Only the **distance** (2) matters.

This is relative positional encoding — the model learns "how far apart are these tokens" rather than "this token is at position 47."

### The 2D rotation formula

Each pair `(x1, x2)` is rotated by angle θ using the standard 2D rotation:

```
y1 =  x1 * cos(θ) + x2 * sin(θ)
y2 = -x1 * sin(θ) + x2 * cos(θ)
```

This is exactly the matrix:
```
┌ cos(θ)   sin(θ) ┐   ┌ x1 ┐   ┌ y1 ┐
│                  │ × │    │ = │    │
└ -sin(θ)  cos(θ) ┘   └ x2 ┘   └ y2 ┘
```

nanochat does this element-wise across all 64 dimension pairs simultaneously. No loop. Pure broadcasting.

> [!versus] nanoGPT vs nanochat — positional encoding
>
> | nanoGPT (your notes) | nanochat |
> |---------------------|----------|
> | `wpe = nn.Embedding(1024, 384)` | No wpe at all |
> | Learned vectors, one per position | Precomputed sin/cos, applied to Q and K |
> | Added to embeddings once: `x = wte(idx) + wpe(pos)` | Applied inside every attention layer |
> | Absolute position: "I am at position 47" | Relative position: "I am 5 tokens away from you" |
> | ~393K learnable parameters | Zero learnable parameters |
> | Limited to trained context length | Generalizes to longer sequences (with some decay) |

> [!keyinsight] Why `x.ndim == 4` is asserted
> RoPE is applied to Q and K **after** they've been reshaped to `(B, T, n_head, d_h)` but **before** the transpose to `(B, n_head, T, d_h)`. The cos/sin tensors are shaped `(1, T, 1, d_h//2)` to broadcast across batch (dim 0) and heads (dim 2). If you accidentally call this on a 3D tensor, the broadcast would silently produce wrong results.

---

## All four pieces together — the dependency map

```
norm()              → used in Block.forward(), GPT.forward(), CausalSelfAttention.forward()
Linear              → used for c_q, c_k, c_v, c_proj, c_fc, lm_head, smear_gate, ve_gate
has_ve()            → used in CausalSelfAttention.__init__(), GPT.__init__()
apply_rotary_emb()  → used in CausalSelfAttention.forward() on q and k
```

Everything downstream depends on these four. Get them right and the rest of the model slots in cleanly.

---

> [!copywork] Copywork checkpoint
> Close this note. From memory, write all four pieces:
>
> 1. `norm(x)` — one line, uses `F.rms_norm`
> 2. `class Linear(nn.Linear)` — three lines, casts weight to input dtype
> 3. `has_ve(layer_idx, n_layer)` — one line, modulo arithmetic
> 4. `apply_rotary_emb(x, cos, sin)` — six lines, split/rotate/reassemble
>
> **Check yourself:**
> - Did you remember the trailing comma in `(x.size(-1),)`?
> - Did you write `F.linear` (lowercase) not `F.Linear`?
> - Did you split on `x.shape[3] // 2` (last dim, integer divide)?
> - Did you concatenate on `dim=3` (not `dim=-1` — though they're equivalent here)?
> - Did you write `-sin` in `y2`, not `sin`?

---

*Previous: [[01 - Config and Imports]]*
*Next: [[03 - CausalSelfAttention]] — Q/K/V projections, RoPE, flash attention*

*Back to [[Index]]*
