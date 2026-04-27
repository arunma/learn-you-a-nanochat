---
aliases: [RoPE, rotary, precompute, position encoding]
tags: [section, phase-7]
source: nanochat/gpt.py lines 268–283
---

# 07 — Rotary Embeddings Deep Dive

<span class="phase-tag">SECTION 7</span> *How `_precompute_rotary_embeddings()` generates the cos/sin tables that RoPE uses*

> **Source:** `nanochat/gpt.py` lines 268–283
> **Copywork target:** ~16 lines

---

## What this code does

In Section 02, you wrote `apply_rotary_emb()` which *uses* cos and sin. This function *creates* them. It builds a 2D grid of rotation angles — one axis is position in the sequence (0 to seq_len), the other is dimension pair (0 to head_dim/2). Each cell contains the angle by which that position/dimension pair should be rotated.

---

## The code

```python
def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
    if device is None:
        device = self.transformer.wte.weight.device

    # Frequencies for each dimension pair
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))

    # Position indices
    t = torch.arange(seq_len, dtype=torch.float32, device=device)

    # 2D grid: every (position, dimension) angle
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()

    cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
    # Add batch and head dims for broadcasting
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return cos, sin
```

> [!shape] Complete shape trace
> ```
> STEP                           SHAPE                 CONCRETE VALUES
> ──────────────────────────────────────────────────────────────────────
> channel_range                  (64,)                 [0, 2, 4, ..., 126]
> channel_range / head_dim       (64,)                 [0/128, 2/128, ..., 126/128]
> base ** (...)                   (64,)                 [100000^0, 100000^0.016, ..., 100000^0.984]
> inv_freq = 1.0 / ...           (64,)                 [1.0, 0.91, ..., 0.00001]
>
> t                              (20480,)              [0, 1, 2, ..., 20479]
>
> freqs = outer(t, inv_freq)     (20480, 64)           every (pos, dim) angle
> cos = freqs.cos()              (20480, 64)           cosines
> sin = freqs.sin()              (20480, 64)           sines
>
> cos[None, :, None, :]          (1, 20480, 1, 64)     ready for broadcasting
> sin[None, :, None, :]          (1, 20480, 1, 64)     ready for broadcasting
> ```

### Step-by-step with concrete numbers

**Channel frequencies — `inv_freq`:**

The 128 dimensions of each head are grouped into 64 pairs. Each pair rotates at a different speed:

```
Pair 0  (dims 0,64):   inv_freq = 1/100000^(0/128)   = 1.000    ← fastest rotation
Pair 1  (dims 1,65):   inv_freq = 1/100000^(2/128)   = 0.912
Pair 2  (dims 2,66):   inv_freq = 1/100000^(4/128)   = 0.832
...
Pair 31 (dims 31,95):  inv_freq = 1/100000^(62/128)  = 0.009
...
Pair 63 (dims 63,127): inv_freq = 1/100000^(126/128) = 0.00001  ← slowest rotation
```

Low-index pairs rotate fast (sensitive to nearby position differences). High-index pairs rotate slowly (sensitive to long-range position differences).

**The outer product — `freqs`:**

```
freqs[pos, pair] = pos × inv_freq[pair]

Position 0:  [0×1.0, 0×0.91, ..., 0×0.00001] = [0, 0, ..., 0]        all zero
Position 1:  [1×1.0, 1×0.91, ..., 1×0.00001] = [1.0, 0.91, ..., 0.00001]
Position 5:  [5×1.0, 5×0.91, ..., 5×0.00001] = [5.0, 4.55, ..., 0.00005]
Position 100: [100.0, 91.2, ..., 0.001]
```

Each row is the rotation angles for one position. `cos()` and `sin()` of these angles become the rotation matrices that `apply_rotary_emb()` uses.

**The unsqueeze — broadcasting shape:**

```
(20480, 64) → (1, 20480, 1, 64)
  ↑               ↑       ↑    ↑
  batch=1         T      head=1 d_h/2

Broadcasts across:
- dim 0 (batch): same rotation for every item in batch ✓
- dim 2 (heads): same rotation for every head ✓
```

### The `base` parameter — θ = 100,000

The base controls how quickly the rotation speeds decay across dimension pairs. Higher base = slower overall rotation = the model can distinguish positions further apart.

- Original Transformer (2017): base = 10,000
- nanochat: base = 100,000 (10× larger, better for longer sequences)

> [!keyinsight] How cos/sin connect back to `apply_rotary_emb()`
> This function produces `cos` and `sin` of shape `(1, T, 1, 64)`.
> `apply_rotary_emb()` receives Q of shape `(B, T, 6, 128)`, splits into two halves of 64.
> The cos/sin broadcast across batch (dim 0) and heads (dim 2).
> Each 64-dim half gets rotated element-wise. The rotation angles encode position.

---

> [!copywork] Copywork checkpoint
> Close this note. From memory, write `_precompute_rotary_embeddings()` (~16 lines):
>
> 1. Device auto-detection from wte
> 2. `channel_range = arange(0, head_dim, 2)` — step 2, only half the dims
> 3. `inv_freq = 1.0 / (base ** (channel_range / head_dim))`
> 4. `t = arange(seq_len)`
> 5. `freqs = torch.outer(t, inv_freq)`
> 6. `cos, sin = freqs.cos(), freqs.sin()`
> 7. Cast to COMPUTE_DTYPE
> 8. `[None, :, None, :]` — add batch and head dims
>
> **Common traps:**
> - Did you use `arange(0, head_dim, 2)` (step 2, not step 1)?
> - Did you write `base ** (channel_range / head_dim)` in the denominator?
> - Did you use `torch.outer` (not `@` or `matmul`)?
> - Did you add `dtype=torch.float32` to both arange calls?
> - Did you unsqueeze with `[None, :, None, :]` (4D, not 2D)?

---

*Previous: [[06 - Weight Initialization]]*
*Next: [[08 - Sliding Window Attention]]*

*Back to [[Index]]*
