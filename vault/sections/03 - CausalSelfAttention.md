---
aliases: [attention, self-attention, CausalSelfAttention, Q K V]
tags: [section, phase-3]
source: nanochat/gpt.py lines 65–127
---

# 03 — CausalSelfAttention

<span class="phase-tag">SECTION 3</span> *The heart of the transformer — how tokens attend to each other*

> **Source:** `nanochat/gpt.py` lines 65–127
> **Copywork target:** ~62 lines (the `__init__` and `forward` methods)

---

## What this code does

This is the single most important class in the model. Every other piece exists to support it.

`CausalSelfAttention` answers one question per token: **"Given everything I've seen so far, which other tokens should I pay attention to, and what should I gather from them?"**

It does this through:
1. **Project** — create Q, K, V from the input (three separate views)
2. **Position** — apply RoPE so tokens know where they sit
3. **Normalize** — QK norm to prevent score explosion
4. **Attend** — compute attention scores and aggregate values (Flash Attention)
5. **Recombine** — merge heads and project back to the residual stream

---

## Part 1: `__init__` — setting up the projections

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head                          # 6 query heads
        self.n_kv_head = config.n_kv_head                    # 6 KV heads (GQA)
        self.n_embd = config.n_embd                          # 768
        self.head_dim = self.n_embd // self.n_head           # 768 // 6 = 128
        assert self.n_embd % self.n_head == 0                # head_dim must be integer
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        # Separate Q, K, V projections (not fused like nanoGPT)
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)      # (768, 768)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)   # (768, 768)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)   # (768, 768)

        # Output projection — recombines all heads back to n_embd
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)                   # (768, 768)

        # Value embedding gate — only on layers that have VE (alternating)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) \
            if has_ve(layer_idx, config.n_layer) else None
```

### What each layer does

| Layer | Shape | Purpose |
|-------|-------|---------|
| `c_q` | `(768, 768)` | Projects input → queries. One set per Q head. |
| `c_k` | `(768, 768)` | Projects input → keys. One set per KV head. |
| `c_v` | `(768, 768)` | Projects input → values. One set per KV head. |
| `c_proj` | `(768, 768)` | Recombines all head outputs back to residual stream. |
| `ve_gate` | `(12, 6)` | Tiny gate controlling how much value embedding mixes in. Only on alternating layers. |

> [!versus] nanoGPT vs nanochat — Q/K/V projection
>
> | nanoGPT (your notes) | nanochat |
> |---------------------|----------|
> | Fused: `c_attn = Linear(C, 3*C)` | Separate: `c_q`, `c_k`, `c_v` |
> | Then `.split(C, dim=2)` → q, k, v | Each already the right size |
> | All heads must have same dim | Q and KV can have different head counts (GQA) |
> | One big matmul, then split | Three smaller matmuls |
> | `c_attn` weight: (768, 2304) | `c_q`: (768,768), `c_k`: (768,768), `c_v`: (768,768) |

> [!qkv] Why separate projections enable GQA
> In nanoGPT, fusing Q+K+V into one matrix forces them all to have the same output size (C each). But with GQA, you want `n_kv_head < n_head` — meaning K and V are *smaller* than Q. Separate projections make this trivial:
>
> ```
> With n_head=6, n_kv_head=2, head_dim=128:
>   c_q output: 6 × 128 = 768
>   c_k output: 2 × 128 = 256   ← smaller!
>   c_v output: 2 × 128 = 256   ← smaller!
> ```
>
> With the defaults (n_kv_head=6 = n_head=6), all three are the same size and it behaves like standard MHA.

### The two asserts — guardrails

```python
assert self.n_embd % self.n_head == 0
```
head_dim must be a whole number. 768 / 6 = 128. If you tried n_head=7, you'd get 768/7 = 109.7... which doesn't work.

```python
assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
```
KV heads must evenly divide Q heads. With n_head=6, valid n_kv_head values are: 1, 2, 3, 6. This ensures each KV head serves an equal number of Q heads.

### The ve_gate — controlling value embedding strength

```python
self.ve_gate_channels = 12
self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) \
    if has_ve(layer_idx, config.n_layer) else None
```

Only exists on alternating layers (where `has_ve()` returns True). It's tiny: (12, 6) = 72 parameters. It reads the first 12 channels of the input to decide how much of the value embedding to mix in — **per head**. Layers without VE just get `None` here.

---

## Part 2: `forward` — the complete attention computation

```python
def forward(self, x, ve, cos_sin, window_size, kv_cache):
    B, T, C = x.size()
```

Five arguments:
- `x` — the input tensor `(B, T, 768)`, already norm'd by the caller
- `ve` — value embedding for this layer's token IDs, or `None`
- `cos_sin` — precomputed rotary embeddings `(cos, sin)` tuple
- `window_size` — `(left, right)` tuple for sliding window attention
- `kv_cache` — KV cache for inference, or `None` during training

### Step 1: Project to Q, K, V and reshape into heads

```python
q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
```

> [!shape] Shape trace — Q/K/V projection
> ```
> STEP 1: PROJECT AND RESHAPE
>
> x                      (B, T, 768)              input (already normed)
>
> self.c_q(x)            (B, T, 768)              Q projection: 768 → 768
> .view(B,T,6,128)       (B, T, 6, 128)           reshape: split into 6 heads of 128
>
> self.c_k(x)            (B, T, 768)              K projection: 768 → 768
> .view(B,T,6,128)       (B, T, 6, 128)           reshape: split into 6 KV heads of 128
>
> self.c_v(x)            (B, T, 768)              V projection: 768 → 768
> .view(B,T,6,128)       (B, T, 6, 128)           reshape: split into 6 KV heads of 128
> ```

> [!versus] nanoGPT vs nanochat — head layout
>
> | nanoGPT (your notes) | nanochat |
> |---------------------|----------|
> | `.view(B, T, n_head, d_h)` then `.transpose(1, 2)` | `.view(B, T, n_head, d_h)` — no transpose! |
> | Final shape: `(B, n_head, T, d_h)` | Final shape: `(B, T, n_head, d_h)` |
> | Heads in dim 1 for manual `q @ k.T` matmul | Heads in dim 2 — Flash Attention's native layout |
> | You needed transpose because `@` operates on last 2 dims | FA3 expects `(B, T, H, D)` directly |

> [!keyinsight] No transpose in nanochat!
> Your notes had `.transpose(1, 2)` as a mandatory step after `.view()`. nanochat skips it entirely. Flash Attention expects `(B, T, H, D)` — heads in dim 2, not dim 1. This avoids one memory operation per Q, K, and V.

### Step 2: Mix in value embeddings (if this layer has them)

```python
if ve is not None:
    ve = ve.view(B, T, self.n_kv_head, self.head_dim)
    gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    v = v + gate.unsqueeze(-1) * ve
```

> [!shape] Shape trace — value embedding mixing
> ```
> STEP 2: VALUE EMBEDDING (only on alternating layers)
>
> ve (input)             (B, T, 768)              raw value embedding lookup
> .view(B,T,6,128)       (B, T, 6, 128)           reshaped to match v
>
> x[..., :12]            (B, T, 12)               first 12 channels of input
> self.ve_gate(...)      (B, T, 6)                gate per KV head (12 → 6)
> torch.sigmoid(...)     (B, T, 6)                squash to (0, 1)
> 3 * ...                (B, T, 6)                scale to (0, 3) range
> .unsqueeze(-1)         (B, T, 6, 1)             add dim for broadcasting
>
> gate * ve              (B, T, 6, 128)           gated value embedding
> v + ...                (B, T, 6, 128)           mixed into V
> ```

The gate reads only the **first 12 channels** of x. It's input-dependent — different tokens at different positions get different gate values. The range (0, 3) means the value embedding can be scaled up beyond the standard V, not just blended in.

### Step 3: Apply RoPE and QK norm

```python
cos, sin = cos_sin
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
q, k = norm(q), norm(k)
q = q * 1.2
k = k * 1.2
```

> [!shape] Shape trace — RoPE and normalization
> ```
> STEP 3: POSITIONAL ENCODING + NORMALIZATION
>
> cos                    (1, T, 1, 64)            precomputed (broadcasts)
> sin                    (1, T, 1, 64)            precomputed (broadcasts)
>
> apply_rotary_emb(q)    (B, T, 6, 128)           Q rotated by position
> apply_rotary_emb(k)    (B, T, 6, 128)           K rotated by position
>
> norm(q)                (B, T, 6, 128)           Q normalized (RMSNorm per head)
> norm(k)                (B, T, 6, 128)           K normalized (RMSNorm per head)
>
> q * 1.2                (B, T, 6, 128)           sharpen attention scores
> k * 1.2                (B, T, 6, 128)           sharpen attention scores
> ```

> [!keyinsight] Why QK norm + the 1.2 scale?
> After RoPE, Q and K vectors can have unpredictable magnitudes. The dot product `Q · K` is proportional to the magnitudes of both vectors — large magnitudes → large scores → softmax collapses to a spike on one token.
>
> QK norm forces both Q and K to unit scale. Then `1.2 × 1.2 = 1.44` effectively replaces the `1/√d_k` scaling from the textbook attention formula. It's a sharper attention pattern — the model is more decisive about which tokens to attend to.
>
> In your notes, the formula was `Q @ K.T / √64`. Here it's `norm(Q) * 1.2 @ norm(K.T) * 1.2`. Different path, same goal: controlled score magnitudes.

### Step 4: Flash Attention

```python
if kv_cache is None:
    # Training: causal attention with optional sliding window
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
else:
    # Inference: use flash_attn_with_kvcache
    k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
    y = flash_attn.flash_attn_with_kvcache(
        q, k_cache, v_cache,
        k=k, v=v,
        cache_seqlens=kv_cache.cache_seqlens,
        causal=True,
        window_size=window_size,
    )
    if self.layer_idx == kv_cache.n_layers - 1:
        kv_cache.advance(T)
```

> [!shape] Shape trace — attention computation
> ```
> STEP 4: FLASH ATTENTION (one fused kernel)
>
> q                      (B, T, 6, 128)           queries
> k                      (B, T, 6, 128)           keys
> v                      (B, T, 6, 128)           values
> causal=True            ← tokens can only attend to past positions
> window_size            ← (512, 0) for short, (2048, 0) for long
>
> INTERNALLY (you never see these tensors — they're fused):
>   scores = q @ k.T     (B, 6, T, T)             ← 25M scores, never materialized!
>   scores = masked       ← causal mask + window
>   weights = softmax     ← probabilities per row
>   y = weights @ v       ← weighted value sum
>
> y (output)             (B, T, 6, 128)           attended values per head
> ```

> [!versus] nanoGPT vs nanochat — attention computation
>
> | nanoGPT (your notes) | nanochat |
> |---------------------|----------|
> | `att = q @ k.transpose(-2, -1)` | One call: `flash_attn.flash_attn_func(q, k, v)` |
> | `att = att * (1.0 / math.sqrt(d))` | Scaling handled by QK norm + 1.2 |
> | `att = att.masked_fill(bias == 0, float('-inf'))` | `causal=True` parameter |
> | `att = F.softmax(att, dim=-1)` | Fused inside the kernel |
> | `att = self.attn_dropout(att)` | No dropout in nanochat |
> | `y = att @ v` | Fused inside the kernel |
> | Materializes full `(B, n_head, T, T)` matrix | **Never materializes the score matrix** |
> | Memory: O(T²) | Memory: O(T) — the key speedup |

> [!keyinsight] Flash Attention's key trick
> Your notes showed the `(B, 6, T, T)` attention score matrix — 25 million numbers that get computed, masked, softmaxed, then multiplied by V. Flash Attention does all of this **without ever storing that matrix**. It processes the attention computation in tiles, streaming through GPU SRAM. This is why nanochat can handle T=2048 without running out of memory.
>
> From your perspective as a copywork learner: you call one function and it handles everything. The internal math is identical to the 5-step manual process in your notes — it's just fused.

### Step 5: Reassemble heads and project

```python
y = y.contiguous().view(B, T, -1)
y = self.c_proj(y)
return y
```

> [!shape] Shape trace — head reassembly
> ```
> STEP 5: REASSEMBLE HEADS
>
> y                      (B, T, 6, 128)           output from flash attention
> .contiguous()          (B, T, 6, 128)           ensure memory is contiguous
> .view(B, T, -1)        (B, T, 768)              6 × 128 = 768, heads merged
> self.c_proj(y)         (B, T, 768)              project back to residual stream
> return                 (B, T, 768)              output — same shape as input
> ```

> [!versus] nanoGPT vs nanochat — head reassembly
>
> | nanoGPT (your notes) | nanochat |
> |---------------------|----------|
> | `.transpose(1, 2)` to move heads back | No transpose needed (heads already in dim 2) |
> | `.contiguous().view(B, T, C)` | `.contiguous().view(B, T, -1)` |
> | `self.c_proj(y)` | Same |
> | `self.resid_dropout(y)` | No dropout |

---

## The complete shape trace — input to output

```
THE COMPLETE MULTI-HEAD ATTENTION — nanochat shape flow

INPUT
x                          (B, T, 768)              entering CausalSelfAttention

STEP 1 — Q, K, V PROJECTIONS
c_q(x)                     (B, T, 768)              query projection
.view(B, T, 6, 128)        (B, T, 6, 128)           heads labelled — 3D → 4D
c_k(x)                     (B, T, 768)              key projection
.view(B, T, 6, 128)        (B, T, 6, 128)           KV heads labelled
c_v(x)                     (B, T, 768)              value projection
.view(B, T, 6, 128)        (B, T, 6, 128)           KV heads labelled

STEP 2 — VALUE EMBEDDING (alternating layers only)
ve.view(B, T, 6, 128)      (B, T, 6, 128)           static token lookup reshaped
gate                        (B, T, 6, 1)             per-head gate from first 12 channels
v = v + gate * ve           (B, T, 6, 128)           V enriched with token identity

STEP 3 — ROPE + QK NORM
apply_rotary_emb(q)         (B, T, 6, 128)           Q rotated by position
apply_rotary_emb(k)         (B, T, 6, 128)           K rotated by position
norm(q), norm(k)            (B, T, 6, 128)           unit scale per head
q * 1.2, k * 1.2           (B, T, 6, 128)           sharpened attention

STEP 4 — FLASH ATTENTION (fused kernel, scores never materialized)
flash_attn(q, k, v)         (B, T, 6, 128)           attended output per head

STEP 5 — REASSEMBLE HEADS
.contiguous().view(B,T,-1)  (B, T, 768)              6 × 128 = 768 — heads merged
c_proj(y)                   (B, T, 768)              projected to residual stream

OUTPUT                      (B, T, 768)              same shape as input ✓
```

---

## For comparison: your nanoGPT notes' shape trace

```
THE COMPLETE MULTI-HEAD ATTENTION — nanoGPT shape flow (your earlier notes)

INPUT
x                          (B, T, 384)              entering CausalSelfAttention

PHASE 3.1 — Q, K, V PROJECTIONS
c_attn(x)                  (B, T, 1152)             fused Q+K+V — C triples
.split(384, dim=2)          q, k, v: (B, T, 384)    separated into three tensors
.view(B, T, 6, 64)         (B, T, 6, 64)            heads labelled — 3D → 4D
.transpose(1, 2)            (B, 6, T, 64)            heads → dim 1 for batched matmul

PHASE 3.2 — SCALED DOT-PRODUCT CAUSAL ATTENTION
q @ k.T(-2, -1)            (B, 6, T, T)             raw scores — 64 dims cancel
× 1/√64                    (B, 6, T, T)             scaled — prevents softmax collapse
masked_fill(-∞)             (B, 6, T, T)             future positions zeroed
softmax(dim=-1)             (B, 6, T, T)             weights sum to 1.0 per row
att @ v                     (B, 6, T, 64)            weighted value sum

PHASE 3.3 — REASSEMBLE HEADS
.transpose(1, 2)            (B, T, 6, 64)            heads back to dim 2
.view(B, T, 384)            (B, T, 384)              6 × 64 concatenated
c_proj                      (B, T, 384)              heads mixed — synthesis complete ✓
```

### Key differences at a glance

| Aspect | nanoGPT | nanochat |
|--------|---------|----------|
| Head dim | 64 | 128 |
| n_embd | 384 | 768 |
| Q/K/V projection | Fused `c_attn` → split | Separate `c_q`, `c_k`, `c_v` |
| Head layout | `(B, n_head, T, d_h)` — transpose needed | `(B, T, n_head, d_h)` — no transpose |
| Positional encoding | Already added to x via wpe | RoPE applied to Q, K here |
| Score scaling | `/ √d_k` | QK norm + `* 1.2` |
| Attention compute | Manual 5-step (matmul, mask, softmax, matmul) | One `flash_attn` call |
| Score matrix | Materialized `(B, 6, T, T)` | Never materialized |
| Dropout | `attn_dropout` + `resid_dropout` | None |
| Value embeddings | None | Gated VE on alternating layers |

---

> [!copywork] Copywork checkpoint
> Close this note. From memory, write the full `CausalSelfAttention` class (~62 lines):
>
> **`__init__`:**
> 1. Store config values: `layer_idx`, `n_head`, `n_kv_head`, `n_embd`, `head_dim`
> 2. Two asserts (divisibility)
> 3. Four Linear layers: `c_q`, `c_k`, `c_v`, `c_proj`
> 4. `ve_gate_channels = 12` and conditional `ve_gate`
>
> **`forward`:**
> 1. Project and `.view()` into heads — no transpose
> 2. Value embedding gate (if `ve is not None`)
> 3. Unpack cos_sin, apply_rotary_emb to q and k
> 4. norm(q), norm(k), multiply by 1.2
> 5. Flash attention (training vs inference branch)
> 6. `.contiguous().view(B, T, -1)` → `c_proj` → return
>
> **Common traps:**
> - Did you use `self.n_kv_head` (not `self.n_head`) for K and V dimensions?
> - Did you skip `.transpose(1, 2)`? (nanochat doesn't need it)
> - Did you write `.view(B, T, -1)` not `.view(B, T, C)`? (The `-1` auto-computes)
> - Did you multiply gate by `3 *` before sigmoid? (Actually: `3 * torch.sigmoid(...)`)
> - Did you remember `.unsqueeze(-1)` on the gate before multiplying with ve?
> - Did you write `q * 1.2` and `k * 1.2` AFTER norm, not before?

---

*Previous: [[02 - Building Blocks]]*
*Next: [[04 - MLP and Block]] — relu², the feed-forward network, and the two-half block*

*Back to [[Index]]*
