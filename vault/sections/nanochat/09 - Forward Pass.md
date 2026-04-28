---
aliases: [forward, forward pass, data flow, smear, backout]
tags: [section, phase-9]
source: nanochat/gpt.py lines 416–481
---

# 09 — Forward Pass

<span class="phase-tag">SECTION 9</span> *The complete data flow — from token IDs to loss*

> **Source:** `nanochat/gpt.py` lines 416–481
> **Copywork target:** ~66 lines (includes smear gate logic)

---

## What this code does

`GPT.forward()` is the path every token takes through the model. Token IDs enter, a loss scalar (or logits) exits. This is where every piece from Sections 01–08 gets called in sequence.

---

## The code — annotated in logical chunks

### Chunk 1: Setup and rotary slicing

```python
def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
    B, T = idx.size()

    # Slice rotary embeddings to current sequence length
    assert T <= self.cos.size(1)
    assert idx.device == self.cos.device
    assert self.cos.dtype == COMPUTE_DTYPE
    T0 = 0 if kv_cache is None else kv_cache.get_pos()
    cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
```

> [!shape] Shape trace — rotary slicing
> ```
> self.cos                (1, 20480, 1, 64)    full precomputed cache
> cos_sin[0]              (1, T, 1, 64)        sliced to current sequence length
>
> During inference with KV cache:
>   T0 = current position in cache (e.g., 100)
>   cos[:, 100:101]       (1, 1, 1, 64)        single new token's rotation
> ```

### Chunk 2: Embed and normalize

```python
    x = self.transformer.wte(idx)      # (B, T) → (B, T, 768)
    x = x.to(COMPUTE_DTYPE)
    x = norm(x)
```

> [!shape] Shape trace — embedding
> ```
> idx                     (B, T)               int64 token IDs
> wte(idx)                (B, T, 768)          float — each int → 768 floats
> .to(COMPUTE_DTYPE)      (B, T, 768)          ensure bf16 (usually no-op)
> norm(x)                 (B, T, 768)          RMSNorm — consistent scale
> ```

> [!nanochat] Norm right after embedding — new in nanochat
> nanoGPT didn't normalize after embedding. nanochat does — it stabilizes the values before any further processing. The norm is applied BEFORE the smear gate.

### Chunk 3: Smear gate — cheap bigram info

```python
    # Training path (full sequence)
    if kv_cache is None:
        assert T > 1
        gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
            self.smear_gate(x[:, 1:, :24]))
        x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
    else:
        # Inference paths (KV cache)
        x_pre_smear = kv_cache.prev_embedding
        kv_cache.prev_embedding = x[:, -1:, :]
        if T > 1:
            # Prefill
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
                self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        elif x_pre_smear is not None:
            # Decode: single token
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
                self.smear_gate(x[:, :, :24]))
            x = x + gate * x_pre_smear
```

> [!shape] Shape trace — smear gate (training path)
> ```
> x                       (B, T, 768)          current embeddings
> x[:, 1:, :24]           (B, T-1, 24)         first 24 channels of positions 1+
> self.smear_gate(...)     (B, T-1, 1)          gate value per position
> torch.sigmoid(...)       (B, T-1, 1)          squashed to (0, 1)
> smear_lambda * ...       (B, T-1, 1)          scaled (starts at 0 → disabled)
>
> x[:, :-1]               (B, T-1, 768)        previous token's embedding
> gate * x[:, :-1]         (B, T-1, 768)        gated previous embedding
> x[:, 1:] + ...           (B, T-1, 768)        mixed into current
>
> torch.cat([x[:,:1], ...], dim=1)   (B, T, 768)  position 0 unchanged, rest smeared
> ```

> [!keyinsight] What smear actually does
> Each token (except position 0) gets a small amount of the previous token's embedding mixed in. This is free bigram information — "what token came right before me?" — without needing an attention computation.
>
> Position 0 has no previous token, so it's left unchanged (the `torch.cat` with `x[:, :1]`).
>
> `smear_lambda` starts at 0 (disabled) and is learned during training. The gate reads only the first 24 channels — it's deliberately cheap.

### Chunk 4: The main transformer loop

```python
    x0 = x  # save initial embedding for x0_lambdas blending
    n_layer = self.config.n_layer
    backout_layer = n_layer // 2  # 6 — cache at halfway point
    x_backout = None

    for i, block in enumerate(self.transformer.h):
        x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        ve = self.value_embeds[str(i)](idx).to(x.dtype) \
             if str(i) in self.value_embeds else None
        x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        if i == backout_layer:
            x_backout = x
```

> [!shape] Shape trace — transformer loop
> ```
> x0                      (B, T, 768)          saved initial embedding
>
> EACH ITERATION (i = 0 to 11):
> resid_lambdas[i] * x    (B, T, 768)          scale residual stream
> + x0_lambdas[i] * x0    (B, T, 768)          blend in original embedding
>
> value_embeds[str(i)](idx)(B, T, 768)          VE lookup (alternating layers)
>                          or None               (layers without VE)
>
> block(x, ve, ...)        (B, T, 768)          attention + MLP
>
> At i=6: x_backout = x   (B, T, 768)          cached for backout
>
> Shape (B, T, 768) preserved through all 12 iterations
> ```

> [!keyinsight] The three things happening before each block
> 1. `resid_lambdas[i] * x` — scale the residual (1.15 → 1.05 across layers)
> 2. `+ x0_lambdas[i] * x0` — blend in the original embedding (0.20 → 0.05)
> 3. `value_embeds[str(i)](idx)` — look up value embedding if this layer has one
>
> These three operations happen BEFORE `block()`, not inside it.

### Chunk 5: Backout, final norm, and output

```python
    # Subtract mid-layer residual
    if x_backout is not None:
        x = x - self.backout_lambda.to(x.dtype) * x_backout
    x = norm(x)

    # Logits with softcap
    softcap = 15
    logits = self.lm_head(x)                          # (B, T, padded_V)
    logits = logits[..., :self.config.vocab_size]      # crop padding
    logits = logits.float()                            # fp32 for stability
    logits = softcap * torch.tanh(logits / softcap)    # squash to [-15, 15]

    if targets is not None:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction)
        return loss
    else:
        return logits
```

> [!shape] Shape trace — output pipeline
> ```
> x (after loop)           (B, T, 768)          12 blocks of processing done
> - backout_lambda * x_backout  (B, T, 768)     subtract mid-layer residual
> norm(x)                  (B, T, 768)          final RMSNorm
>
> lm_head(x)               (B, T, 32768)        project to vocab (padded)
> [..., :vocab_size]        (B, T, 32768)        crop padding (no-op if already aligned)
> .float()                  (B, T, 32768)        cast to fp32
> softcap * tanh(x/softcap) (B, T, 32768)       squash to [-15, 15]
>
> TRAINING (targets given):
> logits.view(-1, V)       (B*T, 32768)         flatten for cross_entropy
> targets.view(-1)         (B*T,)               flatten targets
> F.cross_entropy(...)     scalar               the loss
>
> INFERENCE (no targets):
> return logits            (B, T, 32768)        raw scores for sampling
> ```

> [!keyinsight] Logit softcap — preventing extreme predictions
> `15 * tanh(logits / 15)` smoothly squashes logits into the range `[-15, 15]`.
>
> Without softcap, a logit of 50 would make softmax assign >99.99% probability to one token — the model becomes overconfident. tanh squashing prevents this while preserving the ranking of predictions.
>
> Values near zero pass through almost unchanged: `15 * tanh(1/15) ≈ 1.0`. Only extreme values get compressed.

> [!versus] nanoGPT vs nanochat — forward pass
>
> | nanoGPT (your notes) | nanochat |
> |---------------------|----------|
> | `tok_emb + pos_emb` (wte + wpe) | `wte(idx)` then `norm(x)` — no wpe |
> | `self.transformer.drop(x)` | No dropout |
> | Simple `for block in h: x = block(x)` | `resid_lambdas * x + x0_lambdas * x0` then `block(...)` |
> | `self.transformer.ln_f(x)` (LayerNorm module) | `norm(x)` (function call) |
> | `self.lm_head(x)` → raw logits | `lm_head(x)` → crop → fp32 → softcap |
> | `logits.view(-1, V), targets.view(-1)` | Same + `ignore_index=-1`, `reduction=loss_reduction` |
> | No smear, no backout, no per-layer scalars | All three present |

---

## The complete shape trace — full forward pass

```
THE COMPLETE FORWARD PASS — TOKEN IDS TO LOSS

INPUT
idx                        (B, T)               int64 token IDs

EMBEDDING
wte(idx)                   (B, T, 768)          token embedding lookup
.to(COMPUTE_DTYPE)         (B, T, 768)          ensure bf16
norm(x)                    (B, T, 768)          post-embedding RMSNorm

SMEAR GATE
smear_gate(x[:, 1:, :24])  (B, T-1, 1)         gate from first 24 channels
x = cat([x[:,:1], x[:,1:] + gate * x[:,:-1]])   (B, T, 768)   bigram mixing

SAVE x0                    (B, T, 768)          for x0_lambdas blending

TRANSFORMER LOOP (12 iterations)
  resid_lambdas[i] * x     (B, T, 768)          scale residual
  + x0_lambdas[i] * x0     (B, T, 768)          blend original embedding
  value_embeds[i](idx)      (B, T, 768) or None  VE lookup (alternating)
  block(x, ve, cos_sin, window, cache)  (B, T, 768)  attention + MLP
  [at i=6: save x_backout]

BACKOUT
x - backout_lambda * x_backout  (B, T, 768)     remove low-level features

FINAL NORM
norm(x)                    (B, T, 768)          RMSNorm

OUTPUT
lm_head(x)                 (B, T, 32768)        vocab projection
[..., :vocab_size]          (B, T, 32768)        crop padding
.float()                    (B, T, 32768)        fp32 for softcap
softcap * tanh(x/softcap)  (B, T, 32768)        squash to [-15, 15]

LOSS (training)
.view(-1, V)                (B*T, 32768)         flatten
cross_entropy               scalar               the loss ✓
```

---

> [!copywork] Copywork checkpoint
> This is the longest section. Write it in three passes:
>
> **Pass 1 — setup + embed + smear (lines 416–449):**
> - `B, T = idx.size()`, rotary assertions and slicing
> - `wte(idx)`, `.to()`, `norm(x)`
> - Smear gate: training path (`kv_cache is None`) and inference paths
>
> **Pass 2 — transformer loop (lines 451–461):**
> - Save `x0`, set `backout_layer = n_layer // 2`
> - Loop: `resid_lambdas * x + x0_lambdas * x0`, VE lookup, `block()`, backout cache
>
> **Pass 3 — output (lines 462–481):**
> - Backout subtraction, `norm(x)`
> - `lm_head(x)`, vocab crop, `.float()`, softcap
> - `cross_entropy` with `ignore_index=-1` and `reduction`, or return logits
>
> **Common traps:**
> - Did you write `T0 = 0 if kv_cache is None else kv_cache.get_pos()`?
> - Did you cat `x[:, :1]` (position 0 unchanged) in smear?
> - Did you write `str(i) in self.value_embeds` (string key check)?
> - Did you crop logits with `[..., :self.config.vocab_size]`?
> - Did you cast to `.float()` before softcap?
> - Did you return `loss` (training) vs `logits` (inference)?

---

*Previous: [[08 - Sliding Window Attention]]*
*Next: [[11 - Optimizer Setup]]*

*Back to [[Index]]*
