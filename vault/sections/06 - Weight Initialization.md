---
aliases: [init_weights, initialization, weight init]
tags: [section, phase-6]
source: nanochat/gpt.py lines 201–267
---

# 06 — Weight Initialization

<span class="phase-tag">SECTION 6</span> *How every parameter starts — the values before any training happens*

> **Source:** `nanochat/gpt.py` lines 201–267
> **Copywork target:** ~67 lines

---

## What this code does

`init_weights()` fills every parameter in the model with carefully chosen starting values. Bad initialization can make training fail entirely — gradients explode or vanish, attention collapses, loss doesn't decrease. This function is the cure.

It runs **once**, after `__init__` creates the structure on the meta device and the model is materialized on the real device.

---

## The initialization table (from the docstring)

```
COMPONENT          METHOD           VALUES              WHY
────────────────────────────────────────────────────────────────
wte (embedding)    normal           mean=0, std=0.8     wide spread — vocab is large
lm_head            normal           mean=0, std=0.001   tiny — logits start near zero (uniform predictions)
attn.c_q           uniform          [-s, s]             s = √3 / √768 ≈ 0.0625
attn.c_k           uniform          [-s, s]             same
attn.c_v           uniform          [-s, s]             same
attn.c_proj        zeros            all 0               residual starts as identity
mlp.c_fc           uniform          [-0.4s, 0.4s]       smaller — MLP starts quieter
mlp.c_proj         zeros            all 0               residual starts as identity
resid_lambdas      decaying         1.15 → 1.05         stronger early, weaker deep
x0_lambdas         decaying         0.20 → 0.05         more input blend early
smear_lambda       zeros            0                   smear disabled initially
backout_lambda     constant         0.2                 mild subtraction
smear_gate         uniform          [0, 0.02]           small positive
value_embeds       uniform          [-s, s]             same as c_v
ve_gate            uniform          [0, 0.02]           small positive (slightly above neutral)
```

---

## The code — annotated

```python
@torch.no_grad()
def init_weights(self):
```

`@torch.no_grad()` — no gradient tracking during initialization. We're just filling in values, not training.

### Embeddings and output head

```python
    torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
    torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
```

- `wte` std=0.8: embeddings start with moderate spread. Each of the 32,768 token vectors gets a different random direction in 768-dim space.
- `lm_head` std=0.001: nearly zero. This means early predictions are almost uniform across the vocab — the model doesn't start with strong biases toward any token.

### Transformer block weights

```python
    n_embd = self.config.n_embd
    s = 3**0.5 * n_embd**-0.5  # √3 × 1/√768 ≈ 0.0625

    for block in self.transformer.h:
        torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
        torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
        torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
        torch.nn.init.zeros_(block.attn.c_proj.weight)
        torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
        torch.nn.init.zeros_(block.mlp.c_proj.weight)
```

> [!keyinsight] Why `√3 × 1/√n_embd` for uniform init?
> A uniform distribution on `[-a, a]` has standard deviation `a / √3`.
> If we want std = `1/√768`, we need `a = √3 / √768`.
> This gives the uniform distribution the **same standard deviation** as `Normal(0, 1/√768)`. Uniform is preferred because it avoids outliers — no values far from the mean.

> [!keyinsight] Why are `c_proj` weights initialized to zeros?
> Both `attn.c_proj` and `mlp.c_proj` are the output projections inside residual connections: `x = x + sublayer(x)`. If the projection starts at zero, `sublayer(x)` returns zero, and `x = x + 0 = x`. The residual starts as a **pure identity** — each layer initially passes input through unchanged, then gradually learns to contribute.

### Per-layer scalar initialization

```python
    n_layer = self.config.n_layer
    for i in range(n_layer):
        self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
    for i in range(n_layer):
        self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))
```

Numeric example with `n_layer=12`:

```
resid_lambdas:
  Layer 0:  1.15 - (0.10 × 0/11)  = 1.150
  Layer 1:  1.15 - (0.10 × 1/11)  = 1.141
  Layer 6:  1.15 - (0.10 × 6/11)  = 1.095
  Layer 11: 1.15 - (0.10 × 11/11) = 1.050

x0_lambdas:
  Layer 0:  0.20 - (0.15 × 0/11)  = 0.200
  Layer 1:  0.20 - (0.15 × 1/11)  = 0.186
  Layer 6:  0.20 - (0.15 × 6/11)  = 0.118
  Layer 11: 0.20 - (0.15 × 11/11) = 0.050
```

Early layers get stronger residual scaling and more input embedding blending. Deep layers are more independent.

### Smear, backout, and gates

```python
    torch.nn.init.zeros_(self.smear_lambda)
    torch.nn.init.constant_(self.backout_lambda, 0.2)
    torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
```

Smear starts **disabled** (lambda=0). Backout starts at 0.2 (mild). Both can be learned during training.

### Value embeddings and their gates

```python
    for ve in self.value_embeds.values():
        torch.nn.init.uniform_(ve.weight, -s, s)

    for block in self.transformer.h:
        if block.attn.ve_gate is not None:
            torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
```

Value embeddings init like `c_v` (same scale). Gates start small positive — slightly above neutral so VE has a mild effect from the start.

### Rotary recompute and dtype casting

```python
    head_dim = self.config.n_embd // self.config.n_head
    cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
    self.cos, self.sin = cos, sin

    if COMPUTE_DTYPE != torch.float16:
        self.transformer.wte.to(dtype=COMPUTE_DTYPE)
        for ve in self.value_embeds.values():
            ve.to(dtype=COMPUTE_DTYPE)
```

Rotary embeddings are recomputed on the real device (they were meta placeholders before). Embeddings are cast to bf16 to save memory — the optimizer tolerates reduced precision for lookups.

---

> [!copywork] Copywork checkpoint
> Close this note. From memory, write `init_weights()` (~67 lines):
>
> 1. `@torch.no_grad()` decorator
> 2. wte: `normal_(mean=0, std=0.8)`, lm_head: `normal_(mean=0, std=0.001)`
> 3. `s = 3**0.5 * n_embd**-0.5` — the uniform bound
> 4. Loop over blocks: Q/K/V `uniform_(-s, s)`, c_proj `zeros_`, c_fc `uniform_(-s*0.4, s*0.4)`, mlp.c_proj `zeros_`
> 5. resid_lambdas: `1.15 - (0.10 * i / max(n_layer-1, 1))`
> 6. x0_lambdas: `0.20 - (0.15 * i / max(n_layer-1, 1))`
> 7. smear_lambda `zeros_`, backout_lambda `constant_(0.2)`, smear_gate `uniform_(0, 0.02)`
> 8. Value embeds `uniform_(-s, s)`, ve_gates `uniform_(0, 0.02)`
> 9. Recompute rotary, cast embeddings to COMPUTE_DTYPE (if not fp16)
>
> **Common traps:**
> - Did you use `.data[i]` for scalar assignment (not `nn.init`)?
> - Did you write `max(n_layer - 1, 1)` to avoid division by zero?
> - Did you use `0.4` multiplier only on `c_fc`, not on c_q/c_k/c_v?
> - Did you check `COMPUTE_DTYPE != torch.float16` (not `!= torch.float32`)?

---

*Previous: [[05 - GPT Init]]*
*Next: [[07 - Rotary Embeddings Deep Dive]]*

*Back to [[Index]]*
