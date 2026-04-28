---
aliases: [GPT, init, wte, lm_head, scalars, smear, backout, value embeds]
tags: [section, phase-5]
source: nanochat/gpt.py lines 154–199
---

# 05 — GPT Init

<span class="phase-tag">SECTION 5</span> *Wiring everything together — the top-level model constructor*

> **Source:** `nanochat/gpt.py` lines 154–199
> **Copywork target:** ~46 lines

---

## What this code does

`GPT.__init__()` is where all the pieces from Sections 01–04 get assembled into a complete model. It creates the embedding table, stacks 12 blocks, sets up the output head, and adds several modern tricks (per-layer scalars, smear gate, backout, value embeddings, rotary buffers).

> [!keyinsight] The meta device footgun
> The docstring warns: this `__init__` runs on a **meta device** — no actual data, just shapes and dtypes. All real parameter initialization happens later in `init_weights()` (Section 06). The values you see here (`torch.ones`, `torch.zeros`, `0.2 * torch.ones`) are **fake placeholders**.

---

## The code — annotated in logical chunks

### Chunk 1: Setup and vocab padding

```python
class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)

        # Pad vocab to nearest multiple of 64 for GPU efficiency
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1)
                             // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size}")
```

With `vocab_size=32768` and `pad_vocab_size_to=64`: `32768` is already divisible by 64, so no padding. But if vocab were 32700, it'd pad to 32704.

> [!keyinsight] Why pad the vocab?
> GPU tensor cores operate on tiles of 8, 16, 32, 64. A vocab size that's a multiple of 64 means the final matrix multiply (`lm_head`) uses every compute unit with zero waste. The padding rows in wte and lm_head are never used — they get cropped in `forward()`.

### Chunk 2: The transformer dict and output head

```python
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx)
                                for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
```

> [!shape] Shape trace — model skeleton
> ```
> COMPONENT              SHAPE                    PARAMETERS
> ────────────────────────────────────────────────────────────────
> wte                    (32768, 768)             25.2M    token embedding table
> h[0] through h[11]     12 × Block               —       each Block has attn + mlp
> lm_head                (768, 32768)             25.2M    output projection
> ```

> [!versus] nanoGPT vs nanochat — model structure
>
> | nanoGPT (your notes) | nanochat |
> |---------------------|----------|
> | `wte`, `wpe`, `drop`, `h`, `ln_f` in ModuleDict | Only `wte` and `h` — no wpe, no drop, no ln_f |
> | `self.lm_head.weight = self.transformer.wte.weight` (tied) | Separate parameters (untied) |
> | `nn.LayerNorm` as `ln_f` (final norm) | `norm(x)` called in `forward()` — not a module |

### Chunk 3: Per-layer learnable scalars

```python
        # resid_lambdas: scales the residual stream at each layer
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))     # (12,)
        # x0_lambdas: blends initial embedding back in at each layer
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))       # (12,)
```

These modify the residual flow in `forward()`:
```python
x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
```

- `resid_lambdas[i]` — scales the accumulated residual at layer i. Starts ~1.15 at layer 0, decays to ~1.05 at layer 11. Controls how much "memory" the residual carries.
- `x0_lambdas[i]` — blends the **original embedding** back in. Starts ~0.20 at layer 0, decays to ~0.05 at layer 11. A shortcut path from the raw input to deep layers.

> [!nanochat] Why per-layer scalars?
> Inspired by modded-nanogpt. Different layers have different roles — early layers handle local patterns, deep layers handle abstract reasoning. Giving each layer its own scaling lets the model tune the residual flow per-layer. Only 24 extra parameters (12 + 12), negligible cost.

### Chunk 4: Smear and backout

```python
        # Smear: mix previous token's embedding into current token
        self.smear_gate = Linear(24, 1, bias=False)        # (24, 1) = 24 params
        self.smear_lambda = nn.Parameter(torch.zeros(1))   # scalar

        # Backout: subtract cached mid-layer residual
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))  # scalar
```

**Smear** gives each token cheap access to the previous token's embedding — like a bigram. The gate reads the first 24 channels of the input to decide how much to mix in. We'll see how it works in [[09 - Forward Pass]].

**Backout** subtracts the residual at the halfway point (layer 6) before the final norm. By layer 12, the residual has accumulated low-level features (character patterns, positional info) that aren't useful for next-token prediction. Subtracting the midpoint removes some of that noise.

### Chunk 5: Value embeddings

```python
        head_dim = config.n_embd // config.n_head                    # 128
        kv_dim = config.n_kv_head * head_dim                         # 6 * 128 = 768
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
```

> [!shape] Shape trace — value embeddings
> ```
> COMPONENT                  SHAPE               PARAMETERS
> ────────────────────────────────────────────────────────────────
> value_embeds["1"]          (32768, 768)         25.2M
> value_embeds["3"]          (32768, 768)         25.2M
> value_embeds["5"]          (32768, 768)         25.2M
> value_embeds["7"]          (32768, 768)         25.2M
> value_embeds["9"]          (32768, 768)         25.2M
> value_embeds["11"]         (32768, 768)         25.2M
>                                          TOTAL: 151.0M
> ```

Six tables (alternating layers, last always included). Each is as large as `wte`. This is a significant parameter investment — which is why they're alternating, not on every layer.

Note: keys are **strings** (`"1"`, `"3"`, ...) because `nn.ModuleDict` requires string keys.

### Chunk 6: Rotary embedding buffers

```python
        self.rotary_seq_len = config.sequence_len * 10    # 20480 — over-allocate
        head_dim = config.n_embd // config.n_head         # 128
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
```

> [!shape] Shape trace — rotary buffers
> ```
> cos:  (1, 20480, 1, 64)    precomputed cosines
> sin:  (1, 20480, 1, 64)    precomputed sines
>
> ↑ 10× over-allocation: sequence_len=2048 × 10 = 20480
> ↑ persistent=False: NOT saved to checkpoint (recomputed on load)
> ↑ head_dim // 2 = 64: rotation operates on pairs of dims
> ```

> [!pytorch] `register_buffer` vs `nn.Parameter`
> Both are tensors stored on the model. The difference:
> - `nn.Parameter` — **trainable**, optimizer updates it, saved to checkpoint
> - `register_buffer` — **not trainable**, moves with `.to(device)`, optionally saved
>
> `persistent=False` means these aren't even saved — they're cheap to recompute.

---

## Complete parameter inventory

```
COMPONENT                  PARAMS         NOTES
────────────────────────────────────────────────────────────────
wte                        25.2M          token embeddings
lm_head                    25.2M          output projection (untied)
12 × Block                 ~56.6M         attention + MLP weights
  per block:
    c_q (768, 768)         590K
    c_k (768, 768)         590K
    c_v (768, 768)         590K
    c_proj (768, 768)      590K
    c_fc (768, 3072)       2.4M
    mlp.c_proj (3072, 768) 2.4M
    ve_gate (12, 6)        72 (alternating)
value_embeds (6 tables)    151.0M         biggest chunk!
resid_lambdas              12             per-layer scalars
x0_lambdas                 12             per-layer scalars
smear_gate                 24             (24, 1)
smear_lambda               1              scalar
backout_lambda             1              scalar
─────────────────────────────────────
TOTAL                      ~258M
```

---

> [!copywork] Copywork checkpoint
> Close this note. From memory, write `GPT.__init__()` (~46 lines):
>
> 1. Store config, compute window_sizes
> 2. Vocab padding formula: `((V + pad - 1) // pad) * pad`
> 3. `self.transformer = nn.ModuleDict({"wte": ..., "h": ...})`
> 4. `self.lm_head = Linear(C, padded_V, bias=False)`
> 5. Two scalar Parameters: `resid_lambdas` (ones), `x0_lambdas` (zeros)
> 6. Smear: `smear_gate` Linear(24, 1), `smear_lambda` Parameter(zeros(1))
> 7. Backout: `backout_lambda` Parameter(0.2 * ones(1))
> 8. Value embeds: ModuleDict with string keys, `has_ve` filter
> 9. Rotary: `rotary_seq_len`, precompute, `register_buffer` with `persistent=False`
>
> **Common traps:**
> - Did you use `padded_vocab_size` (not `config.vocab_size`) for wte and lm_head?
> - Did you compute `kv_dim = config.n_kv_head * head_dim` for value_embeds?
> - Did you use `str(i)` as keys in the value_embeds dict?
> - Did you write `persistent=False` on register_buffer?
> - Did you multiply `config.sequence_len * 10` for rotary_seq_len?

---

*Previous: [[04 - MLP and Block]]*
*Next: [[06 - Weight Initialization]]*

*Back to [[Index]]*
