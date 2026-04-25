---
aliases: [comparison, versus, differences]
tags: [reference]
---

# nanoGPT vs nanochat

> Your earlier tutorial notes were based on nanoGPT — a simpler GPT-2 implementation.
> nanochat is the successor. Same author (Karpathy), significantly upgraded architecture.
> This page maps every difference so you can connect what you already know to what's new.

## Side-by-side comparison

| Component | nanoGPT (your notes) | nanochat (this repo) | Why the change |
|-----------|---------------------|---------------------|----------------|
| **Positional encoding** | Learned `wpe = nn.Embedding(T, C)` | Rotary embeddings (RoPE) — no learned params | Relative positions generalize better; no parameter cost |
| **Normalization** | `nn.LayerNorm(C)` with learned `gamma`, `beta` | `F.rms_norm()` — no learned params | Simpler, one fewer parameter set per norm, works equally well |
| **Norm placement** | Pre-norm (before attn and MLP) | Same pattern + extra norm right after wte | Extra post-embedding norm stabilizes early layers |
| **Q/K/V projection** | Fused: `c_attn = Linear(C, 3C)` → `.split()` | Separate: `c_q`, `c_k`, `c_v` | Required for GQA (different head counts for Q vs KV) |
| **Attention heads** | Multi-Head Attention (MHA): all heads equal | Group Query Attention (GQA): `n_kv_head <= n_head` | Fewer KV heads = less memory, same quality |
| **QK normalization** | None | `norm(q)`, `norm(k)` + scale by 1.2 | Prevents attention score explosion at long sequences |
| **Attention compute** | Manual: `q @ k.T / sqrt(d)` → mask → softmax → `@ v` | Flash Attention (FA3 / SDPA) — one fused kernel | Much faster, much less memory (no (T,T) matrix) |
| **Attention window** | Full context everywhere | Per-layer sliding window (`"SSSL"` pattern) | Short windows save memory; full context only where needed |
| **MLP activation** | `nn.GELU()` | `F.relu(x).square()` (relu^2) | Sharper gating, computationally cheaper |
| **Weight tying** | `lm_head.weight = wte.weight` (shared) | Separate (untied) | Better at large scale; entrance/exit can specialize |
| **Bias terms** | `bias=True` in most layers | `bias=False` everywhere | Cleaner training, fewer params, no downside |
| **Value embeddings** | None | ResFormer-style: separate embeddings mixed into V | Gives V direct access to token identity at alternating layers |
| **Per-layer scalars** | None | `resid_lambdas`, `x0_lambdas` | Fine-grained control over residual stream per layer |
| **Smear gate** | None | Mixes previous token's embedding into current | Cheap bigram information without extra attention |
| **Backout** | None | Subtracts mid-layer residual before lm_head | Removes low-level features that hurt logit prediction |
| **Logit processing** | Raw logits → `F.cross_entropy` | Softcap: `15 * tanh(logits/15)` first | Prevents extreme logits from destabilizing training |
| **Optimizer** | `torch.optim.AdamW` (one optimizer) | AdamW (embeddings/scalars) + Muon (matrices) | Different param types benefit from different optimizers |
| **Vocab size** | 50,257 (GPT-2 tokenizer) | 32,768 (custom BPE) | Smaller vocab = faster softmax, custom to training data |
| **Context length** | 1024 | 2048 | Twice the context window |
| **Linear layer** | `nn.Linear` | Custom `Linear` subclass (casts weights to input dtype) | Replaces autocast; master weights stay fp32 for optimizer |

## What transfers directly from your notes

These concepts are identical — only the implementation details changed:

- **Embeddings are learned lookup tables** — `nn.Embedding` works the same way
- **Q, K, V are three projections of the same input** — the library/search/book analogy still holds
- **Attention = "how much should I look at each other token"** — same idea, fused kernel
- **Residual connections** — `x = x + sublayer(x)` — identical pattern
- **MLP = expand → activate → compress** — same 4x expansion, different activation
- **The block = attention half + MLP half** — same two-sublayer structure
- **Cross-entropy loss** — identical usage
- **The shape is preserved through all blocks** — `(B, T, C)` in, `(B, T, C)` out

## What's genuinely new

These need fresh learning — your notes don't cover them:

1. **Rotary embeddings (RoPE)** — rotation matrices instead of learned position vectors
2. **Group Query Attention** — fewer KV heads than Q heads
3. **Flash Attention** — fused kernel, no explicit score matrix
4. **relu^2** — why `relu(x).square()` works
5. **Per-layer scalars** — resid_lambdas, x0_lambdas
6. **Value embeddings** — ResFormer approach
7. **Smear + backout** — input-level tricks
8. **Muon optimizer** — polar decomposition for matrix params
9. **Logit softcap** — tanh squashing

---

*Back to [[Index]]*
