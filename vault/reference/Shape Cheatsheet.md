---
aliases: [shapes, dimensions, shape trace]
tags: [reference, shape]
---

# Shape Cheatsheet

> Every tensor dimension from token IDs to loss — nanochat defaults.
> `B` = batch size, `T` = 2048, `C` = 768, `V` = 32768, `N` = 12 layers, `n_head` = 6, `d_h` = 128

## The full pipeline

| Step | Operation | Shape | Notes |
|------|-----------|-------|-------|
| 0 | `get_batch()` | `(B, T)` | int64 — raw token IDs |
| 1 | `wte(idx)` | `(B, T, C)` | <span class="badge-shape">+C</span> each int becomes 768 floats |
| 2 | `norm(x)` | `(B, T, C)` | <span class="badge-new">NEW</span> RMSNorm right after embedding |
| 3 | smear gate | `(B, T, C)` | mix previous token's embedding in |
| — | *save x0* | `(B, T, C)` | initial embedding for x0_lambdas blending |
| 4 | `c_q(x)` | `(B, T, n_head * d_h)` | = `(B, T, 768)` query projection |
| 4 | `c_k(x)` | `(B, T, n_kv_head * d_h)` | = `(B, T, 768)` key projection |
| 4 | `c_v(x)` | `(B, T, n_kv_head * d_h)` | = `(B, T, 768)` value projection |
| 5 | `.view(B,T,H,d_h)` | `(B, T, 6, 128)` | heads labelled, no data copy |
| 6 | `apply_rotary_emb` | `(B, T, 6, 128)` | <span class="badge-new">NEW</span> RoPE replaces learned pos embeddings |
| 7 | `norm(q), norm(k)` | `(B, T, 6, 128)` | <span class="badge-new">NEW</span> QK norm for stability |
| 8 | flash attention | `(B, T, 6, 128)` | scores + value aggregation in one fused op |
| 9 | `.view(B, T, -1)` | `(B, T, C)` | heads reassembled |
| 10 | `c_proj(y)` | `(B, T, C)` | output projection back to residual |
| 11 | `x + attn_out` | `(B, T, C)` | residual connection (scaled by resid_lambda) |
| 12 | MLP: `c_fc` | `(B, T, 4C)` | = `(B, T, 3072)` expand |
| 13 | `relu(x).square()` | `(B, T, 4C)` | <span class="badge-new">NEW</span> relu^2 activation |
| 14 | MLP: `c_proj` | `(B, T, C)` | compress back |
| 15 | `x + mlp_out` | `(B, T, C)` | residual connection |
| — | *repeat N=12 times* | `(B, T, C)` | shape preserved every block |
| 16 | backout subtraction | `(B, T, C)` | <span class="badge-new">NEW</span> remove mid-layer residual |
| 17 | `norm(x)` | `(B, T, C)` | final RMSNorm |
| 18 | `lm_head(x)` | `(B, T, V)` | = `(B, T, 32768)` logits |
| 19 | softcap tanh | `(B, T, V)` | <span class="badge-new">NEW</span> squash to [-15, 15] |
| 20 | `cross_entropy` | scalar | one number — the loss |

## What's new vs nanoGPT (your earlier notes)

| nanoGPT | nanochat | Why |
|---------|----------|-----|
| Learned `wpe` (positional embedding) | Rotary embeddings (RoPE) | Relative positions, no learned params |
| `nn.LayerNorm` with learnable gamma/beta | `F.rms_norm` with no learnable params | Simpler, equally effective |
| Fused `c_attn` → `.split(C, dim=2)` | Separate `c_q`, `c_k`, `c_v` | Supports GQA (different head counts) |
| GELU activation | relu^2 | Sharper, cheaper, works well in practice |
| Weight tying (wte = lm_head) | Untied weights | Separate entrance/exit geometry |
| No QK norm | QK norm before attention | Training stability |
| Standard softmax | Logit softcap (tanh squash) | Prevents extreme logits |
| 1 norm before attention, 1 before MLP | Same pattern but RMSNorm | No bias, no learnable params |

---

*Back to [[Index]]*
