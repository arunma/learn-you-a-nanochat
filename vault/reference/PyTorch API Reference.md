---
aliases: [pytorch, pt reference]
tags: [reference, pt]
---

# PyTorch API Reference

> Every <span class="badge-pt">PT</span> built-in used in nanochat/gpt.py — grouped by purpose.

## Tensor creation

| API | What it does | Used in |
|-----|-------------|---------|
| `torch.arange(n, device=)` | `[0, 1, 2, ..., n-1]` on device | Rotary embeddings |
| `torch.ones(n)` | Tensor of 1s | resid_lambdas init |
| `torch.zeros(n)` | Tensor of 0s | x0_lambdas, smear_lambda init |
| `torch.outer(a, b)` | Outer product: every pair `a[i] * b[j]` | Rotary frequency grid |
| `torch.tensor(data, dtype=)` | Python list → tensor | Generation (token list) |
| `torch.cat([t1, t2], dim=)` | Concatenate along a dimension | Rotary halves, smear, generation |
| `torch.topk(x, k)` | Top-k values and indices | Top-k sampling |

## Neural network layers

| API | What it does | Used in |
|-----|-------------|---------|
| `nn.Embedding(num, dim)` | Lookup table: int → row of floats | wte, wpe, value_embeds |
| `nn.Linear(in, out, bias=)` | `y = xW^T + b` | c_q, c_k, c_v, c_proj, c_fc, lm_head |
| `nn.Module` | Base class for all layers | Every class |
| `nn.ModuleDict({})` | Named dict of layers (tracked by PyTorch) | transformer dict |
| `nn.ModuleList([])` | Ordered list of layers (tracked) | transformer.h blocks |
| `nn.Parameter(tensor)` | Tensor registered as trainable | resid_lambdas, x0_lambdas, etc. |

## Functional operations

| API | What it does | Used in |
|-----|-------------|---------|
| `F.rms_norm(x, (dim,))` | RMSNorm: `x / sqrt(mean(x^2) + eps)` | `norm()` function |
| `F.linear(x, weight)` | Matrix multiply without bias | Custom `Linear.forward()` |
| `F.relu(x)` | `max(0, x)` — zero out negatives | MLP activation |
| `F.cross_entropy(logits, targets)` | Fused log-softmax + NLL loss | Loss computation |
| `F.softmax(x, dim=)` | Probabilities summing to 1.0 | Generation sampling |

## Tensor manipulation

| API | What it does | Used in |
|-----|-------------|---------|
| `.view(*shape)` | Reshape without copying data | Head splitting |
| `.contiguous()` | Ensure memory is contiguous | Before `.view()` after attention |
| `.to(dtype=)` / `.to(device=)` | Cast dtype or move device | Everywhere |
| `.size()` / `.shape` | Get tensor dimensions | Everywhere |
| `.float()` | Cast to float32 | Logits before softcap |
| `.square()` | Element-wise `x^2` | relu^2 activation |
| `.item()` | Single-element tensor → Python number | Generation yield |
| `tensor[..., :n]` | Slice last dimension | Vocab padding removal, smear |

## Training utilities

| API | What it does | Used in |
|-----|-------------|---------|
| `torch.no_grad()` | Disable gradient tracking | init_weights |
| `torch.inference_mode()` | Faster than no_grad, disables autograd entirely | generate() |
| `register_buffer(name, tensor)` | Saved with model but NOT trainable | cos, sin (rotary) |
| `torch.sigmoid(x)` | `1 / (1 + exp(-x))` — output in (0, 1) | Smear gate, VE gate |
| `torch.tanh(x)` | Output in (-1, 1) | Logit softcap |
| `torch.multinomial(probs, n)` | Weighted random sampling | Generation |
| `torch.argmax(x, dim=)` | Index of maximum value | Greedy decoding |

## Initialization

| API | What it does | Used in |
|-----|-------------|---------|
| `nn.init.normal_(t, mean, std)` | Fill with normal distribution | wte, lm_head |
| `nn.init.uniform_(t, a, b)` | Fill with uniform distribution | c_q/k/v, c_fc, gates |
| `nn.init.zeros_(t)` | Fill with zeros | c_proj (residual start) |
| `nn.init.constant_(t, val)` | Fill with constant | backout_lambda |

---

*Back to [[Index]]*
