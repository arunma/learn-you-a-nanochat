---
aliases: [generate, inference, sampling, top-k, temperature]
tags: [section, phase-12]
source: nanochat/gpt.py lines 483–513
---

# 12 — Generation

<span class="phase-tag">SECTION 12</span> *Autoregressive decoding — generating text one token at a time*

> **Source:** `nanochat/gpt.py` lines 483–513
> **Copywork target:** ~30 lines

---

## What this code does

After training, `generate()` produces text by repeatedly: predict the next token → append it → predict the next → append → ... This is **autoregressive generation** — each new token conditions on all previous tokens.

This is the naive version (no KV cache). The optimized version lives in `nanochat/engine.py`.

---

## The code

```python
@torch.inference_mode()
def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
    """
    Naive autoregressive streaming inference.
    Assumes batch size 1, tokens are Python lists/ints.
    """
    assert isinstance(tokens, list)
    device = self.get_device()
    rng = None
    if temperature > 0:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

    ids = torch.tensor([tokens], dtype=torch.long, device=device)  # (1, T)

    for _ in range(max_tokens):
        logits = self.forward(ids)         # (1, T, V) — full forward pass
        logits = logits[:, -1, :]          # (1, V)    — last position only

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        if temperature > 0:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
        else:
            next_ids = torch.argmax(logits, dim=-1, keepdim=True)

        ids = torch.cat((ids, next_ids), dim=1)
        token = next_ids.item()
        yield token
```

> [!shape] Shape trace — one generation step
> ```
> EACH ITERATION
> ids                     (1, T)               all tokens so far
> self.forward(ids)       (1, T, 32768)        logits for all positions
> [:, -1, :]              (1, 32768)           last position's logits only
>
> TOP-K FILTERING (if enabled)
> torch.topk(logits, k)   (1, k)              top-k values
> logits[< threshold]     (1, 32768)           non-top-k set to -inf
>
> SAMPLING (temperature > 0)
> logits / temperature     (1, 32768)           scaled logits
> F.softmax(...)           (1, 32768)           probabilities summing to 1
> torch.multinomial(...)   (1, 1)               sampled token ID
>
> OR GREEDY (temperature = 0)
> torch.argmax(...)        (1, 1)               highest-scoring token ID
>
> APPEND
> torch.cat(ids, next)     (1, T+1)            sequence grows by 1
> yield token              int                  stream out the new token
> ```

### Temperature — controlling randomness

Temperature scales the logits before softmax:

```
Original logits: [8.7, 6.2, 2.1, -1.4]

Temperature = 0.5 (conservative):
  [17.4, 12.4, 4.2, -2.8] → softmax → [99.3%, 0.7%, 0.0%, 0.0%]
  Almost always picks the top token.

Temperature = 1.0 (normal):
  [8.7, 6.2, 2.1, -1.4]  → softmax → [92.3%, 7.4%, 0.3%, 0.0%]
  Mostly top token, some variation.

Temperature = 2.0 (creative):
  [4.35, 3.1, 1.05, -0.7] → softmax → [58%, 28%, 10%, 4%]
  Much more variety — lower-ranked tokens have real chances.
```

### Top-k — cutting the long tail

Without top-k, every token in the 32,768 vocabulary has some (tiny) probability. Top-k keeps only the k most likely tokens and sets the rest to -infinity (probability 0 after softmax).

```python
v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
logits[logits < v[:, [-1]]] = -float('Inf')
```

`v[:, [-1]]` is the k-th highest value — the threshold. Everything below becomes -inf.

### `@torch.inference_mode()` vs `@torch.no_grad()`

Both disable gradient tracking. `inference_mode()` is faster because it also disables version tracking (PyTorch's mechanism for detecting in-place modifications). Safe to use when you know you won't call `.backward()`.

### The `yield` — streaming

`generate()` is a **Python generator** (uses `yield` instead of `return`). Each token is yielded as it's generated — the caller can print tokens in real-time without waiting for the full sequence.

```python
for token in model.generate(prompt_ids, max_tokens=100):
    print(tokenizer.decode([token]), end='', flush=True)
```

### Why this is "naive"

Every call to `self.forward(ids)` reprocesses the **entire sequence** from scratch. As the sequence grows from T to T+100, you're doing 100 full forward passes of increasing length. The KV cache version (in `engine.py`) stores intermediate results and only processes the new token each step — much faster.

> [!versus] nanoGPT vs nanochat — generation
>
> | nanoGPT (your notes) | nanochat |
> |---------------------|----------|
> | `@torch.no_grad()` | `@torch.inference_mode()` (faster) |
> | `idx_cond = idx[:, -block_size:]` (crop to context) | No cropping — rotary handles any length |
> | No seeded RNG | `torch.Generator` with explicit seed (reproducible) |
> | Returns full tensor | `yield` — streams tokens one at a time |
> | No top-k | Top-k filtering supported |

---

> [!copywork] Copywork checkpoint
> Close this note. From memory, write `generate()` (~30 lines):
>
> 1. `@torch.inference_mode()` decorator
> 2. Assert tokens is a list, create seeded RNG
> 3. `ids = torch.tensor([tokens], dtype=torch.long, device=device)`
> 4. Loop: forward → slice last → top-k → temperature → sample or argmax
> 5. `torch.cat` to grow sequence, `yield token`
>
> **Common traps:**
> - Did you wrap tokens in `[tokens]` to add the batch dimension?
> - Did you write `logits[:, -1, :]` (slice last position, not first)?
> - Did you write `v[:, [-1]]` with `[-1]` (keepdim) not just `[-1]`?
> - Did you use `generator=rng` in `torch.multinomial`?
> - Did you use `yield` not `return`?
> - Did you handle `temperature = 0` (greedy) vs `temperature > 0` (sampling)?

---

*Previous: [[11 - Optimizer Setup]]*

*Back to [[Index]]*
