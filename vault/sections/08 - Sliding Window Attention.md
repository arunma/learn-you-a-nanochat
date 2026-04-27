---
aliases: [sliding window, window sizes, compute window]
tags: [section, phase-8]
source: nanochat/gpt.py lines 285–312
---

# 08 — Sliding Window Attention

<span class="phase-tag">SECTION 8</span> *Per-layer window patterns — which layers see the full context*

> **Source:** `nanochat/gpt.py` lines 285–312
> **Copywork target:** ~28 lines

---

## What this code does

Not every layer needs to attend to the full 2048-token context. `_compute_window_sizes()` converts the pattern string `"SSSL"` into a list of `(left, right)` window tuples — one per layer. Short-window layers save memory (attention is quadratic in window size). The final layer always gets full context.

---

## The code

```python
def _compute_window_sizes(self, config):
    pattern = config.window_pattern.upper()
    assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}"

    long_window = config.sequence_len                                    # 2048
    short_window = -(-long_window // 4 // 128) * 128                    # ceil to FA3 tile: 512 → 512
    char_to_window = {
        "L": (long_window, 0),     # (2048, 0) — full context, causal
        "S": (short_window, 0),    # (512, 0)  — quarter context, causal
    }

    window_sizes = []
    for layer_idx in range(config.n_layer):
        char = pattern[layer_idx % len(pattern)]
        window_sizes.append(char_to_window[char])

    window_sizes[-1] = (long_window, 0)  # final layer always full context
    return window_sizes
```

### Concrete output with defaults

Pattern `"SSSL"` tiled across 12 layers:

```
Layer  0:  S → (512, 0)     short window
Layer  1:  S → (512, 0)     short window
Layer  2:  S → (512, 0)     short window
Layer  3:  L → (2048, 0)    full context
Layer  4:  S → (512, 0)     short window    (pattern repeats)
Layer  5:  S → (512, 0)     short window
Layer  6:  S → (512, 0)     short window
Layer  7:  L → (2048, 0)    full context
Layer  8:  S → (512, 0)     short window
Layer  9:  S → (512, 0)     short window
Layer 10:  S → (512, 0)     short window
Layer 11:  L → (2048, 0)    full context    (forced — final layer override)
```

9 short layers, 3 long layers. Short layers use 4× less attention memory.

### The short window formula

```python
short_window = -(-long_window // 4 // 128) * 128
```

This is ceiling division to the nearest Flash Attention tile size (128):

```
long_window = 2048
long_window // 4 = 512         ← quarter context
-(-512 // 128) = -(−4) = 4    ← ceiling division trick
4 * 128 = 512                  ← aligned to FA3 tile
```

The `-(-x // y)` pattern is Python's ceiling division: round up instead of down. FA3 processes attention in tiles of 128 — an unaligned window would waste compute.

> [!keyinsight] Why sliding windows help
> Attention is O(T²). With T=2048, each full-context layer computes 2048² = 4.2M scores per head. With a 512-token window, that drops to 2048 × 512 = 1.0M scores — a 4× reduction. Most local patterns (grammar, adjacent words) only need short context. Long-range patterns (coreference, topic) are handled by the few full-context layers.

### The window_size tuple

```
(left, right)
left = how many tokens BEFORE current position to attend to
right = how many tokens AFTER (always 0 for causal — can't see future)
```

Flash Attention uses this tuple directly. `-1` means unlimited (not used here but supported).

---

> [!copywork] Copywork checkpoint
> 1. Pattern validation with `assert all(c in "SL" ...)`
> 2. `long_window = config.sequence_len`
> 3. `short_window` — the ceiling division to 128 alignment
> 4. `char_to_window` dict mapping S/L to tuples
> 5. Loop tiling pattern across layers with `%`
> 6. Override last layer to full context
>
> **Common traps:**
> - Did you write `-(-long_window // 4 // 128) * 128` (the ceiling trick)?
> - Did you write `pattern[layer_idx % len(pattern)]` (modulo for tiling)?
> - Did you override `window_sizes[-1]` after the loop?

---

*Previous: [[07 - Rotary Embeddings Deep Dive]]*
*Next: [[09 - Forward Pass]]*

*Back to [[Index]]*
