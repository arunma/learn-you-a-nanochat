---
aliases: [overview, big picture, stages, stage 0]
tags: [section, overview]
---

# 00 — The Big Picture

<span class="phase-tag">STAGE 0</span> *Understanding what you're building before you build it*

> Before writing a single line, you need the map. This document explains what an LLM is, what stages go into building one, and how every piece fits together. No code yet — just concepts and intuition.

---

## What is an LLM, really?

Strip away the hype and an LLM is a **next-token prediction machine**.

You give it a sequence of tokens: `["The", "cat", "sat", "on", "the"]`
It returns a probability distribution over what comes next: `{"mat": 42%, "floor": 18%, "roof": 7%, ...}`

That's it. Every capability you've seen — writing essays, coding, reasoning, translating — emerges from doing this one thing extremely well across billions of training examples.

The model never "understands" language the way you do. It learns statistical patterns: which tokens tend to follow which other tokens, in what contexts. But those patterns run deep enough that the behavior *looks* like understanding.

---

## The five stages of building an LLM

```
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │  1. TOKENIZE │────▶│  2. ARCHITECT│────▶│  3. TRAIN    │
 │              │     │              │     │              │
 │  text → ints │     │  the model   │     │  adjust      │
 │  (once)      │     │  (gpt.py)    │     │  weights     │
 └──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                      ┌──────────────┐     ┌──────▼───────┐
                      │  5. INFER    │◀────│  4. EVALUATE │
                      │              │     │              │
                      │  generate    │     │  measure     │
                      │  text        │     │  quality     │
                      └──────────────┘     └──────────────┘
```

Each stage answers one question. Let's walk through them.

---

## Stage 1 — Tokenization

> **Question:** How do you turn raw text into numbers a GPU can process?

Neural networks operate on numbers, not characters. Tokenization is the bridge.

### The problem with obvious approaches

| Approach | Example: "unhappiness" | Vocab size | Problem |
|----------|----------------------|------------|---------|
| Character-level | `u, n, h, a, p, p, i, n, e, s, s` (11 tokens) | ~100 | Wastes context window — 2048 chars isn't much text |
| Word-level | `unhappiness` (1 token) | millions | New words get no representation. "unhappiest" = unknown. |
| **BPE subword** | `un, happiness` (2-3 tokens) | 32,768 | Best of both worlds |

### BPE (Byte Pair Encoding) — how it actually works

BPE is embarrassingly simple. No neural networks, no GPU. Just counting.

1. Start with every character as its own token: `h, e, l, l, o`
2. Count every adjacent pair in the training text
3. The most frequent pair becomes a new token: `l + l` → `ll`
4. Repeat from step 2 until you have enough tokens (32,768 for nanochat)

The **ordered list of merges** IS the tokenizer. To encode new text, replay the merges in order.

### What tokenization produces

```
"The cat sat on the mat"
           ↓ tokenizer.encode()
[464, 3797, 3332, 389, 279, 2868]      ← list of integers
           ↓ torch.tensor()
tensor([464, 3797, 3332, 389, 279, 2868])  ← ready for the model
```

### In nanochat

| File | What it does |
|------|-------------|
| `nanochat/tokenizer.py` | BPE tokenizer (GPT-4 style), encode/decode, special tokens |
| `scripts/tok_train.py` | Train the tokenizer on raw text |
| `nanochat/dataset.py` | Download text data, prepare shards |
| `nanochat/dataloader.py` | Load token sequences into batches for training |

> [!keyinsight] Tokenization happens ONCE, before training
> The tokenizer is not part of the neural network. It has no trainable parameters. It runs on CPU, converts text to integers, and saves them to disk. The model never sees raw text — only token IDs.

---

## Stage 2 — Model Architecture

> **Question:** Given a sequence of token IDs, how do you predict the next token?

This is `nanochat/gpt.py` — 512 lines, the entire model. This is where your copywork lives.

### The data flow — from integers to prediction

```
Token IDs:  [464, 3797, 3332, 389, 279, 2868]
                │
                ▼
        ┌───────────────┐
        │  EMBEDDING    │  Each integer → 768-dimensional float vector
        │               │  "464" → [0.21, -0.54, 0.88, ..., 0.13]  (768 numbers)
        │  + position   │  Rotary embeddings encode WHERE each token sits
        │  + normalize  │  Scale values to consistent range
        └───────┬───────┘
                │  shape: (B, T, 768) — B sequences, T tokens, 768 dims each
                ▼
        ┌───────────────────────────────────────────┐
        │         TRANSFORMER BLOCK  ×  12          │
        │                                           │
        │  ┌─────────────────────────────────────┐  │
        │  │  SELF-ATTENTION                     │  │
        │  │                                     │  │
        │  │  Every token asks: "which other     │  │
        │  │  tokens in this sequence are         │  │
        │  │  relevant to predicting what         │  │
        │  │  comes after ME?"                    │  │
        │  │                                     │  │
        │  │  Produces a weighted mix of          │  │
        │  │  information from relevant tokens.   │  │
        │  └─────────────────────────────────────┘  │
        │  x = x + attention_output  (residual)     │
        │                                           │
        │  ┌─────────────────────────────────────┐  │
        │  │  MLP (FEED-FORWARD)                 │  │
        │  │                                     │  │
        │  │  Each token independently processes  │  │
        │  │  what it gathered from attention.    │  │
        │  │  "Think about what I just learned."  │  │
        │  │                                     │  │
        │  │  Expand: 768 → 3072  (more room)    │  │
        │  │  Activate: relu²    (non-linear)    │  │
        │  │  Compress: 3072 → 768 (back down)   │  │
        │  └─────────────────────────────────────┘  │
        │  x = x + mlp_output  (residual)           │
        │                                           │
        │  Shape in = shape out = (B, T, 768)       │
        │  Meaning changes. Dimensions don't.       │
        └───────────────────┬───────────────────────┘
                            │  12 rounds of this
                            ▼
        ┌───────────────┐
        │  OUTPUT HEAD  │  768 dims → 32,768 scores (one per vocab token)
        │  (lm_head)    │  Highest score = model's prediction
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │  LOSS         │  Compare prediction against the actual next token
        │               │  loss = -log(probability of correct token)
        └───────────────┘
                │
                ▼
           single number → used to update weights in Stage 3
```

### The building blocks inside the model

There are exactly **7 building blocks** in nanochat's transformer. Everything else is glue.

#### 1. Embedding (`nn.Embedding`)

A lookup table. Token ID 464 → go to row 464 → get 768 floats.

- **wte** (word token embedding): (32768, 768) = 25.2M parameters
- No matrix multiply. Pure row lookup. Extremely fast.
- The 768 numbers start random and are learned during training
- After training, tokens used in similar contexts have similar vectors

> [!keyinsight] Why 768 dimensions?
> Each dimension captures some aspect of meaning. No single dimension means "noun" or "positive sentiment" — meaning is distributed across all 768 numbers. More dimensions = more capacity for nuance, but more compute per token.

#### 2. Positional encoding (Rotary Embeddings / RoPE)

Attention has no built-in sense of order. Without position info, "cat bit dog" and "dog bit cat" look identical.

nanochat uses **Rotary Position Embeddings (RoPE)** — instead of adding a position vector (like your notes described), it **rotates** the Q and K vectors by an angle that depends on position.

Two tokens close together rotate by similar angles → their dot product is high.
Two tokens far apart rotate by different angles → their dot product decays.

The key insight: position becomes **relative**, not absolute. The model learns "how far apart are these tokens" rather than "this token is at position 47."

No learned parameters. Just precomputed sine/cosine values.

#### 3. Normalization (`F.rms_norm`)

Before every attention and MLP layer, the input is normalized so values have consistent scale.

Without normalization: some vectors might have values around 0.001, others around 1000. The next layer has to cope with this wild variation. Training becomes unstable.

With RMSNorm: divide each vector by its root-mean-square. Now everything is roughly unit scale. Stable. Consistent.

```
Before: [200, -400, 100, 300]   ← wild range
After:  [0.63, -1.26, 0.32, 0.95]  ← consistent scale
```

nanochat uses `F.rms_norm` — no learned gamma/beta parameters, unlike your notes' `nn.LayerNorm`.

#### 4. Self-Attention (the core mechanism)

This is the transformer's superpower. Every token looks at every other (past) token and asks: "how relevant are you to me?"

Three projections of each token:
- **Q (Query):** "What am I looking for?"
- **K (Key):** "What do I advertise?"
- **V (Value):** "What do I actually give?"

Score = Q dot K (how well does my question match your advertisement?)
Output = weighted sum of V (gather information proportional to scores)

The **causal mask** ensures tokens can only look backward — position 5 cannot attend to positions 6, 7, 8, etc. This is what makes it a *language model* (predicting the future, not peeking at it).

> [!qkv] Why three separate projections?
> A token's embedding carries everything about it in one vector. But "what am I searching for" (Q) is a different question than "what do I contain" (K) which is different from "what information should I share" (V).
>
> A book might have a great catalogue entry (strong K) but poor content (weak V), or vice versa. Separating these lets the model learn nuanced retrieval patterns.

#### 5. MLP (Feed-Forward Network)

After attention gathers information from other tokens, the MLP processes it *per token independently*.

```
768  →  3072  →  768
     expand    compress
      ↑ relu² activation in between
```

Attention = "gather information from context"
MLP = "think about what I gathered"

The 4× expansion (768 → 3072) gives more working space for intermediate computation. The activation function (relu²) introduces non-linearity — without it, stacking layers would just be a bigger linear transformation (useless).

#### 6. Residual connections

Every sublayer uses: `x = x + sublayer(x)`

The `+ x` is the residual connection. It means:
- Gradients flow directly backward through the `+` (they don't vanish in deep networks)
- Each layer adds *refinement* rather than computing from scratch
- If a layer learns nothing useful, the `+ 0` pass-through doesn't damage the signal

Without residuals, 12-layer transformers are essentially untrainable. Gradients die.

#### 7. Output head (`lm_head`)

A single linear projection: 768 dims → 32,768 scores. One score per token in the vocabulary. The highest score is the prediction.

```
768-dim context vector → lm_head → 32768 scores (logits)
                                     ↓
                            "mat": 8.7  ← highest = prediction
                            "floor": 6.2
                            "roof": 3.1
                            ...
```

These raw scores (logits) are turned into probabilities via softmax, then compared against the actual next token using cross-entropy loss.

### How these blocks compose

```
Block(config, layer_idx):
    def forward(x):
        x = x + attention(norm(x))    ← gather info from context
        x = x + mlp(norm(x))          ← process independently
        return x                       ← same shape (B, T, 768)
```

That's two lines. The entire transformer block is two lines of logic. The complexity is inside `attention` and `mlp`, not in how they compose.

---

## Stage 3 — Training

> **Question:** How do you adjust millions of weights so predictions improve?

Training is a loop:

```
for step in range(num_iterations):
    1. Load a batch of token sequences     (data)
    2. Forward pass: predict next tokens   (model)
    3. Compute loss: how wrong were we?    (cross-entropy)
    4. Backward pass: compute gradients    (autograd)
    5. Update weights: nudge toward less wrong  (optimizer)
```

### The training signal

The genius of language model training is self-supervision. You don't need labeled data. The text IS the labels.

Given `["The", "cat", "sat", "on"]`:
- Position 0: given `[The]`, correct answer = `cat`
- Position 1: given `[The, cat]`, correct answer = `sat`
- Position 2: given `[The, cat, sat]`, correct answer = `on`

One sequence of T tokens gives T training examples. The causal mask ensures each position only sees its past.

### Loss function

```
loss = -log(probability assigned to the correct token)
```

- Model gives "cat" 95% probability → loss = 0.05 (good)
- Model gives "cat" 1% probability → loss = 4.61 (bad)
- Loss = 0 means perfect prediction. Loss → ∞ means completely wrong.

The loss is averaged across all B×T token positions in the batch, producing one scalar number.

### Backpropagation (the "backward pass")

This is where calculus enters. `loss.backward()` computes: for every single weight in the model, how much would the loss change if I nudged that weight slightly?

These are **gradients** — the direction and magnitude of the steepest descent. You don't need to implement backprop yourself; PyTorch does it automatically via its autograd engine.

> [!keyinsight] The chain rule — backprop in one sentence
> If `loss` depends on `y` which depends on `x`, then:
> `d(loss)/d(x) = d(loss)/d(y) × d(y)/d(x)`
>
> PyTorch chains this rule through every operation in the forward pass, back to every weight. That's all backprop is — repeated application of the chain rule.

### Optimizer

The optimizer takes the gradients and decides how to update each weight.

nanochat uses two optimizers:
- **AdamW** — for embeddings, scalars, and the output head. Tracks momentum and variance per parameter for adaptive learning rates.
- **Muon** — for the matrix parameters in attention and MLP. Uses a technique called polar decomposition for better updates.

You don't need to understand optimizer internals to write the model. Just know: `optimizer.step()` uses the gradients to make every weight slightly better.

### In nanochat

| File | What it does |
|------|-------------|
| `scripts/base_train.py` | The training loop (630 lines) |
| `nanochat/optim.py` | AdamW + Muon optimizer |
| `nanochat/dataloader.py` | Loads batches of token sequences |

---

## Stage 4 — Evaluation

> **Question:** Is the model actually getting smarter?

Two main metrics in nanochat:

**Bits Per Byte (BPB):** How many bits of information does the model need per byte of text? Lower = better compression = better understanding. Vocab-size-invariant (so you can compare models with different tokenizers).

**CORE Score:** Few-shot accuracy across 10 downstream tasks (ARC, MMLU, BoolQ, etc.). "Can this model actually answer questions?" The target is 0.2565 — matching original GPT-2.

### In nanochat

| File | What it does |
|------|-------------|
| `nanochat/loss_eval.py` | Bits-per-byte evaluation |
| `nanochat/core_eval.py` | CORE metric (10-task accuracy) |
| `scripts/base_eval.py` | Run all evaluations |

---

## Stage 5 — Inference (Generation)

> **Question:** How do you generate text one token at a time?

Generation is the forward pass in a loop, without targets:

```
tokens = [start_token]
for i in range(max_tokens):
    logits = model.forward(tokens)       # predict next token scores
    next_token = sample(logits[-1])      # pick one from the distribution
    tokens.append(next_token)            # add it to the sequence
    yield next_token                     # stream it out
```

Two sampling strategies:
- **Greedy:** always pick the highest-scoring token (deterministic)
- **Sampling:** pick randomly, weighted by probabilities (creative, varied)

**Temperature** controls randomness: temperature < 1 = more conservative, temperature > 1 = more creative.

**Top-k:** only consider the top k highest-scoring tokens (ignore the long tail of unlikely tokens).

**KV Cache:** During generation, tokens 0 through t-1 have already been processed. Their K and V values don't change. The KV cache stores them so you only compute the new token at each step — massive speedup.

### In nanochat

| File | What it does |
|------|-------------|
| `nanochat/engine.py` | Inference engine with KV cache (357 lines) |
| `nanochat/gpt.py` `generate()` | Simple generation without KV cache (30 lines) |
| `scripts/chat_cli.py` | Interactive CLI chat |
| `scripts/chat_web.py` | Web UI chat server |

---

## Stage 3.5 (optional) — Fine-tuning

> **Question:** How do you turn a base model into a chatbot?

The base model predicts next tokens — it doesn't know how to have a conversation. Fine-tuning teaches it a format:

```
<human>What is the capital of France?</human>
<assistant>The capital of France is Paris.</assistant>
```

**Supervised Fine-Tuning (SFT):** Train on human-written conversations. The model learns to respond in the assistant role.

**Reinforcement Learning (RL):** Further refine responses using reward signals (human preferences or automated evaluation).

### In nanochat

| File | What it does |
|------|-------------|
| `scripts/chat_sft.py` | Supervised fine-tuning |
| `scripts/chat_rl.py` | Reinforcement learning |
| `tasks/` | Evaluation tasks (MMLU, GSM8K, ARC, etc.) |

---

## The one file that matters most

All five stages matter, but **Stage 2 (Model Architecture) is the core**.

`nanochat/gpt.py` — 512 lines. That's the entire transformer. Everything else feeds it or uses it.

Your copywork journey through this file:

| Lines | What | Building block |
|-------|------|---------------|
| 1–40 | Config + imports | The blueprint |
| 42–63 | `norm()`, `Linear`, `apply_rotary_emb()` | Helper functions |
| 65–127 | `CausalSelfAttention` | The attention mechanism |
| 129–139 | `MLP` | The feed-forward network |
| 142–152 | `Block` | Attention + MLP composed |
| 154–199 | `GPT.__init__()` | Wire everything together |
| 201–267 | `GPT.init_weights()` | How parameters start |
| 268–312 | Rotary embeddings + sliding window | Position encoding + window computation |
| 416–481 | `GPT.forward()` | The complete data flow |
| 483–513 | `GPT.generate()` | Autoregressive inference |

---

## What's next

You now have the map. Next step: [[01 - Config and Imports]] — the first 40 lines of `gpt.py`.

The copywork directory is: `vault/copywork/`

Read the section → close it → write `copywork/01_config.py` from memory → diff against `nanochat/gpt.py`.

---

*Back to [[Index]]*
