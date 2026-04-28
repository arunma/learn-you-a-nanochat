---
aliases: [MOC, Home, Map of Content]
---

# nanochat — Learning Vault

> *Building a language model from scratch — copywork, concepts, and code*

## How this vault works

Each section maps to a chunk of `nanochat/gpt.py` (512 lines — the whole model). Every note follows the same rhythm:

1. **Concept** — what problem this code solves and why it exists
2. **Code walkthrough** — annotated line-by-line with shape traces
3. **Versus your notes** — what changed from nanoGPT to nanochat
4. **Copywork checklist** — close the note, write it, diff it

### Color coding

| Badge | Meaning | Used for |
|-------|---------|----------|
| <span class="badge-pt">PT</span> | PyTorch built-in | Just use it — learn the API |
| <span class="badge-nc">NC</span> | nanochat custom | Understand every line |
| <span class="badge-shape">SHAPE</span> | Tensor dimensions | Track shapes through the pipeline |
| <span class="badge-new">NEW</span> | New vs nanoGPT | Things your earlier notes didn't cover |

### Callout types

> [!pytorch] PyTorch built-in
> Blue — library code, learn the interface

> [!nanochat] nanochat custom
> Green — Karpathy wrote this, understand every line

> [!shape] Shape trace
> Orange — tensor dimensions at this point

> [!keyinsight] Key insight
> Gold — the thing that clicks everything together

> [!qkv] Attention mechanics
> Purple — Q, K, V and attention internals

> [!copywork] Copywork checkpoint
> Orange pencil — close the note and write this from memory

---

## Copywork

Your code goes in `vault/copywork/`. Read a section → close it → write the code from memory → diff against `nanochat/gpt.py`.

---

## Sections

### Stage 0 — The Map
- [[00 - The Big Picture]] — The 5 stages of building an LLM, all 7 building blocks, how they compose

### Foundations (gpt.py lines 1–63)
- [[01 - Config and Imports]] — GPTConfig, imports, the model's DNA
- [[02 - Building Blocks]] — `norm()`, `Linear`, `has_ve()`, `apply_rotary_emb()`

### Attention (gpt.py lines 65–127)
- [[03 - CausalSelfAttention]] — Q/K/V projections, RoPE, flash attention, head assembly

### Feed-Forward + Block (gpt.py lines 129–152)
- [[04 - MLP and Block]] — relu^2, the two-half block structure, residual connections

### The Full Model (gpt.py lines 154–313)
- [[05 - GPT Init]] — wte, lm_head, per-layer scalars, smear, backout, value embeddings
- [[06 - Weight Initialization]] — `init_weights()` — how every parameter starts
- [[07 - Rotary Embeddings Deep Dive]] — `_precompute_rotary_embeddings()` — RoPE from scratch
- [[08 - Sliding Window Attention]] — `_compute_window_sizes()` — per-layer window patterns

### Forward Pass (gpt.py lines 416–481)
- [[09 - Forward Pass]] — the full data flow from token IDs to loss
- [[10 - Smear Gate]] — cheap bigram trick, training vs inference paths

### Optimizer + Training (gpt.py lines 374–415)
- [[11 - Optimizer Setup]] — AdamW + Muon, parameter groups, LR scaling

### Inference (gpt.py lines 483–513)
- [[12 - Generation]] — autoregressive decoding, top-k, temperature, sampling

---

## Track 2 — Hugging Face Pipeline

Copywork goes in `vault/copywork/hf/`. Same method: read → close → write from memory.

### HF Copywork Sections
- [[HF-01 - Data and Tokenizer]] — Download WikiText, train BPE, tokenize corpus
- [[HF-02 - Model and Training]] — LlamaConfig, LlamaForCausalLM, training loop
- [[HF-03 - Evaluate Generate Save]] — Perplexity, model.generate(), save/load/Hub

### HF Reference
- [[HF-04 - Deep Dive Internals]] — Every nanochat component mapped to HF equivalent (constraints, wrapping, tradeoffs)

### Pipeline and Scaling
- [[13 - Speedrun Walkthrough]] — nanochat's end-to-end training pipeline on 8×H100
- [[15 - Build Your Own LLM]] — Complete 9-stage guide with open source tools
- [[16 - HF Pipeline End to End]] — Working pipeline with cost analysis and 6× FLOPs derivation

---

## Reference

- [[Shape Cheatsheet]] — every tensor dimension from input to loss
- [[Glossary]] — terms, symbols, and their meanings
- [[nanoGPT vs nanochat]] — side-by-side comparison of the old and new architecture
- [[PyTorch API Reference]] — every PT built-in used in nanochat

---

## Key numbers (nanochat defaults)

| Symbol | Name | Value | Where it appears |
|--------|------|-------|-----------------|
| `T` | sequence_len | 2048 | GPTConfig, rotary, window sizes |
| `V` | vocab_size | 32,768 | GPTConfig, wte, lm_head |
| `N` | n_layer | 12 | GPTConfig, block loop |
| `n_head` | query heads | 6 | GPTConfig, attention |
| `n_kv_head` | KV heads (GQA) | 6 | GPTConfig, attention |
| `C` / `n_embd` | embedding dim | 768 | GPTConfig, everywhere |
| `d_h` | head_dim | 128 | `n_embd // n_head` |
| `4C` | MLP hidden | 3072 | MLP expand/compress |
