---
aliases: [HF pipeline, end to end, run_all, build LLM hands-on]
tags: [section, hands-on]
source: hf_pipeline/run_all.py
---

# 16 — HF Pipeline: Hands-On End to End

<span class="phase-tag">HANDS-ON</span> *A working LLM pipeline you can run on your laptop in 30 seconds*

> **Source:** `hf_pipeline/run_all.py` (~280 lines)
> **Run it:** `python -m hf_pipeline.run_all`

---

## What this is

A complete, runnable implementation of all 8 stages of building an LLM — using Hugging Face and open source tools. Designed to run on a laptop (CPU/MPS) with a tiny model so you can see the full pipeline work end-to-end.

The architecture uses `LlamaForCausalLM` — which makes the same design choices as nanochat (RoPE, RMSNorm, GQA, no bias, untied embeddings).

---

## The stages and what each one produces

```
python -m hf_pipeline.run_all

Stage 1 — Download WikiText-2         → 10.9M chars of clean Wikipedia
Stage 2 — Train BPE tokenizer          → 4096-token vocabulary, 3.6 chars/token
Stage 3 — Tokenize dataset             → 3.07M train tokens, 322K val tokens
Stage 4 — Build LLaMA model            → 6.3M parameters (4 layers, 4 heads, 256 dim)
Stage 5 — Pretrain (500 steps)          → val loss: 6.47, perplexity: 644
Stage 6 — Evaluate                      → full validation metrics
Stage 7 — Generate text                 → (gibberish — tiny model, few steps)
Stage 8 — Save & reload                 → HF format, ready for Hub push
```

---

## Stage-by-stage walkthrough

### Stage 1 — Download data

```python
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
```

> [!pytorch] `datasets` library
> HF `datasets` downloads, caches, and streams datasets from the Hub. `streaming=True` for huge datasets (terabytes) that don't fit in RAM. WikiText-2 is small enough to load entirely.

**At scale:** Replace `wikitext` with `HuggingFaceFW/fineweb-edu` (1.3T tokens) and use `streaming=True`.

### Stage 2 — Train BPE tokenizer

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(vocab_size=4096, special_tokens=["<|pad|>", "<|bos|>", "<|eos|>"])
tokenizer.train_from_iterator(text_chunks, trainer=trainer)
```

> [!keyinsight] Same BPE algorithm as your Section 00 notes
> The `tokenizers` library implements the same BPE merge algorithm you learned about — but compiled in Rust for speed. Training on 10M characters takes <1 second vs minutes in pure Python.

**At scale:** Set `vocab_size=32768` (nanochat) or `128256` (LLaMA 3). Train on ~2B+ characters.

### Stage 3 — Tokenize

```python
train_ids = tokenizer.encode(train_text).ids
train_tensor = torch.tensor(train_ids, dtype=torch.long)
```

Same as nanochat's `prepare.py` — convert text to integers once, save, load during training.

### Stage 4 — Build model

```python
from transformers import LlamaConfig, LlamaForCausalLM

config = LlamaConfig(
    vocab_size=4096,
    hidden_size=256,            # n_embd
    num_hidden_layers=4,        # n_layer
    num_attention_heads=4,      # n_head
    num_key_value_heads=4,      # n_kv_head (GQA)
    intermediate_size=1024,     # 4 * n_embd (MLP expansion)
    max_position_embeddings=256,
    rope_theta=100000,          # RoPE base
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,  # untied, like nanochat
)
model = LlamaForCausalLM(config)
```

> [!versus] nanochat GPTConfig vs LlamaConfig mapping
>
> | nanochat GPTConfig | LlamaConfig | Value |
> |-------------------|-------------|-------|
> | `n_embd` | `hidden_size` | 256 (demo) / 768 (full) |
> | `n_layer` | `num_hidden_layers` | 4 / 12 |
> | `n_head` | `num_attention_heads` | 4 / 6 |
> | `n_kv_head` | `num_key_value_heads` | 4 / 6 |
> | `4 * n_embd` | `intermediate_size` | 1024 / 3072 |
> | `sequence_len` | `max_position_embeddings` | 256 / 2048 |
> | (base=100000) | `rope_theta` | 100000 |
> | (no wpe) | (RoPE built-in) | RoPE |

### Stage 5 — Pretrain

Custom training loop — intentionally simple, matches nanochat's pattern:

```python
for step in range(500):
    x, y = get_batch(train_data, batch_size=8, seq_len=256)
    outputs = model(input_ids=x, labels=y)
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
```

> [!shape] Training shapes
> ```
> x (input_ids)     (8, 256)        B=8 sequences of T=256 tokens
> y (labels)        (8, 256)        same shape, shifted by 1
> outputs.logits    (8, 256, 4096)  one score per vocab token per position
> outputs.loss      scalar          cross-entropy averaged across B*T
> ```

**At scale:** Use HF Trainer, add gradient accumulation, learning rate scheduling, distributed training (DDP/FSDP), mixed precision, and checkpointing.

### Stage 6 — Evaluate

```python
perplexity = math.exp(avg_val_loss)
bpb = avg_val_loss / math.log(2)
```

Our tiny model achieved perplexity ~644 (random would be 4096). GPT-2 on WikiText gets ~29. The gap is entirely about scale — more params, more data, more steps.

### Stage 7 — Generate

```python
generated = model.generate(
    input_ids,
    max_new_tokens=60,
    do_sample=True,
    temperature=0.8,
    top_k=50,
)
```

HF's `model.generate()` handles everything — no manual loop needed. Output is gibberish because the model is tiny and undertrained, but the pipeline is correct.

### Stage 8 — Save and reload

```python
model.save_pretrained("path")           # saves config.json + model.safetensors
hf_tokenizer.save_pretrained("path")    # saves tokenizer.json + special_tokens_map.json

# Reload with one line:
model = LlamaForCausalLM.from_pretrained("path")

# Push to Hub:
model.push_to_hub("arunma/my-first-llm")
```

---

## Cost: library doesn't matter, compute does

The cost of training is determined by one formula:

```
Total FLOPs ≈ 6 × params × tokens
```

| | Value | Source |
|-|-------|--------|
| params | 300M | Your model size |
| tokens | 10B | Your training data |
| FLOPs | 6 × 300M × 10B = 18 exaFLOPs | The math |
| H100 speed | ~1000 TFLOPs/s (bf16) | Hardware spec |
| Time (1 GPU) | ~5 hours | 18e18 / 3.6e18 |
| Time (8 GPUs) | ~40 minutes | With 80% scaling |
| **Cost** | **~$30-50** | At $24/hr for 8×H100 |

This cost is the same whether you use nanochat, HF Trainer, raw PyTorch, or any other framework. The GPU does identical matrix multiplies regardless of wrapping. Framework overhead is 0-10% — a few dollars difference on a $50 run.

### Cost scaling

| Model | Params | Tokens | FLOPs | 8×H100 time | Cost |
|-------|--------|--------|-------|-------------|------|
| This demo | 6M | 1M | tiny | 20 seconds | $0 |
| nanochat speedrun | ~300M | ~10B | 18 exaFLOPs | ~2 hours | ~$50 |
| LLaMA-7B | 7B | 1T | 42 zettaFLOPs | ~1 week (64 GPUs) | ~$75K |
| LLaMA-70B | 70B | 2T | 840 zettaFLOPs | ~3 weeks (2048 GPUs) | ~$1.5M |

The jump from 300M to 7B is 23× in params and 100× in tokens = **2,333× more compute**. That's where $50 becomes $75K.

---

*Back to [[Index]]*
