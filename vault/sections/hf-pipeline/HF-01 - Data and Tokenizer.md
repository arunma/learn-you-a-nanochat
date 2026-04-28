---
aliases: [hf data, hf tokenizer, BPE training, datasets]
tags: [hf-section, copywork]
source: hf_pipeline/run_all.py stages 1-3
---

# HF-01 — Data Download, Tokenizer Training, and Tokenization

<span class="phase-tag">HF TRACK</span> *Stages 1-3: from raw text to integer tensors*

> **Source:** `hf_pipeline/run_all.py` — `stage_1_download_data()`, `stage_2_train_tokenizer()`, `stage_3_tokenize()`
> **Copywork target:** ~60 lines

---

## What these stages do

Before any neural network runs, you need:
1. Text data to train on
2. A tokenizer to convert text → integers
3. The tokenized dataset as tensors on disk

These three stages are **pure data prep** — no GPU, no model, no gradients.

---

## Stage 1: Download data

```python
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_text = "\n".join([t for t in dataset["train"]["text"] if len(t) > 0])
val_text = "\n".join([t for t in dataset["validation"]["text"] if len(t) > 0])
```

> [!pytorch] `datasets` library — key concepts
> - `load_dataset("name", "config")` — downloads from HF Hub, caches locally
> - Returns a `DatasetDict` with splits: `train`, `validation`, `test`
> - Each split is a table of rows, access columns by name: `dataset["train"]["text"]`
> - For huge datasets, use `streaming=True` — processes row by row, never loads all into RAM

> [!shape] What we get
> ```
> dataset["train"]["text"]     list of ~36K strings (Wikipedia paragraphs)
> train_text                   one big string, ~10.9M characters
> val_text                     one big string, ~1.1M characters
> ```

**At scale:** Replace `"wikitext"` with `"HuggingFaceFW/fineweb-edu"` (1.3T tokens). Add `streaming=True`. Process in chunks.

---

## Stage 2: Train BPE tokenizer

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# Create a blank BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

# Configure the trainer
trainer = trainers.BpeTrainer(
    vocab_size=4096,
    special_tokens=["<|pad|>", "<|bos|>", "<|eos|>"],
    min_frequency=2,
    show_progress=True,
)

# Train from text chunks (memory efficient)
tokenizer.train_from_iterator(
    [train_text[i:i+10000] for i in range(0, len(train_text), 10000)],
    trainer=trainer,
)

# Save
tokenizer.save("tokenizer.json")
```

### The components

| Component | What it does | Why |
|-----------|-------------|-----|
| `models.BPE()` | The merge-based subword algorithm | Same algorithm as your Section 00 notes |
| `pre_tokenizers.ByteLevel` | Splits on bytes before BPE | Handles any Unicode character |
| `decoders.ByteLevel` | Reverses byte-level encoding back to text | Needed for `decode()` |
| `BpeTrainer` | Runs the merge algorithm | `vocab_size` controls how many merges |
| `special_tokens` | Reserved tokens (pad, bos, eos) | IDs 0, 1, 2 — never merged |
| `min_frequency=2` | Skip pairs that appear only once | Reduces noise |
| `train_from_iterator` | Feed text in chunks | Memory-efficient for large corpora |

> [!versus] nanochat vs HF tokenizer
>
> | nanochat | HF `tokenizers` |
> |----------|-----------------|
> | `nanochat/tokenizer.py` + `rustbpe` | `tokenizers` library (also Rust) |
> | Custom BPE implementation | Standard BPE with many options |
> | `scripts/tok_train.py` | 10 lines of Python |
> | Vocab: 32768 | Any size you want |

> [!keyinsight] The tokenizer is NOT part of the model
> It has no trainable parameters. It runs on CPU. You train it once and save it. The model only ever sees integer IDs — it never sees raw text. Changing the tokenizer means retraining the model from scratch (because the integer IDs change meaning).

---

## Stage 3: Tokenize

```python
train_ids = tokenizer.encode(train_text).ids
val_ids = tokenizer.encode(val_text).ids

train_tensor = torch.tensor(train_ids, dtype=torch.long)
val_tensor = torch.tensor(val_ids, dtype=torch.long)
```

> [!shape] Shape trace
> ```
> train_text                   string, 10.9M characters
> tokenizer.encode(...)        Encoding object with .ids, .tokens
> .ids                         list[int], ~3.07M integers
> torch.tensor(..., long)      (3073683,)  — one long 1D tensor
>
> val_text                     string, 1.1M characters
> → val_tensor                 (321853,)
> ```

This is equivalent to nanochat's `prepare.py` writing `train.bin` and `val.bin`.

---

## The `get_batch` function

```python
def get_batch(data, batch_size, seq_len, device):
    """Random batch of sequences from the token stream."""
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y
```

> [!shape] Shape trace — get_batch
> ```
> data                    (3073683,)         full tokenized corpus
> ix                      (8,)               8 random start positions
> data[i:i+256]           (256,)             one sequence of 256 tokens
> torch.stack(...)         (8, 256)           batch of 8 sequences
>
> x                       (8, 256)           input: positions 0..255
> y                       (8, 256)           target: positions 1..256 (shifted by 1)
> ```

> [!keyinsight] x and y are shifted by 1 — the training signal
> Position 0: given `x[0]`, predict `y[0]` (which is `x[1]`)
> Position 1: given `x[0:2]`, predict `y[1]` (which is `x[2]`)
> ...
> One sequence of 256 tokens = 256 training examples. Same idea as nanochat.

---

> [!copywork] Copywork checkpoint
> Write stages 1-3 + `get_batch` into `vault/copywork/hf/01_data_tokenizer.py` (~60 lines):
>
> 1. `load_dataset("wikitext", "wikitext-2-raw-v1")` and join text
> 2. Create `Tokenizer(models.BPE())` with ByteLevel pre-tokenizer and decoder
> 3. `BpeTrainer(vocab_size=4096, special_tokens=[...], min_frequency=2)`
> 4. `train_from_iterator` with text chunks
> 5. `tokenizer.encode(text).ids` → `torch.tensor(..., dtype=torch.long)`
> 6. `get_batch` function
>
> **Common traps:**
> - Did you write `pre_tokenizers.ByteLevel` (not `BytePair`)?
> - Did you include `decoders.ByteLevel()`? (Without it, `decode()` fails)
> - Did you add `add_prefix_space=False`?
> - Did you use `torch.long` not `torch.int`? (Embedding requires int64)
> - In get_batch: did you shift y by +1? `data[i+1:i+seq_len+1]`

---

*Next: [[HF-02 - Model and Training]]*

*Back to [[Index]]*
