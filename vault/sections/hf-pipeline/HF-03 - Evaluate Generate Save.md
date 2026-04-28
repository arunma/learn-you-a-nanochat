---
aliases: [hf eval, hf generate, save pretrained, perplexity]
tags: [hf-section, copywork]
source: hf_pipeline/run_all.py stages 6-8
---

# HF-03 — Evaluate, Generate, and Save

<span class="phase-tag">HF TRACK</span> *Stages 6-8: measure quality, generate text, save for sharing*

> **Source:** `hf_pipeline/run_all.py` — `stage_6_evaluate()`, `stage_7_generate()`, `stage_8_save()`
> **Copywork target:** ~50 lines

---

## Stage 6: Evaluate

```python
model.eval()
total_loss = 0
n_batches = 50

with torch.no_grad():
    for _ in range(n_batches):
        x, y = get_batch(val_data, BATCH_SIZE, SEQ_LEN, DEVICE)
        outputs = model(input_ids=x, labels=y)
        total_loss += outputs.loss.item()

avg_loss = total_loss / n_batches
perplexity = math.exp(avg_loss)
bpb = avg_loss / math.log(2)    # bits per byte (approx)
```

### What the metrics mean

**Loss** — cross-entropy averaged across all positions. Lower = better. Random baseline = `log(vocab_size)` = 8.32 for vocab 4096.

**Perplexity** — `e^loss`. Intuitively: "how many tokens is the model confused between?" Perplexity 100 means the model is as uncertain as if choosing randomly between 100 options. GPT-2 on WikiText ≈ 29.

**Bits per byte (BPB)** — `loss / log(2)`. Vocab-size-invariant metric (nanochat's primary metric). Measures information-theoretic compression quality.

---

## Stage 7: Generate

```python
model.eval()

input_ids = torch.tensor(
    [tokenizer.encode(prompt).ids],
    dtype=torch.long, device=DEVICE
)

with torch.no_grad():
    generated = model.generate(
        input_ids,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.8,
        top_k=50,
    )

text = tokenizer.decode(generated[0].tolist())
```

> [!pytorch] `model.generate()` — what HF gives you for free
>
> | Parameter | What it does | nanochat equivalent |
> |-----------|-------------|-------------------|
> | `max_new_tokens` | Stop after N tokens | `max_tokens` in the for loop |
> | `do_sample=True` | Weighted random sampling | `torch.multinomial` |
> | `do_sample=False` | Greedy (always pick highest) | `torch.argmax` |
> | `temperature` | Scale logits before softmax | `logits / temperature` |
> | `top_k` | Keep only top-k candidates | Manual threshold + `-inf` |
> | `top_p` | Nucleus sampling (top-p cumulative) | Not in nanochat |
> | `num_beams` | Beam search | Not in nanochat |
> | `repetition_penalty` | Penalize repeated tokens | Not in nanochat |
> | `no_repeat_ngram_size` | Ban repeated n-grams | Not in nanochat |

---

## Stage 8: Save and reload

```python
# Save model (creates config.json + model.safetensors)
model.save_pretrained("my-first-llm")

# Wrap raw tokenizer in HF format
from transformers import PreTrainedTokenizerFast
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|bos|>",
    eos_token="<|eos|>",
    pad_token="<|pad|>",
)
hf_tokenizer.save_pretrained("my-first-llm")

# Reload — one line each
from transformers import LlamaForCausalLM, AutoTokenizer
loaded_model = LlamaForCausalLM.from_pretrained("my-first-llm")
loaded_tokenizer = AutoTokenizer.from_pretrained("my-first-llm")

# Push to Hub
model.push_to_hub("arunma/my-first-llm")
hf_tokenizer.push_to_hub("arunma/my-first-llm")
```

### What save_pretrained writes to disk

```
my-first-llm/
├── config.json              ← LlamaConfig as JSON
├── model.safetensors        ← all weight tensors (safe binary format)
├── tokenizer.json           ← BPE merge rules + vocab
├── special_tokens_map.json  ← bos, eos, pad token mappings
└── tokenizer_config.json    ← tokenizer metadata
```

> [!keyinsight] `PreTrainedTokenizerFast` wraps your raw tokenizer
> You trained the tokenizer using the `tokenizers` library (Rust). `PreTrainedTokenizerFast` wraps it to add HF conventions: special tokens, padding behavior, `return_tensors="pt"` support, and Hub compatibility. The raw tokenizer does the actual encoding — the wrapper adds ecosystem integration.

---

> [!copywork] Copywork checkpoint
> Write stages 6-8 into `vault/copywork/hf/03_eval_generate_save.py` (~50 lines):
>
> 1. Eval loop: `model.eval()`, `torch.no_grad()`, compute avg loss, perplexity, BPB
> 2. Generate: encode prompt, `model.generate(...)` with temperature and top_k
> 3. Save: `model.save_pretrained()`, wrap tokenizer, `hf_tokenizer.save_pretrained()`
> 4. Reload: `LlamaForCausalLM.from_pretrained()`, verify param count match
>
> **Common traps:**
> - Did you call `model.eval()` before eval and generate? (Disables dropout)
> - Did you use `torch.no_grad()` for eval? (Saves memory, faster)
> - Did you wrap the tokenizer with `PreTrainedTokenizerFast`? (Raw tokenizers can't save to HF format)
> - Did you include all three special tokens in the wrapper?
> - `math.exp(loss)` for perplexity, `loss / math.log(2)` for BPB

---

*Previous: [[HF-02 - Model and Training]]*
*Next: [[HF-04 - Deep Dive Internals]]*

*Back to [[Index]]*
