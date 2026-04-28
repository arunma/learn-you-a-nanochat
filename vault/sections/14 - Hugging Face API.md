---
aliases: [hugging face, HF, transformers, PreTrainedModel]
tags: [section, alternative]
source: hf_nanochat/model.py
---

# 14 — Building with Hugging Face Transformers

<span class="phase-tag">ALTERNATIVE</span> *The same GPT architecture, wrapped in HF APIs*

> **Source:** `hf_nanochat/model.py`
> **Purpose:** Show how nanochat maps to the Hugging Face ecosystem

---

## Why a HF version?

nanochat is deliberately minimal — no frameworks, no abstractions. That's perfect for learning. But in production, most people use Hugging Face Transformers because it gives you:

1. **`model.generate()`** — beam search, top-k, top-p, temperature, repetition penalty, all built-in
2. **`model.save_pretrained()` / `from_pretrained()`** — one-line save/load
3. **`model.push_to_hub()`** — share on Hugging Face Hub
4. **`Trainer` API** — fine-tuning with built-in logging, checkpointing, eval
5. **`pipeline("text-generation")`** — zero-code inference

The architecture is identical. Only the wrapping changes.

---

## The mapping — nanochat → Hugging Face

| nanochat | Hugging Face | What changes |
|----------|-------------|-------------|
| `@dataclass GPTConfig` | `class NanoChatConfig(PretrainedConfig)` | JSON serialization, `from_pretrained()` support |
| `class GPT(nn.Module)` | `class NanoChatModel(PreTrainedModel)` | Save/load, generate(), Hub integration |
| `def forward(self, idx, targets)` | `def forward(self, input_ids, labels)` | HF naming convention, returns dict |
| `def generate()` (manual loop) | Inherited from `GenerationMixin` | All sampling strategies built-in |
| `def init_weights()` | `def _init_weights(module)` | HF calls it per-module automatically |
| `torch.save(model.state_dict())` | `model.save_pretrained("path")` | Saves config + weights + metadata |
| Manual tokenizer | `AutoTokenizer.from_pretrained()` | Hub-hosted tokenizer |

---

## Key differences in the code

### Config: `PretrainedConfig` vs `@dataclass`

```python
# nanochat — plain dataclass
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    ...

# Hugging Face — inherits PretrainedConfig
class NanoChatConfig(PretrainedConfig):
    model_type = "nanochat"     # ← required for Auto class registration

    def __init__(self, sequence_len=2048, vocab_size=32768, n_layer=12, ..., **kwargs):
        self.sequence_len = sequence_len
        ...
        super().__init__(**kwargs)  # ← HF passes extra fields (torch_dtype, etc.)
```

`PretrainedConfig` gives you `config.save_pretrained()` and `NanoChatConfig.from_pretrained()` for free. The `**kwargs` is important — HF passes metadata fields that your config doesn't know about.

### Forward: return dict, use HF naming

```python
# nanochat
def forward(self, idx, targets=None):
    ...
    if targets is not None:
        loss = F.cross_entropy(...)
        return loss
    return logits

# Hugging Face
def forward(self, input_ids, labels=None, **kwargs):
    ...
    loss = None
    if labels is not None:
        loss = F.cross_entropy(...)
    return {"loss": loss, "logits": logits}    # ← dict with both
```

HF conventions:
- `input_ids` not `idx`
- `labels` not `targets`
- Always return **both** loss and logits in a dict (Trainer needs both)
- Accept `**kwargs` to handle extra arguments HF might pass

### Weight init: per-module pattern

```python
# nanochat — one big function, explicit per-parameter
def init_weights(self):
    torch.nn.init.normal_(self.transformer.wte.weight, std=0.8)
    torch.nn.init.normal_(self.lm_head.weight, std=0.001)
    for block in self.transformer.h:
        torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
        ...

# Hugging Face — called once per module, check type
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, std=0.02)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, std=0.02)
```

HF's `post_init()` walks the module tree and calls `_init_weights()` on every sub-module. You check the type and init accordingly. Less control than nanochat's explicit approach, but simpler.

### Generation: free vs manual

```python
# nanochat — manual loop
@torch.inference_mode()
def generate(self, tokens, max_tokens, temperature=1.0, top_k=None):
    ids = torch.tensor([tokens], ...)
    for _ in range(max_tokens):
        logits = self.forward(ids)
        logits = logits[:, -1, :]
        # manual top-k, temperature, sampling...
        yield token

# Hugging Face — one line, all strategies built-in
output = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    num_beams=4,           # beam search!
)
```

`GenerationMixin` provides `model.generate()` with dozens of options. You just need to implement `prepare_inputs_for_generation()` to tell HF how to feed inputs to your model.

---

## What stays identical

The actual model architecture — attention, MLP, block composition — is the same:

```python
# These are identical in both versions:
class CausalSelfAttention(nn.Module):
    # Q, K, V projections, RoPE, QK norm, attention, reassemble

class MLP(nn.Module):
    # expand, relu², compress

class Block(nn.Module):
    # x = x + attn(norm(x))
    # x = x + mlp(norm(x))
```

The building blocks don't change. HF is a wrapper around the same math.

---

## Running the HF version

```bash
# From the project root:
python -m hf_nanochat.model
```

This runs the built-in demo which:
1. Creates a small model (4 layers, 256 dim)
2. Forward pass with random input
3. Computes loss with labels
4. Generates text using HF's `model.generate()`
5. Saves and reloads the model
6. Compares parameter counts

---

## When to use which

| Use case | Use nanochat | Use HF |
|----------|-------------|--------|
| **Learning** how transformers work | Yes | No — too much abstraction |
| **Research** on architecture changes | Yes — minimal, hackable | Maybe — depends on complexity |
| **Production** inference | No | Yes — optimized pipelines |
| **Sharing** models with others | No | Yes — Hub ecosystem |
| **Fine-tuning** on custom data | Either | Yes — Trainer handles boilerplate |
| **Speed records** / benchmarking | Yes — less overhead | No — framework overhead |

---

*Back to [[Index]]*
