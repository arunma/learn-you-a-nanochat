---
aliases: [hf model, hf training, LlamaConfig, pretrain]
tags: [hf-section, copywork]
source: hf_pipeline/run_all.py stages 4-5
---

# HF-02 — Build Model and Pretrain

<span class="phase-tag">HF TRACK</span> *Stages 4-5: create a LLaMA model and train it*

> **Source:** `hf_pipeline/run_all.py` — `stage_4_build_model()`, `stage_5_pretrain()`
> **Copywork target:** ~55 lines

---

## What these stages do

Stage 4 creates a GPT model using HF's `LlamaForCausalLM` — the off-the-shelf architecture closest to nanochat. Stage 5 runs a training loop that's nearly identical to what you'd write with raw PyTorch.

---

## Stage 4: Build the model

```python
from transformers import LlamaConfig, LlamaForCausalLM

config = LlamaConfig(
    vocab_size=4096,
    hidden_size=256,              # n_embd in nanochat
    num_hidden_layers=4,          # n_layer in nanochat
    num_attention_heads=4,        # n_head in nanochat
    num_key_value_heads=4,        # n_kv_head in nanochat (GQA)
    intermediate_size=4 * 256,    # MLP 4x expansion
    max_position_embeddings=256,  # sequence_len in nanochat
    rope_theta=100000,            # RoPE base — same as nanochat
    rms_norm_eps=1e-6,
    use_cache=False,              # disable KV cache during training
    tie_word_embeddings=False,    # untied — same as nanochat
)

model = LlamaForCausalLM(config)
model = model.to(DEVICE)
```

### Config field mapping

| nanochat GPTConfig | LlamaConfig | Demo | Full |
|-------------------|-------------|------|------|
| `n_embd = 768` | `hidden_size` | 256 | 768 |
| `n_layer = 12` | `num_hidden_layers` | 4 | 12 |
| `n_head = 6` | `num_attention_heads` | 4 | 6 |
| `n_kv_head = 6` | `num_key_value_heads` | 4 | 6 |
| `4 * n_embd` | `intermediate_size` | 1024 | 3072 |
| `sequence_len = 2048` | `max_position_embeddings` | 256 | 2048 |
| `(base=100000)` | `rope_theta` | 100000 | 100000 |

> [!keyinsight] What LlamaForCausalLM gives you for free
> When you call `LlamaForCausalLM(config)`, HF builds the entire model:
> - Token embeddings (`model.model.embed_tokens`)
> - N transformer blocks, each with attention (RoPE, GQA) + MLP
> - RMSNorm (no learned params)
> - Output head (`model.lm_head`)
> - `forward()` that accepts `input_ids` and `labels`, returns loss + logits
> - `generate()` with all sampling strategies
>
> This is the same architecture you wrote by hand in `my_nanochat/gpt.py` — just pre-built.

> [!nanochat] What LlamaForCausalLM does NOT have (vs nanochat)
> - No per-layer scalars (resid_lambdas, x0_lambdas)
> - No smear gate
> - No backout
> - No value embeddings
> - No logit softcap
> - No sliding window pattern (though LLaMA 3 supports it)
> - Uses SiLU activation, not relu²
>
> These are nanochat-specific optimizations from the modded-nanogpt speedrun community.

---

## Stage 5: Pretrain

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))

model.train()
for step in range(NUM_STEPS):
    # Forward
    x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN, DEVICE)
    outputs = model(input_ids=x, labels=y)
    loss = outputs.loss

    # Backward
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    # Evaluate periodically
    if step % EVAL_EVERY == 0:
        model.eval()
        with torch.no_grad():
            val_losses = []
            for _ in range(10):
                vx, vy = get_batch(val_data, BATCH_SIZE, SEQ_LEN, DEVICE)
                val_out = model(input_ids=vx, labels=vy)
                val_losses.append(val_out.loss.item())
            val_loss = sum(val_losses) / len(val_losses)
        model.train()
        print(f"Step {step} | train: {loss.item():.4f} | val: {val_loss:.4f}")
```

> [!shape] Shape trace — one training step
> ```
> STEP                      SHAPE                 NOTES
> ─────────────────────────────────────────────────────────────
> x = get_batch(...)        (8, 256)              B=8 sequences of T=256
> y = get_batch(...)        (8, 256)              targets (x shifted by 1)
>
> model(input_ids=x)
>   embed_tokens(x)         (8, 256, 256)         token embeddings
>   + RoPE                  (8, 256, 256)         position via rotation
>   × 4 blocks              (8, 256, 256)         attention + MLP
>   lm_head(x)              (8, 256, 4096)        logits
>
> outputs.loss              scalar                cross-entropy
> outputs.logits            (8, 256, 4096)        raw scores
>
> loss.backward()           —                     compute all gradients
> clip_grad_norm_           —                     cap gradient magnitude
> optimizer.step()          —                     update all weights
> optimizer.zero_grad()     —                     clear gradients for next step
> ```

### The 5-line training loop pattern

This is the universal PyTorch training loop. It's the same whether you use nanochat, HF, or raw PyTorch:

```
1. forward:     outputs = model(x, labels=y)     → compute loss
2. backward:    loss.backward()                   → compute gradients
3. clip:        clip_grad_norm_(params, 1.0)      → safety belt
4. step:        optimizer.step()                  → update weights
5. zero:        optimizer.zero_grad()             → clear for next step
```

> [!versus] nanochat vs HF training loop
>
> | nanochat | HF (this code) |
> |----------|---------------|
> | `logits, loss = model(x, y)` | `outputs = model(input_ids=x, labels=y)` |
> | `loss` returned directly | `outputs.loss` (attribute access) |
> | Muon + AdamW (hybrid) | AdamW only |
> | `optimizer.zero_grad(set_to_none=True)` | `optimizer.zero_grad()` |
> | DDP via `torchrun` | Same (or use HF Trainer) |

### Alternative: HF Trainer (zero-loop approach)

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=8,
    learning_rate=3e-4,
    max_steps=500,
    bf16=True,
    logging_steps=50,
)

trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()  # one line — handles everything
```

HF Trainer wraps the same 5 lines plus: gradient accumulation, mixed precision, distributed training, logging, checkpointing, evaluation, and learning rate scheduling. For learning purposes, the manual loop is better — you see exactly what happens.

---

> [!copywork] Copywork checkpoint
> Write stages 4-5 into `vault/copywork/hf/02_model_training.py` (~55 lines):
>
> 1. `LlamaConfig(...)` with all fields
> 2. `LlamaForCausalLM(config).to(DEVICE)`
> 3. `AdamW` optimizer with `lr=3e-4, betas=(0.9, 0.95)`
> 4. Training loop: forward → backward → clip → step → zero
> 5. Eval loop inside `torch.no_grad()` + `model.eval()`
>
> **Common traps:**
> - Did you write `model(input_ids=x, labels=y)` not `model(x, y)`?
> - Did you access `outputs.loss` not `outputs["loss"]`? (CausalLMOutput uses attributes)
> - Did you call `model.train()` after eval? (Otherwise dropout stays off)
> - Did you write `use_cache=False` in config? (KV cache breaks training)
> - Did you write `tie_word_embeddings=False`? (Default is True in LLaMA)

---

*Previous: [[HF-01 - Data and Tokenizer]]*
*Next: [[HF-03 - Evaluate Generate Save]]*

*Back to [[Index]]*
