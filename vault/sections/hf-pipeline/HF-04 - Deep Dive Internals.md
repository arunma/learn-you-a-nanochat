---
aliases: [hf internals, deep dive, library mapping, constraints]
tags: [hf-section, deep-dive]
---

# HF-04 — Deep Dive: nanochat Internals vs HF Libraries

<span class="phase-tag">DEEP DIVE</span> *Every nanochat component mapped to its HF equivalent — what it wraps, what it hides, and what constraints the libraries impose*

> This is the most important reference section for understanding what open source LLM libraries actually do under the hood.

---

## The big picture: layers of abstraction

```
YOUR CODE (top)
    │
    ▼
HF Transformers (model wrappers, configs, generation)
    │
    ▼
PyTorch (tensors, autograd, nn.Module, optimizers)
    │
    ▼
CUDA / cuDNN / cuBLAS / Triton (GPU kernels)
    │
    ▼
GPU HARDWARE (bottom)
```

nanochat operates at layer 2 (PyTorch) with some dips into layer 3 (Flash Attention kernels). HF Transformers adds layer 1 on top. The GPU does the same work either way.

---

## Component 1: Configuration

### nanochat — `@dataclass GPTConfig`

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
```

A plain Python dataclass. 7 fields. No methods. No serialization. You pass it to `GPT(config)` and that's it.

### HF — `PretrainedConfig`

```python
class LlamaConfig(PretrainedConfig):
    model_type = "llama"
    # 40+ fields including: hidden_size, num_hidden_layers, num_attention_heads,
    # num_key_value_heads, intermediate_size, hidden_act, max_position_embeddings,
    # rope_theta, rms_norm_eps, use_cache, tie_word_embeddings, attention_bias,
    # attention_dropout, mlp_bias, torch_dtype, ...
```

**What HF adds:**
- `config.save_pretrained("path")` → writes `config.json`
- `LlamaConfig.from_pretrained("path")` → reads `config.json`
- `AutoConfig.from_pretrained("meta-llama/Llama-3-8B")` → auto-detects type
- JSON serialization / deserialization
- Validation of field combinations
- Backward compatibility with older config versions

**What HF constrains:**
- Must use their field names (`hidden_size` not `n_embd`, `num_hidden_layers` not `n_layer`)
- `model_type` must be registered for Auto classes to work
- `**kwargs` must be passed to `super().__init__()` (HF passes metadata you don't control)
- Some fields have side effects: `use_cache=True` changes how `forward()` behaves
- Default values may differ from what you expect (e.g., `tie_word_embeddings=True` in LLaMA)

**What HF hides:**
- Config versioning and migration across library versions
- Auto-detection logic (`model_type` → class mapping)
- The ~30 fields you never set but exist for edge cases

---

## Component 2: Embeddings

### nanochat

```python
# Token embedding — a lookup table
self.transformer = nn.ModuleDict({
    "wte": nn.Embedding(padded_vocab_size, config.n_embd),
})

# In forward:
x = self.transformer.wte(idx)    # (B, T) → (B, T, 768)
x = norm(x)                       # RMSNorm immediately after
```

No positional embedding module. RoPE is applied later inside attention.

### HF LLaMA

```python
# Inside LlamaModel.__init__():
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.padding_idx)

# Inside LlamaModel.forward():
inputs_embeds = self.embed_tokens(input_ids)
```

**What's the same:** `nn.Embedding` is identical — pure row lookup. Same PyTorch primitive underneath.

**What HF adds:**
- `padding_idx` support (optional — embedding for pad token stays zero)
- Accepts `inputs_embeds` directly (skip the lookup — useful for prompt tuning)
- Handles dtype casting

**What nanochat adds that HF doesn't:**
- Vocab padding to multiple of 64 (tensor core alignment)
- RMSNorm immediately after embedding
- Smear gate (mixing previous token's embedding in)

> [!keyinsight] The embedding IS the same
> `nn.Embedding` is `nn.Embedding`. There is no HF magic here. Both libraries call the same PyTorch operation: given integer ID `i`, return row `i` of the weight matrix. The differences are in what happens BEFORE (padding) and AFTER (norm, smear).

---

## Component 3: Positional Encoding (RoPE)

### nanochat

```python
# Precompute in __init__:
cos, sin = self._precompute_rotary_embeddings(seq_len, head_dim)
self.register_buffer("cos", cos, persistent=False)
self.register_buffer("sin", sin, persistent=False)

# Apply in CausalSelfAttention.forward:
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
```

Custom implementation: 16 lines for precompute, 6 lines for application. You control the base theta, the broadcasting shape, everything.

### HF LLaMA

```python
# Inside LlamaAttention.__init__:
self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

# Inside LlamaAttention.forward:
cos, sin = self.rotary_emb(value_states, position_ids)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

**What HF wraps into `LlamaRotaryEmbedding`:**
- The same `inv_freq = 1.0 / (base ** (arange / head_dim))` computation
- Support for different RoPE scaling methods:
  - `linear` — simple frequency scaling for longer contexts
  - `dynamic` — NTK-aware dynamic scaling
  - `yarn` — Yet Another RoPE extensioN (best for length extrapolation)
  - `llama3` — LLaMA 3's custom scaling recipe
- Automatic caching (only recomputes if sequence length changes)
- `position_ids` support (for non-contiguous positions in KV cache)

**What HF constrains:**
- Must use their `LlamaRotaryEmbedding` class (or subclass it)
- RoPE scaling type is configured via `config.rope_scaling` dict
- The internal representation uses `(B, n_head, T, d_h)` layout (transposed — different from nanochat's `(B, T, n_head, d_h)`)

**What HF hides:**
- The `inv_freq` computation
- Cache management for different sequence lengths
- The complexity of scaling methods (yarn alone is ~50 lines of math)

> [!keyinsight] Same math, different packaging
> `apply_rotary_pos_emb` in HF does the exact same rotation:
> `y1 = x1 * cos + x2 * sin`, `y2 = -x1 * sin + x2 * cos`.
> But HF interleaves the dimensions (pairs are adjacent: dims 0,1 then 2,3) while nanochat splits in half (first half / second half). Same result, different memory layout.

---

## Component 4: Attention

### nanochat

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        self.c_q = Linear(n_embd, n_head * head_dim, bias=False)
        self.c_k = Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_v = Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_proj = Linear(n_embd, n_embd, bias=False)
        self.ve_gate = Linear(12, n_kv_head, bias=False) if has_ve(...) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        q = self.c_q(x).view(B, T, n_head, head_dim)      # (B, T, H, D) — FA3 layout
        k = self.c_k(x).view(B, T, n_kv_head, head_dim)
        v = self.c_v(x).view(B, T, n_kv_head, head_dim)
        # VE gating, RoPE, QK norm, * 1.2
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)
```

62 lines. Separate Q/K/V, Flash Attention with sliding window, value embeddings, QK norm.

### HF LLaMA

```python
class LlamaAttention(nn.Module):
    def __init__(self, config, layer_idx):
        self.q_proj = nn.Linear(hidden_size, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_size, bias=attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(self, hidden_states, attention_mask, position_ids, past_key_value, ...):
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # Reshape: (B, T, H, D) → (B, H, T, D)    ← NOTE: transposed!
        query_states = query_states.view(B, T, n_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(B, T, n_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(B, T, n_kv_heads, head_dim).transpose(1, 2)
        # RoPE
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # KV cache
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, layer_idx)
        # GQA: repeat KV heads to match Q heads
        key_states = repeat_kv(key_states, n_rep)
        value_states = repeat_kv(value_states, n_rep)
        # Attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attn_weights = attn_weights + attention_mask   # causal mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, T, -1)
        return self.o_proj(attn_output)
```

**What HF adds:**
- `attention_mask` parameter (supports padding masks, custom masks, prefix caching)
- `position_ids` parameter (non-contiguous positions for batched inference)
- `past_key_value` (KV cache management integrated)
- `repeat_kv()` — explicitly copies KV heads to match Q heads for GQA
- Output `attentions` (optional — return attention weights for visualization)
- Multiple attention implementations: `eager` (manual matmul), `sdpa` (PyTorch SDPA), `flash_attention_2`

**What HF constrains:**
- Head layout is `(B, H, T, D)` — transposed. nanochat uses `(B, T, H, D)` for Flash Attention's native layout
- GQA is handled by `repeat_kv()` which physically copies K/V tensors (nanochat lets FA3 handle it internally)
- Causal mask is additive (add -inf), not multiplicative. Passed as a full `(1, 1, T, T)` tensor
- No sliding window in the default eager path (requires `flash_attention_2` backend)
- No QK norm (not in standard LLaMA — nanochat adds this)

**What HF hides:**
- The choice between eager/SDPA/Flash backends (controlled by `config._attn_implementation`)
- KV cache growth and memory management
- The `repeat_kv` expansion for GQA
- Mask construction (different masks for padding, causal, prefix)

> [!keyinsight] The `.transpose(1, 2)` difference
> Your nanoGPT notes had `.transpose(1, 2)` — moving heads to dim 1 for manual `q @ k.T` matmul. nanochat removed it because Flash Attention expects `(B, T, H, D)`. HF **keeps it** because their default `eager` attention path uses manual matmul. This is a real architectural difference that affects performance — the transpose costs memory and time.

---

## Component 5: MLP / Feed-Forward

### nanochat

```python
class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc = Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()    # relu²
        return self.c_proj(x)
```

Two linear layers, relu² activation. 6 lines.

### HF LLaMA

```python
class LlamaMLP(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]  # typically SiLU

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

**What's different:**
- LLaMA uses a **gated** MLP: `SiLU(gate(x)) * up(x)` — three linear layers, not two
- The activation is SiLU (Sigmoid Linear Unit), not relu²
- `intermediate_size` is configured independently (not forced to 4× hidden_size)

> [!shape] Shape comparison — MLP variants
> ```
> nanochat MLP (2 matrices):
>   x (B,T,768) → c_fc (768,3072) → relu² → c_proj (3072,768) → (B,T,768)
>   Parameters: 768×3072 + 3072×768 = 4.7M
>
> LLaMA MLP (3 matrices, gated):
>   x (B,T,768) → gate_proj (768,3072) → SiLU ──┐
>   x (B,T,768) → up_proj (768,3072) ────────────┤ element-wise multiply
>                                                  ↓
>                  (B,T,3072) → down_proj (3072,768) → (B,T,768)
>   Parameters: 768×3072 × 3 = 7.1M  (50% more params)
> ```

**What HF constrains:**
- Activation function is configured via string in `config.hidden_act` (mapped through `ACT2FN` dict)
- Can't easily use relu² — you'd need to subclass `LlamaMLP` or register a custom activation
- The gated structure (3 matrices) is hardcoded in `LlamaMLP` — you can't switch to nanochat's 2-matrix design without replacing the class

---

## Component 6: Normalization

### nanochat

```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

One line. No learnable parameters. Called as a plain function.

### HF LLaMA

```python
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # learnable scale!
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states
```

**What HF adds:**
- Learnable `weight` parameter (γ scale factor) — nanochat strips this out
- Configurable epsilon (default 1e-6)
- Stored as `nn.Module` (not a plain function)

**What this means in practice:**
- HF LLaMA RMSNorm has `hidden_size` extra parameters per norm (768 per call)
- nanochat has zero extra parameters per norm
- With 2 norms per block × 12 blocks + 1 final norm = 25 norms
- HF: 25 × 768 = 19,200 extra params. nanochat: 0.
- At scale this is negligible, but it's a philosophical difference: nanochat prefers "no learnable params if not needed"

---

## Component 7: Residual and Block

### nanochat

```python
class Block(nn.Module):
    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x
```

Two lines. Pre-norm residual. `norm()` is a function call.

### HF LLaMA

```python
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask, position_ids, past_key_value, ...):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids, past_key_value, ...)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

**What HF adds:**
- Explicit `residual = hidden_states` variable (clearer, more debuggable)
- Norm modules stored as attributes (`self.input_layernorm`, `self.post_attention_layernorm`)
- Returns optional `attentions`, `present_key_value` for caching and visualization
- Gradient checkpointing support (trade compute for memory)

**What nanochat adds that HF doesn't:**
- Per-layer `resid_lambdas` scaling the residual
- Per-layer `x0_lambdas` blending in the original embedding
- Value embeddings passed through to attention
- Sliding window sizes passed through to attention

---

## Component 8: The Full Model

### nanochat — `GPT(nn.Module)`

```python
class GPT(nn.Module):
    def __init__(self, config):
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(...),
            "h": nn.ModuleList([Block(...) for _ in range(n_layer)]),
        })
        self.lm_head = Linear(...)
        self.resid_lambdas = nn.Parameter(...)
        self.x0_lambdas = nn.Parameter(...)
        self.smear_gate = Linear(...)
        self.backout_lambda = nn.Parameter(...)
        self.value_embeds = nn.ModuleDict(...)
        self.register_buffer("cos", ...), self.register_buffer("sin", ...)
```

### HF — `LlamaForCausalLM(PreTrainedModel)`

```python
class LlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config):
        self.model = LlamaModel(config)  # embeddings + blocks + final norm
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, i) for i in range(n_layer)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
```

**What `PreTrainedModel` gives you:**

| Method | What it does | nanochat equivalent |
|--------|-------------|-------------------|
| `save_pretrained("path")` | Save config + weights to disk | `torch.save(model.state_dict(), path)` |
| `from_pretrained("path")` | Load model from disk or Hub | `model.load_state_dict(torch.load(path))` |
| `push_to_hub("name")` | Upload to Hugging Face Hub | N/A |
| `num_parameters()` | Count params (trainable/total) | `sum(p.numel() for p in model.parameters())` |
| `gradient_checkpointing_enable()` | Trade compute for memory | N/A |
| `resize_token_embeddings(n)` | Change vocab size after init | Manual resize |
| `tie_weights()` | Share wte and lm_head weights | `self.lm_head.weight = self.transformer.wte.weight` |
| `_init_weights(module)` | Per-module weight initialization | `init_weights()` (one big function) |

**What `PreTrainedModel` constrains:**
- `forward()` must accept HF-convention args (`input_ids`, `labels`, `attention_mask`, etc.)
- Return type should be a `ModelOutput` subclass (not a plain tuple or dict)
- Weight initialization is per-module (`_init_weights(module)`) not per-model — less control
- The model tree structure must match the checkpoint format (can't easily rearrange)

---

## Component 9: Training Loop

### nanochat — `scripts/base_train.py`

```python
for step in range(num_iterations):
    x, y = next(dataloader)
    logits, loss = model(x, y)
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
```

Custom loop, custom optimizer (Muon + AdamW), custom LR schedule, custom data loading.

### HF Trainer

```python
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        warmup_steps=100,
        max_steps=50000,
        bf16=True,
        logging_steps=10,
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=500,
    ),
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()
```

**What Trainer gives you:**
- Gradient accumulation (simulate larger batches than GPU memory allows)
- Mixed precision (`bf16=True` — automatic autocast)
- Distributed training (DDP out of the box, FSDP with `fsdp` arg)
- Learning rate scheduling (linear warmup + cosine decay default)
- Automatic checkpointing (save every N steps, keep best K)
- Logging (TensorBoard, wandb, CSV — all via `report_to` arg)
- Evaluation during training (run eval dataset every N steps)
- Gradient clipping (built-in, configurable)
- Early stopping (via callbacks)
- Resume from checkpoint (automatic if checkpoint dir exists)

**What Trainer constrains:**
- Must use their `TrainingArguments` (many options, not always obvious)
- Dataset must be HF `Dataset` format (or at least iterable with right column names)
- Optimizer defaults to AdamW (can override with `optimizers` param, but complex)
- Can't easily use Muon or other exotic optimizers
- LR schedule is limited to built-in options (linear, cosine, constant, polynomial)
- The inner loop is hidden — debugging training issues means reading Trainer source code
- Data collation assumptions may not match your data format

---

## Component 10: Generation / Inference

### nanochat — `GPT.generate()` + `nanochat/engine.py`

```python
# Simple version (gpt.py): no KV cache, full recompute each step
for _ in range(max_tokens):
    logits = self.forward(ids)
    logits = logits[:, -1, :]
    # manual top-k, temperature, sampling
    yield token

# Optimized version (engine.py): KV cache, Flash Attention with cache
y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
```

### HF — `GenerationMixin.generate()`

```python
output = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    num_beams=4,
    early_stopping=True,
    no_repeat_ngram_size=3,
    length_penalty=1.0,
    use_cache=True,
)
```

**What `generate()` provides beyond nanochat:**

| Feature | In nanochat | In HF |
|---------|------------|-------|
| Greedy decoding | Yes | Yes |
| Temperature sampling | Yes | Yes |
| Top-k | Yes | Yes |
| Top-p (nucleus) | No | Yes |
| Beam search | No | Yes |
| Repetition penalty | No | Yes |
| No-repeat n-gram | No | Yes |
| Length penalty | No | Yes |
| Diverse beam search | No | Yes |
| Constrained generation | No | Yes |
| Contrastive search | No | Yes |
| Assisted decoding (speculative) | No | Yes |
| KV cache | Yes (engine.py) | Yes (automatic) |
| Streaming | Yes (yield) | Yes (TextStreamer) |
| Stopping criteria | No | Yes (StoppingCriteria) |
| Logits processors | No | Yes (LogitsProcessorList) |

**What HF constrains:**
- Must implement `prepare_inputs_for_generation()` for custom models
- Return type must be `ModelOutput` with `logits` attribute
- KV cache format must be `DynamicCache` compatible
- `config.num_hidden_layers` must exist (for cache initialization)
- Generation config is complex — `GenerationConfig` has 50+ fields

---

## Summary: when each approach wins

| Scenario | nanochat wins | HF wins |
|----------|-------------|---------|
| **Learning** | Yes — you see every line | No — too much hidden |
| **Research** (new architectures) | Yes — fully hackable | No — constraints fight you |
| **Research** (established architectures) | Maybe | Yes — LlamaConfig is instant |
| **Speed records** | Yes — less overhead | No — framework overhead |
| **Production inference** | No | Yes — vLLM, TGI integrate with HF |
| **Sharing models** | No | Yes — Hub ecosystem |
| **Team collaboration** | Maybe | Yes — standard interfaces |
| **Fine-tuning** (LoRA/QLoRA) | No | Yes — PEFT library |
| **Multi-GPU training** | Both work | Trainer is easier |

> [!keyinsight] The fundamental tradeoff
> nanochat gives you **control** — you understand and can modify everything. HF gives you **velocity** — you ship faster by accepting their abstractions. For learning, start with nanochat (which you did). For building products, use HF. For research, it depends on whether your work fits inside HF's constraints.

---

*Previous: [[HF-03 - Evaluate Generate Save]]*

*Back to [[Index]]*
