"""
nanochat reimplemented using Hugging Face Transformers APIs.

This shows how the same GPT architecture maps to HF's abstractions:
  nanochat                    →  Hugging Face
  ─────────────────────────────────────────────
  GPTConfig (dataclass)       →  PretrainedConfig
  GPT(nn.Module)              →  PreTrainedModel
  CausalSelfAttention         →  same (custom module)
  MLP                         →  same (custom module)
  Block                       →  same (custom module)
  generate() (manual loop)    →  model.generate() (HF built-in)
  tokenizer (custom BPE)      →  AutoTokenizer (HF hub)

Why use HF APIs?
  1. model.generate() handles beam search, top-k, top-p, temperature, repetition penalty, etc.
  2. model.push_to_hub() shares your model on Hugging Face Hub
  3. AutoModel.from_pretrained() loads it back with one line
  4. Pipeline API for zero-code inference
  5. Trainer API for fine-tuning with built-in logging, checkpointing, eval

The model architecture is identical to my_nanochat/gpt.py — same attention,
same MLP, same shapes. Only the wrapping changes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput


# ═══════════════════════════════════════════════════════════════
# CONFIG — PretrainedConfig instead of @dataclass
# ═══════════════════════════════════════════════════════════════

class NanoChatConfig(PretrainedConfig):
    """
    Same fields as GPTConfig, but inherits from PretrainedConfig.

    This gives you:
      - config.save_pretrained("path") / NanoChatConfig.from_pretrained("path")
      - JSON serialization for free
      - Compatibility with AutoConfig.from_pretrained()

    HF requires model_type to register with Auto classes.
    """
    model_type = "nanochat"

    def __init__(
        self,
        sequence_len=2048,
        vocab_size=32768,
        n_layer=12,
        n_head=6,
        n_kv_head=6,
        n_embd=768,
        window_pattern="SSSL",
        **kwargs,  # HF passes extra fields (torch_dtype, etc.)
    ):
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.num_hidden_layers = n_layer  # HF alias — required for generate()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.window_pattern = window_pattern
        super().__init__(**kwargs)


# ═══════════════════════════════════════════════════════════════
# BUILDING BLOCKS — identical to my_nanochat/gpt.py
# ═══════════════════════════════════════════════════════════════

def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    """Same as my_nanochat — but simplified (no VE, no sliding window, no KV cache)."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head

        self.c_q = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.size()

        # Project and reshape into heads — (B, T, H, D)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # RoPE + QK norm
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # Transpose for attention: (B, H, T, D) — needed for scaled_dot_product_attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # PyTorch native scaled dot-product attention (fused kernel)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reassemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    """Same as my_nanochat — expand, relu², compress."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Same as my_nanochat — pre-norm residual pattern."""

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(norm(x), cos, sin)
        x = x + self.mlp(norm(x))
        return x


# ═══════════════════════════════════════════════════════════════
# THE MODEL — PreTrainedModel instead of nn.Module
# ═══════════════════════════════════════════════════════════════

class NanoChatModel(PreTrainedModel, GenerationMixin):
    """
    Same architecture as my_nanochat/gpt.py, wrapped in HF APIs.

    What PreTrainedModel gives you over nn.Module:
      - model.save_pretrained("path") / NanoChatModel.from_pretrained("path")
      - model.push_to_hub("username/model-name")
      - Automatic weight initialization via _init_weights()
      - Integration with HF Trainer for fine-tuning
      - model.generate() for text generation (beam search, sampling, etc.)
      - model.num_parameters() built-in
      - Gradient checkpointing support
    """
    config_class = NanoChatConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Precompute rotary embeddings
        head_dim = config.n_embd // config.n_head
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self._init_rope(config.sequence_len, head_dim)

        # Initialize weights (HF calls self.post_init() which calls _init_weights)
        self.post_init()

    def _init_rope(self, seq_len, head_dim, base=100000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        self.cos = cos[None, :, None, :]  # (1, T, 1, d_h/2)
        self.sin = sin[None, :, None, :]

    def _init_weights(self, module):
        """HF calls this for every module. Pattern: check type, init accordingly."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, **kwargs):
        """
        HF convention:
          - input_ids (not idx) — (B, T) token IDs
          - labels (not targets) — (B, T) for loss computation
          - Returns a dict/tuple with 'loss' and 'logits' keys

        This is what makes it compatible with HF Trainer and pipeline APIs.
        """
        B, T = input_ids.size()
        cos = self.cos[:, :T].to(input_ids.device)
        sin = self.sin[:, :T].to(input_ids.device)

        x = self.wte(input_ids)  # (B, T, C)
        x = norm(x)

        for block in self.blocks:
            x = block(x, cos, sin)
        x = norm(x)

        logits = self.lm_head(x)  # (B, T, V)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-1,
            )

        # HF expects CausalLMOutput for generate() and Trainer compatibility
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Required by GenerationMixin for model.generate() to work."""
        return {"input_ids": input_ids}


# ═══════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("NanoChatModel — Hugging Face Transformers API demo")
    print("=" * 60)

    # --- 1. Create model from config ---
    config = NanoChatConfig(
        sequence_len=256,   # small for demo
        vocab_size=32768,
        n_layer=4,          # small for demo
        n_head=4,
        n_kv_head=4,
        n_embd=256,         # small for demo
    )
    model = NanoChatModel(config)
    print(f"\n1. Model created: {model.num_parameters():,} parameters")

    # --- 2. Forward pass ---
    input_ids = torch.randint(0, config.vocab_size, (2, 64))  # batch=2, seq=64
    output = model(input_ids)
    print(f"2. Forward pass: logits shape = {output['logits'].shape}")

    # --- 3. Forward pass with labels (computes loss) ---
    labels = torch.randint(0, config.vocab_size, (2, 64))
    output = model(input_ids, labels=labels)
    print(f"3. Loss: {output['loss'].item():.4f}")

    # --- 4. Generate text (HF built-in!) ---
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(
        prompt,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.8,
        top_k=50,
    )
    print(f"4. Generated: {prompt.shape} → {generated.shape}")

    # --- 5. Save and reload ---
    model.save_pretrained("/tmp/nanochat-demo")
    loaded = NanoChatModel.from_pretrained("/tmp/nanochat-demo")
    print(f"5. Saved and reloaded: {loaded.num_parameters():,} parameters")

    # --- 6. Compare parameter counts ---
    print(f"\n--- Comparison ---")
    print(f"This model (n_layer=4, n_embd=256): {model.num_parameters():,}")

    full_config = NanoChatConfig()  # defaults: n_layer=12, n_embd=768
    full_model = NanoChatModel(full_config)
    print(f"Full model  (n_layer=12, n_embd=768): {full_model.num_parameters():,}")

    print("\nDone!")
