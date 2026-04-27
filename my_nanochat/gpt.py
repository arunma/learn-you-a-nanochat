"""
My nanochat GPT — built through copywork.

This file will grow section by section as I learn each piece.
When complete, it should be a drop-in replacement for nanochat/gpt.py.

Support modules (tokenizer, dataloader, optim, etc.) are imported
from Karpathy's nanochat/ package — no need to rewrite those.
"""

# === Section 01: Config and Imports ===
# (start writing here after completing the copywork for Section 01)
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.flash_attention import flash_attn


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_index, n_layer):
    return layer_index % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):  # RoPE
    # Needs further reading. It doesn't loop but multiplies the source matrix element-wise across all 64 dimension pairs (d). But the exact math doesn't click right.
    assert x.ndim == 4  # B,T,n_head, d_h -> B,T,6,128
    d = x.shape[3] // 2  # 64
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin  # rotates first half - I am going to accept it at face value.
    y2 = x1 * (-sin) + x2 * cos  # rotates second half
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head  # 6
        self.n_kv_head = config.n_kv_head  # 6 KV heads (no GQA at this point. Behaves like MHA)
        self.n_embd = config.n_embd  # 768
        self.head_dim = self.n_embd // self.n_head  # 768/6=128

        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)  # 768*768
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)  # Q,K,V must be the same dimension
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)

        # Output projection - should also be 768*768 (C*C)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)

        self.ve_gate_channels = 12
        # 12*6 - for every alternative layer
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)  # B,T,H,D - avoids transposing while attention calc
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)  # B,T,C -> B,T,H,D
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        # q,k = B,T,6,128
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * 1.2  # Revisit: Need to understand more.  Can't really follow why 1.2 at this point.
        k = k * 1.2

        if kv_cache is None:
            # During training, no caching to enable maximum parallelism. Maximum throughput
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference - Use KV cache which is already persisted while going through T. Minimum latency
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            # Just a fancy optimized version of Q@K.T. seq_lens is just the current token (or rather last token index) in T
            y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, cache_seqlens=kv_cache.cache_seqlens, causal=True,
                                                   window_size=window_size)  # B,T,6,128 - attention per head

            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)  # Increments seqlens by the size of the prompt or context window

        y = y.contiguous().view(B, T, -1)  # B,T,768 - consolidate attention scores of all heads
        y = self.c_proj(y)  # B,T,768
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        # expand and contract - 768*3072 -> 3072*768
        x = self.c_fc(x)
        # Relu^2 - Zeroes negatives with relu and square positives with square().
        # Looks like modern LLMs use some Gated LU, where value stream is considered (similar to ve tensor)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # This is the residual bit where the derivate gets the `1+`
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x
