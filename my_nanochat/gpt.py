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


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to

        # Basically,
        # 1. convert vocab to embedding using wte (32768,768)
        # 2. Do all the processing with h - 12 * Block that operates on B,T,768
        # 3. convert embedding to vocab using lm_head (768,32768)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)])  # Construct all hidden layers/transformer blocks
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)

        # # Research: This is interesting. Scales the accumulated residual. Controls how much does the residual carry across layers
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        # Research: Same as above but this is for embedding. How much does the original embedding carry through the layers.
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        # Research: Mix previous token embedding into current token. Like a bigram? Reads first 24 channels of the input
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))

        # Research. No freaking idea
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))

        head_dim = config.n_embd // config.n_head  # 768/6=128
        kv_dim = config.n_kv_head * head_dim  # 768
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, kv_dim)  # 32768,768
            for i in range(config.n_layer) if has_ve(i, config.n_layer)  # only for half the layers
        })  # Each alternating layer has 32768*768

        # Research - no idea how the math works
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3 ** 0.5 * n_embd ** -0.5  # -0.0625 to +0.0625 for all attn tensors

        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # magic constants, magic constants everywhere !
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))  # Apparently this goes from 1.15 to 1.05
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (
                    0.15 * i / max(n_layer - 1, 1))  # This one goes from 0.20 to 0.05 - No idea why these specific constants

        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    # Research. A LOT. Very little idea on what this does. I get that for positions 1..128, we are rotating by a number.
    # The proximity of the token gets a bigger value - the 1/ 100000^0/128 =1.0 for the first token and progressively smaller as the distance is higher
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device

        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))

        t = torch.arange(seq_len, dtype=torch.float32, device=device)

        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        # WTF. Why?
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()

        long_window = config.sequence_len  # 2048
        short_window = -(-long_window // 4 // 128) * 128  # 512

        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0)
        }

        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])

        window_sizes[-1] = (long_window, 0)  # This must be already calculated. Just reconfirming I think
        return window_sizes


    def forward (self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B,T=idx.size()
        T0= 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[,:T0:T0+T], self.sin[:, T0:T0+T]


