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


def apply_rotary_emb(x, cos, sin): #RoPE
    # Needs further reading. It doesn't loop but multiplies the source matrix element-wise across all 64 dimension pairs (d). But the exact math doesn't click right.
    assert x.ndim == 4  # B,T,n_head, d_h -> B,T,6,128
    d = x.shape[3] // 2  # 64
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin  # rotates first half - I am going to accept it at face value.
    y2 = x1 * (-sin) + x2 * cos  # rotates second half
    return torch.cat([y1, y2], 3)


