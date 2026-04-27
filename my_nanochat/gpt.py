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
