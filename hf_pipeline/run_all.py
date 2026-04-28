"""
End-to-end LLM training pipeline using Hugging Face open source tools.

This script runs ALL stages of building an LLM — from raw text to a model
that generates text. Designed to run on a laptop (CPU/MPS) in ~5-10 minutes
with a tiny model. The architecture and pipeline are identical to what you'd
use at scale — just smaller numbers.

Stages:
  1. Download data (WikiText-2 — small, clean, public domain)
  2. Train a BPE tokenizer from scratch on that data
  3. Tokenize the dataset
  4. Build a GPT model (LLaMA-style, matching nanochat's design choices)
  5. Pretrain (custom loop with AdamW, ~500 steps)
  6. Evaluate (perplexity on validation set)
  7. Generate text (sample from the trained model)
  8. Save and reload (HF format — ready for Hub)

Usage:
  python -m hf_pipeline.run_all
"""

import os
import math
import time
import torch
import torch.nn.functional as F
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — one place to change everything
# ═══════════════════════════════════════════════════════════════

BASE_DIR = Path(os.path.expanduser("~/.cache/hf_pipeline"))
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Model size — tiny for laptop training
VOCAB_SIZE = 4096        # small vocab (real models use 32K-128K)
N_LAYER = 4              # 4 transformer blocks (real: 12-96)
N_HEAD = 4               # 4 attention heads (real: 6-128)
N_EMBD = 256             # 256-dim embeddings (real: 768-8192)
SEQ_LEN = 256            # 256-token context (real: 2048-128K)

# Training
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
NUM_STEPS = 500          # ~5 min on CPU
EVAL_EVERY = 50
GENERATE_EVERY = 100

# Device
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════
# STAGE 1 — DOWNLOAD DATA
# ═══════════════════════════════════════════════════════════════

def stage_1_download_data():
    """Download WikiText-2: ~2M tokens of clean Wikipedia text."""
    print_header("STAGE 1 — Download Data")

    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = "\n".join([t for t in dataset["train"]["text"] if len(t) > 0])
    val_text = "\n".join([t for t in dataset["validation"]["text"] if len(t) > 0])

    print(f"Train: {len(train_text):,} characters ({len(train_text.split()):,} words)")
    print(f"Val:   {len(val_text):,} characters ({len(val_text.split()):,} words)")

    return train_text, val_text


# ═══════════════════════════════════════════════════════════════
# STAGE 2 — TRAIN TOKENIZER
# ═══════════════════════════════════════════════════════════════

def stage_2_train_tokenizer(train_text):
    """Train a BPE tokenizer from scratch on the training data."""
    print_header("STAGE 2 — Train BPE Tokenizer")

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|pad|>", "<|bos|>", "<|eos|>"],
        min_frequency=2,
        show_progress=True,
    )

    # Train from an iterator (memory efficient)
    tokenizer.train_from_iterator(
        [train_text[i:i+10000] for i in range(0, len(train_text), 10000)],
        trainer=trainer,
    )

    # Save
    tok_path = str(BASE_DIR / "tokenizer.json")
    tokenizer.save(tok_path)

    # Test
    encoded = tokenizer.encode("The quick brown fox jumps over the lazy dog")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"Test encode: '{encoded.tokens[:10]}...'")
    print(f"Test IDs:    {encoded.ids[:10]}...")
    print(f"Avg chars/token: {len(train_text) / len(tokenizer.encode(train_text).ids):.1f}")

    return tokenizer


# ═══════════════════════════════════════════════════════════════
# STAGE 3 — TOKENIZE DATASET
# ═══════════════════════════════════════════════════════════════

def stage_3_tokenize(tokenizer, train_text, val_text):
    """Convert text to token ID tensors."""
    print_header("STAGE 3 — Tokenize Dataset")

    train_ids = tokenizer.encode(train_text).ids
    val_ids = tokenizer.encode(val_text).ids

    train_tensor = torch.tensor(train_ids, dtype=torch.long)
    val_tensor = torch.tensor(val_ids, dtype=torch.long)

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens:   {len(val_ids):,}")
    print(f"Train tensor: {train_tensor.shape}")

    return train_tensor, val_tensor


# ═══════════════════════════════════════════════════════════════
# STAGE 4 — BUILD MODEL
# ═══════════════════════════════════════════════════════════════

def stage_4_build_model():
    """Build a GPT model using LLaMA architecture (closest to nanochat)."""
    print_header("STAGE 4 — Build Model")

    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=N_EMBD,              # n_embd in nanochat
        num_hidden_layers=N_LAYER,       # n_layer in nanochat
        num_attention_heads=N_HEAD,      # n_head in nanochat
        num_key_value_heads=N_HEAD,      # n_kv_head in nanochat (GQA)
        intermediate_size=4 * N_EMBD,    # MLP 4x expansion
        max_position_embeddings=SEQ_LEN,
        rope_theta=100000,               # same base as nanochat
        rms_norm_eps=1e-6,
        use_cache=False,                 # disable KV cache during training
        tie_word_embeddings=False,       # untied, like nanochat
    )

    model = LlamaForCausalLM(config)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"Architecture: LLaMA (matches nanochat design choices)")
    print(f"Config: {N_LAYER} layers, {N_HEAD} heads, {N_EMBD} dim, {SEQ_LEN} ctx")
    print(f"Parameters: {n_params:,}")
    print(f"Device: {DEVICE}")

    model = model.to(DEVICE)
    return model, config


# ═══════════════════════════════════════════════════════════════
# STAGE 5 — PRETRAIN
# ═══════════════════════════════════════════════════════════════

def get_batch(data, batch_size, seq_len, device):
    """Random batch of sequences from the token stream."""
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y


def stage_5_pretrain(model, train_data, val_data):
    """Pretrain with a custom training loop (like nanochat, but simpler)."""
    print_header("STAGE 5 — Pretrain")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))

    model.train()
    t0 = time.time()
    best_val_loss = float("inf")

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
        if step % EVAL_EVERY == 0 or step == NUM_STEPS - 1:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(10):
                    vx, vy = get_batch(val_data, BATCH_SIZE, SEQ_LEN, DEVICE)
                    val_out = model(input_ids=vx, labels=vy)
                    val_losses.append(val_out.loss.item())
                val_loss = sum(val_losses) / len(val_losses)
            model.train()

            elapsed = time.time() - t0
            tokens_per_sec = (step + 1) * BATCH_SIZE * SEQ_LEN / elapsed if elapsed > 0 else 0
            print(f"Step {step:4d}/{NUM_STEPS} | train loss: {loss.item():.4f} | val loss: {val_loss:.4f} | {tokens_per_sec:,.0f} tok/s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best val loss: {best_val_loss:.4f} | Perplexity: {math.exp(best_val_loss):.1f}")

    return model


# ═══════════════════════════════════════════════════════════════
# STAGE 6 — EVALUATE
# ═══════════════════════════════════════════════════════════════

def stage_6_evaluate(model, val_data):
    """Full evaluation on validation set."""
    print_header("STAGE 6 — Evaluate")

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
    bpb = avg_loss / math.log(2)

    print(f"Validation loss:  {avg_loss:.4f}")
    print(f"Perplexity:       {perplexity:.1f}")
    print(f"Bits per token:   {bpb:.4f}")

    # For reference:
    print(f"\n--- Reference points ---")
    print(f"Random baseline:  loss = {math.log(VOCAB_SIZE):.4f}, perplexity = {VOCAB_SIZE}")
    print(f"GPT-2 on WikiText: perplexity ≈ 29")
    print(f"Your model:        perplexity = {perplexity:.1f}")

    return avg_loss


# ═══════════════════════════════════════════════════════════════
# STAGE 7 — GENERATE
# ═══════════════════════════════════════════════════════════════

def stage_7_generate(model, tokenizer):
    """Generate text samples from the trained model."""
    print_header("STAGE 7 — Generate Text")

    model.eval()
    prompts = [
        "The meaning of life is",
        "In the year 2025,",
        "The president of",
    ]

    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.8,
                top_k=50,
            )

        text = tokenizer.decode(generated[0].tolist())
        print(f"Prompt: '{prompt}'")
        print(f"Output: {text[:200]}")
        print()


# ═══════════════════════════════════════════════════════════════
# STAGE 8 — SAVE AND RELOAD
# ═══════════════════════════════════════════════════════════════

def stage_8_save(model, tokenizer):
    """Save in HF format — ready for Hub upload."""
    print_header("STAGE 8 — Save & Reload")

    save_path = str(BASE_DIR / "my-first-llm")

    # Save model
    model.save_pretrained(save_path)
    print(f"Model saved to: {save_path}")

    # Save tokenizer
    from transformers import PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        pad_token="<|pad|>",
    )
    hf_tokenizer.save_pretrained(save_path)
    print(f"Tokenizer saved to: {save_path}")

    # Reload to verify
    from transformers import LlamaForCausalLM, AutoTokenizer
    loaded_model = LlamaForCausalLM.from_pretrained(save_path).to(DEVICE)
    loaded_tokenizer = AutoTokenizer.from_pretrained(save_path)

    n_params_original = sum(p.numel() for p in model.parameters())
    n_params_loaded = sum(p.numel() for p in loaded_model.parameters())
    print(f"Original params: {n_params_original:,}")
    print(f"Loaded params:   {n_params_loaded:,}")
    print(f"Match: {n_params_original == n_params_loaded}")

    print(f"\nTo share on Hugging Face Hub:")
    print(f"  model.push_to_hub('arunma/my-first-llm')")
    print(f"  hf_tokenizer.push_to_hub('arunma/my-first-llm')")


# ═══════════════════════════════════════════════════════════════
# MAIN — run all stages
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_header("HF Pipeline — Build an LLM from Scratch")
    print(f"Device: {DEVICE}")
    print(f"Model:  {N_LAYER} layers, {N_HEAD} heads, {N_EMBD} dim")
    print(f"Data:   WikiText-2")
    print(f"Steps:  {NUM_STEPS}")

    # Run every stage
    train_text, val_text = stage_1_download_data()
    tokenizer = stage_2_train_tokenizer(train_text)
    train_data, val_data = stage_3_tokenize(tokenizer, train_text, val_text)
    model, config = stage_4_build_model()
    model = stage_5_pretrain(model, train_data, val_data)
    stage_6_evaluate(model, val_data)
    stage_7_generate(model, tokenizer)
    stage_8_save(model, tokenizer)

    print_header("DONE")
    print("You just built an LLM from scratch.")
    print("Same pipeline, same stages, same architecture as GPT / LLaMA.")
    print("The only difference at scale is bigger numbers.")
