---
aliases: [terms, definitions]
tags: [reference]
---

# Glossary

> Quick-reference for every term and symbol used in nanochat. Sorted by category.

## Tensor dimensions

| Symbol | Full name | Default value | Meaning |
|--------|-----------|---------------|---------|
| `B` | batch_size | varies | Number of sequences processed in parallel |
| `T` | sequence_len | 2048 | Tokens per sequence (context window) |
| `C` | n_embd | 768 | Embedding dimension (channels) |
| `V` | vocab_size | 32,768 | Number of tokens in the vocabulary |
| `N` | n_layer | 12 | Number of transformer blocks |
| `n_head` | query heads | 6 | Number of attention heads for queries |
| `n_kv_head` | KV heads | 6 | Number of heads for keys/values (GQA) |
| `d_h` | head_dim | 128 | Dimensions per head = `C // n_head` |
| `4C` | MLP hidden dim | 3072 | FFN intermediate size |

## Architecture terms

| Term | What it is | First appears |
|------|-----------|---------------|
| **wte** | Token embedding table. Integer ID → float vector. | [[05 - GPT Init]] |
| **lm_head** | Output projection. Float vector → vocab scores (logits). | [[05 - GPT Init]] |
| **RoPE** | Rotary Position Embedding. Encodes position via rotation matrices applied to Q and K. No learned parameters. | [[07 - Rotary Embeddings Deep Dive]] |
| **RMSNorm** | Root Mean Square Normalization. Like LayerNorm but no mean subtraction, no learned params in nanochat. | [[02 - Building Blocks]] |
| **GQA** | Group Query Attention. Fewer KV heads than Q heads — saves memory, same quality. | [[03 - CausalSelfAttention]] |
| **QK Norm** | Normalizing Q and K before computing attention scores. Prevents score explosion. | [[03 - CausalSelfAttention]] |
| **relu^2** | `F.relu(x).square()` — activation function. Sharper gating than GELU. | [[04 - MLP and Block]] |
| **Flash Attention** | Fused CUDA kernel that computes attention without materializing the full (T,T) score matrix. | [[03 - CausalSelfAttention]] |
| **Sliding window** | Per-layer limit on how far back a token can attend. Saves memory. | [[08 - Sliding Window Attention]] |
| **Value embeddings** | ResFormer-style: separate learned embeddings mixed into V at alternating layers. | [[05 - GPT Init]] |
| **resid_lambdas** | Per-layer scalar that scales the residual stream. Stronger early, weaker deep. | [[05 - GPT Init]] |
| **x0_lambdas** | Per-layer scalar that blends the initial embedding back in. | [[05 - GPT Init]] |
| **Smear gate** | Mixes previous token's embedding into current token. Cheap bigram info. | [[10 - Smear Gate]] |
| **Backout** | Subtracts the mid-layer residual before final norm. Removes low-level features. | [[09 - Forward Pass]] |
| **Softcap** | `softcap * tanh(logits / softcap)` — smoothly caps logits to a range. | [[09 - Forward Pass]] |
| **Muon** | Optimizer for matrix parameters. Uses polar decomposition for orthogonalization. | [[11 - Optimizer Setup]] |
| **AdamW** | Optimizer for embeddings and scalars. Adam with decoupled weight decay. | [[11 - Optimizer Setup]] |
| **KV cache** | During inference, stores computed K and V so they don't need recomputation. | [[12 - Generation]] |

## Math concepts you'll encounter

| Concept | Plain English | Where |
|---------|--------------|-------|
| **Dot product** | Multiply matching elements, sum them. Measures similarity. | Attention scores |
| **Matrix multiply** | Each output element is a dot product of a row and column. | Every `nn.Linear` |
| **Softmax** | `exp(x) / sum(exp(x))` — turns scores into probabilities summing to 1. | Attention, generation |
| **Cross-entropy** | `-log(probability of correct answer)` — the loss function. | Forward pass |
| **Broadcasting** | PyTorch auto-expands smaller tensors to match larger ones in element-wise ops. | Position embeddings, RoPE |
| **Residual connection** | `x = x + f(x)` — skip connection. Gradients flow directly through the `+`. | Every block |

---

*Back to [[Index]]*
