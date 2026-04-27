---
aliases: [optimizer, setup_optimizer, AdamW, Muon, param groups]
tags: [section, phase-11]
source: nanochat/gpt.py lines 374–415
---

# 11 — Optimizer Setup

<span class="phase-tag">SECTION 11</span> *AdamW + Muon — different optimizers for different parameter types*

> **Source:** `nanochat/gpt.py` lines 374–415
> **Copywork target:** ~42 lines

---

## What this code does

Different parameters benefit from different optimizers. nanochat splits all parameters into 7+ groups and assigns each group its own learning rate, optimizer type, betas, and weight decay.

- **AdamW** — for embeddings, lm_head, and scalar parameters
- **Muon** — for matrix parameters in attention and MLP (uses polar decomposition)

---

## The code — annotated

```python
def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2,
                    matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
    model_dim = self.config.n_embd
    ddp, rank, local_rank, world_size = get_dist_info()
```

### Parameter group separation

```python
    matrix_params = list(self.transformer.h.parameters())      # attention + MLP weights
    value_embeds_params = list(self.value_embeds.parameters())  # VE tables
    embedding_params = list(self.transformer.wte.parameters())  # token embeddings
    lm_head_params = list(self.lm_head.parameters())           # output head
    resid_params = [self.resid_lambdas]                        # per-layer scalar
    x0_params = [self.x0_lambdas]                              # per-layer scalar
    smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]

    # Sanity check: all params accounted for
    assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params)
        + len(lm_head_params) + len(value_embeds_params) + len(resid_params)
        + len(x0_params) + len(smear_params))
```

### LR scaling by model dimension

```python
    dmodel_lr_scale = (model_dim / 768) ** -0.5
```

At `model_dim=768`: scale = 1.0 (no change). At `model_dim=1536`: scale = 0.707 (lower LR for wider models). This implements the heuristic that wider models need smaller learning rates — proportional to `1/√width`.

### AdamW parameter groups

```python
    param_groups = [
        dict(kind='adamw', params=lm_head_params,
             lr=unembedding_lr * dmodel_lr_scale,
             betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
        dict(kind='adamw', params=embedding_params,
             lr=embedding_lr * dmodel_lr_scale,
             betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
        dict(kind='adamw', params=value_embeds_params,
             lr=embedding_lr * dmodel_lr_scale * 0.5,
             betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
        dict(kind='adamw', params=resid_params,
             lr=scalar_lr * 0.01,
             betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
        dict(kind='adamw', params=x0_params,
             lr=scalar_lr,
             betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=smear_params,
             lr=0.2,
             betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
    ]
```

| Group | LR (at 768) | beta1 | beta2 | Weight decay | Why different |
|-------|-------------|-------|-------|-------------|---------------|
| lm_head | 0.004 | 0.8 | 0.96 | 0.01 | Small — logits are sensitive |
| wte | 0.2 | 0.8 | 0.995 | 0.001 | Higher — embeddings need bigger updates |
| value_embeds | 0.1 | 0.8 | 0.995 | 0.01 | Half of embedding LR |
| resid_lambdas | 0.005 | 0.8 | 0.95 | 0.05 | Tiny — scalars move carefully |
| x0_lambdas | 0.5 | 0.96 | 0.95 | 0.0 | Higher beta1 — smoother momentum |
| smear params | 0.2 | 0.8 | 0.95 | 0.0 | Fixed LR, no decay |

### Muon parameter groups (matrix params)

```python
    for shape in sorted({p.shape for p in matrix_params}):
        group_params = [p for p in matrix_params if p.shape == shape]
        param_groups.append(dict(
            kind='muon', params=group_params, lr=matrix_lr,
            momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
        ))
```

Matrix parameters (c_q, c_k, c_v, c_proj, c_fc weights) are grouped **by shape** and given to Muon. Grouping by shape lets Muon stack them into one large tensor for efficient polar decomposition.

### Create the optimizer

```python
    Factory = DistMuonAdamW if ddp else MuonAdamW
    optimizer = Factory(param_groups)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer
```

`DistMuonAdamW` for multi-GPU (DDP), `MuonAdamW` for single GPU. Both handle the hybrid AdamW + Muon dispatch based on the `kind` field.

> [!keyinsight] Why two different optimizers?
> Embedding tables and scalars are updated element-wise — AdamW's per-parameter adaptive LR works well.
>
> Weight matrices (attention, MLP) benefit from **structured updates** that respect the geometry of the matrix. Muon uses polar decomposition to orthogonalize the update direction — it pushes weight matrices toward good conditioning. This is a 2024 technique from the modded-nanogpt community.

---

> [!copywork] Copywork checkpoint
> This section is more about understanding the *why* than memorizing exact LR values. For copywork, focus on:
>
> 1. The 7 parameter groups and how they're separated
> 2. The `dmodel_lr_scale` formula
> 3. The AdamW dict structure (kind, params, lr, betas, eps, weight_decay)
> 4. The Muon grouping-by-shape loop
> 5. The Factory dispatch (`DistMuonAdamW if ddp else MuonAdamW`)
>
> Don't try to memorize exact LR values — those are hyperparameters that get tuned.

---

*Previous: [[09 - Forward Pass]]*
*Next: [[12 - Generation]]*

*Back to [[Index]]*
