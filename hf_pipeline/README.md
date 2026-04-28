# HF Pipeline — Build an LLM from Scratch with Hugging Face

A complete end-to-end LLM training pipeline using open source tools. Same architecture as nanochat, built with Hugging Face libraries.

## Quick start

```bash
# From the project root:
python -m hf_pipeline.run_all
```

Runs all 8 stages on your laptop (CPU/MPS) in ~30 seconds with a tiny model.

## What it does

| Stage | What | Tool | Time |
|-------|------|------|------|
| 1. Download data | WikiText-2 (~11M chars) | `datasets` | 2s |
| 2. Train tokenizer | BPE, vocab=4096 | `tokenizers` | 1s |
| 3. Tokenize corpus | Text → integer tensors | `tokenizers` | 1s |
| 4. Build model | LLaMA architecture (4 layers, 256 dim) | `transformers` | <1s |
| 5. Pretrain | 500 steps, AdamW | PyTorch | ~20s |
| 6. Evaluate | Perplexity, BPB | PyTorch | 2s |
| 7. Generate | Sample text from trained model | `transformers` | 2s |
| 8. Save/reload | HF format, ready for Hub | `transformers` | 1s |

## Where to learn

The vault has two tracks. **Read these in order for the HF track:**

### Copywork sections (write code from memory)

1. **`vault/sections/hf-pipeline/HF-01 - Data and Tokenizer.md`**
   - `datasets.load_dataset()` — downloading and streaming data
   - `tokenizers` library — training BPE from scratch (models, trainers, pre_tokenizers)
   - Tokenizing a corpus and creating PyTorch tensors
   - The `get_batch()` function (random sequences from the token stream)

2. **`vault/sections/hf-pipeline/HF-02 - Model and Training.md`**
   - `LlamaConfig` — every field mapped to nanochat's `GPTConfig`
   - `LlamaForCausalLM` — what it builds for you (and what nanochat features it lacks)
   - The 5-line training loop: forward → backward → clip → step → zero
   - HF Trainer alternative (zero-loop approach)

3. **`vault/sections/hf-pipeline/HF-03 - Evaluate Generate Save.md`**
   - Perplexity and bits-per-byte calculation
   - `model.generate()` — all sampling parameters (temperature, top-k, top-p, beam search)
   - `save_pretrained()` / `from_pretrained()` — what files it writes
   - `PreTrainedTokenizerFast` — wrapping your raw tokenizer for HF compatibility
   - `push_to_hub()` — sharing on Hugging Face Hub

### Deep reference (read, don't copywork)

4. **`vault/sections/hf-pipeline/HF-04 - Deep Dive Internals.md`** — THE KEY DOCUMENT
   - Every nanochat component mapped to its HF equivalent
   - Side-by-side code comparisons with shape traces
   - What HF wraps, what it hides, what constraints it imposes
   - Covers all 10 components: config, embeddings, RoPE, attention, MLP, norm, block, full model, training, generation

### Supporting reference

- `vault/sections/hf-pipeline/14 - Hugging Face API.md` — nanochat→HF class mapping overview
- `vault/sections/hf-pipeline/15 - Build Your Own LLM.md` — full 9-stage guide with all tool options
- `vault/sections/hf-pipeline/16 - HF Pipeline End to End.md` — cost analysis, 6× FLOPs derivation

## Copywork flow

```
vault/sections/hf-pipeline/HF-01...   ← read this
                    ↓
vault/copywork/hf/01_data_tokenizer.py ← write from memory
                    ↓
hf_pipeline/run_all.py                 ← diff against this
```

Same method as the nanochat track: read → close → write → diff → fix.

## File structure

```
hf_pipeline/
├── __init__.py
├── run_all.py        ← the complete working pipeline (all 8 stages)
└── README.md         ← you are here

hf_nanochat/
├── __init__.py
└── model.py          ← nanochat architecture wrapped in HF APIs
                        (PreTrainedModel, PretrainedConfig, GenerationMixin)
```

## Key differences from nanochat track

| Aspect | nanochat track | HF track |
|--------|---------------|----------|
| Model code | Write `gpt.py` from scratch (512 lines) | Use `LlamaForCausalLM` off the shelf |
| Focus | Architecture internals | Pipeline + ecosystem integration |
| Tokenizer | Understand BPE algorithm | Use `tokenizers` library |
| Training | Understand the loop | Use the loop (or Trainer) |
| Generation | Write manual loop with yield | Use `model.generate()` |
| Saving | `torch.save()` | `save_pretrained()` → Hub |

## What to do tomorrow

1. Open Obsidian, read **HF-04 (Deep Dive Internals)** first — this connects everything you learned in the nanochat track to HF's abstractions
2. Then read **HF-01** and do the copywork (data + tokenizer)
3. Then **HF-02** (model + training)
4. Then **HF-03** (eval + generate + save)
5. Run `python -m hf_pipeline.run_all` to see it all work

## Cost reference

| Scale | Cost | What you get |
|-------|------|-------------|
| This demo (6M params, 500 steps) | $0 | Gibberish text (learning exercise) |
| nanochat speedrun (300M, 10B tokens) | ~$50 | GPT-2 grade chatbot |
| LLaMA-7B equivalent | ~$75K | Production-quality model |

The cost is `6 × params × tokens` FLOPs regardless of which library you use.
