---
aliases: [build LLM, from scratch, HF pipeline, open source LLM]
tags: [section, guide]
---

# 15 — Build Your Own LLM with Open Source Tools

<span class="phase-tag">GUIDE</span> *The complete recipe — from raw text to a working chatbot using Hugging Face and open source libraries*

---

## The 8 stages

Building an LLM from scratch requires 8 stages. nanochat does everything custom. In production, you'd use existing libraries for most of it.

```
 1. COLLECT DATA          → datasets, datatrove
 2. CLEAN DATA            → datatrove, text-dedup
 3. TRAIN TOKENIZER       → tokenizers (HF)
 4. TOKENIZE DATASET      → datasets .map()
 5. BUILD MODEL            → transformers (HF) or custom
 6. PRETRAIN              → transformers.Trainer or custom loop
 7. EVALUATE              → lm-evaluation-harness
 8. FINE-TUNE             → trl (HF), peft (LoRA)
 9. DEPLOY                → vllm, text-generation-inference
```

---

## Stage 1 — Collect data

An LLM needs a LOT of text. GPT-2 trained on ~40GB. GPT-3 on ~570GB. Modern models use 1-15 trillion tokens.

### Open source datasets

| Dataset | Size | What it is | Good for |
|---------|------|-----------|----------|
| **FineWeb-Edu** | 1.3T tokens | Filtered web text (educational) | General pretraining |
| **RedPajama v2** | 30T tokens | Web, books, code, Wikipedia | General pretraining |
| **The Stack v2** | 900B+ tokens | Source code from GitHub | Code models |
| **ClimbMix** | 400B tokens | NVIDIA curated mix | What nanochat uses |
| **Wikipedia** | ~4B tokens | Clean, factual | Small experiments |
| **C4** | 365B tokens | Cleaned Common Crawl | Classic pretraining set |

### Code

```python
from datasets import load_dataset

# Download a pretraining dataset from Hugging Face Hub
dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

# Streaming = don't download all at once (it's huge)
for sample in dataset:
    text = sample["text"]
    # process...
```

> [!keyinsight] You don't collect your own web crawl
> Unless you're Google/OpenAI-scale, you use existing curated datasets. The data cleaning step (next) is more important than the raw collection.

---

## Stage 2 — Clean data

Raw web text is full of duplicates, boilerplate, ads, encoding errors, and toxic content. Cleaning is where most of the quality comes from.

### Tools

| Tool | What it does |
|------|-------------|
| **datatrove** (HF) | Full pipeline: extract, filter, deduplicate, format |
| **text-dedup** | Fuzzy deduplication (MinHash, SimHash) |
| **fasttext** | Language identification and quality scoring |
| **trafilatura** | HTML → clean text extraction |

### What FineWeb-Edu does (the gold standard pipeline)

```
Raw Common Crawl (90TB compressed HTML)
    ↓ trafilatura — extract text from HTML
    ↓ fasttext — filter to English only
    ↓ MinHash dedup — remove near-duplicate documents
    ↓ Quality classifier — score educational value (0-5)
    ↓ Keep only score ≥ 3
    ↓ 1.3T tokens of high-quality educational text
```

### Minimal cleaning code

```python
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.dedup import MinhashDeduplication

# Build a cleaning pipeline
pipeline = [
    HuggingFaceDatasetReader("HuggingFaceFW/fineweb", streaming=True),
    LambdaFilter(lambda doc: len(doc.text) > 100),  # remove tiny docs
    LambdaFilter(lambda doc: doc.text.count("\n") / len(doc.text) < 0.3),  # remove list-heavy
    MinhashDeduplication(),
]
```

---

## Stage 3 — Train tokenizer

You need a tokenizer that converts text → integer IDs. Options:

| Approach | When to use | Tool |
|----------|------------|------|
| **Train your own BPE** | Custom domain, want control | `tokenizers` (HF) |
| **Reuse existing** | General English, quick start | `AutoTokenizer.from_pretrained()` |

### Training a BPE tokenizer from scratch

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Create a BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# Train on your corpus
trainer = trainers.BpeTrainer(
    vocab_size=32768,          # same as nanochat
    special_tokens=["<|bos|>", "<|eos|>", "<|pad|>"],
    min_frequency=2,
)

# Feed it text files or an iterator
tokenizer.train(files=["data/train.txt"], trainer=trainer)

# Save
tokenizer.save("my_tokenizer.json")

# Use
encoded = tokenizer.encode("Hello world")
print(encoded.ids)    # [15496, 995]
print(encoded.tokens) # ['Hello', ' world']
```

### Reusing an existing tokenizer

```python
from transformers import AutoTokenizer

# GPT-2 tokenizer (50257 tokens)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# LLaMA tokenizer (32000 tokens)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Use it
tokens = tokenizer("Hello world", return_tensors="pt")
print(tokens.input_ids)  # tensor([[15496, 995]])
```

---

## Stage 4 — Tokenize the dataset

Convert your entire training corpus from text to integer arrays on disk.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("my_tokenizer")
dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=2048)

# Tokenize entire dataset (parallelized across CPUs)
tokenized = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=8,
    remove_columns=dataset.column_names,
)

# Save to disk (Arrow format — fast, memory-mapped)
tokenized.save_to_disk("data/tokenized")
```

> [!keyinsight] This runs once before training
> Tokenization is CPU-bound and can take hours on large datasets. Do it once, save to disk, and load during training. Same as nanochat's `prepare.py` / `dataset.py` approach.

---

## Stage 5 — Build the model

Three options, from most control to least:

### Option A: Pure PyTorch (what you did in copywork)

```python
# Your my_nanochat/gpt.py — full control, you understand every line
from my_nanochat.gpt import GPT, GPTConfig
model = GPT(GPTConfig(n_layer=24))
```

### Option B: Hugging Face custom model (hf_nanochat)

```python
# Your hf_nanochat/model.py — same architecture, HF ecosystem compatibility
from hf_nanochat.model import NanoChatModel, NanoChatConfig
model = NanoChatModel(NanoChatConfig(n_layer=24))
```

### Option C: Use an existing HF architecture

```python
from transformers import AutoConfig, AutoModelForCausalLM

# Create a GPT-2-style model from scratch (random weights)
config = AutoConfig.from_pretrained("gpt2")
config.n_layer = 24
config.n_embd = 1024
config.vocab_size = 32768
model = AutoModelForCausalLM.from_config(config)

# Or use LLaMA architecture (closer to nanochat)
from transformers import LlamaConfig, LlamaForCausalLM
config = LlamaConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=6,
    num_key_value_heads=6,     # GQA!
    intermediate_size=3072,     # 4x expansion
    vocab_size=32768,
    max_position_embeddings=2048,
    rope_theta=100000,          # RoPE base
    rms_norm_eps=1e-6,
)
model = LlamaForCausalLM(config)
print(f"Parameters: {model.num_parameters():,}")
```

> [!keyinsight] LlamaConfig is the closest to nanochat
> LLaMA uses RoPE, RMSNorm, GQA, no bias — the same design decisions as nanochat. If you want to use an off-the-shelf architecture that matches what you learned, LLaMA is it.

---

## Stage 6 — Pretrain

### Option A: HF Trainer (simplest)

```python
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,     # effective batch = 16 * 4 * 8 GPUs = 512
    learning_rate=3e-4,
    warmup_steps=100,
    max_steps=50000,
    bf16=True,                         # mixed precision
    logging_steps=10,
    save_steps=1000,
    dataloader_num_workers=4,
)

# Data collator handles padding and creating labels (shifted by 1)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
```

### Option B: Custom training loop (more control, like nanochat)

```python
import torch
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# DDP setup (multi-GPU)
init_process_group("nccl")
model = DDP(model.cuda(), device_ids=[local_rank])

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))

for step in range(max_steps):
    batch = next(dataloader)
    input_ids = batch["input_ids"].cuda()
    labels = input_ids.clone()

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    if step % 100 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")
```

### Option C: Use a training framework

| Framework | What it gives you | Complexity |
|-----------|------------------|-----------|
| **HF Trainer** | Simple API, logging, checkpointing, eval | Low |
| **PyTorch Lightning** | Hooks-based, multi-GPU, mixed precision | Medium |
| **DeepSpeed** | ZeRO optimization, offloading, 3D parallelism | High |
| **FSDP** (PyTorch native) | Fully Sharded Data Parallel | Medium |
| **Megatron-LM** (NVIDIA) | Tensor/pipeline parallelism for huge models | Very high |
| **nanotron** (HF) | Optimized pretraining, 3D parallelism | Medium |

For a GPT-2 scale model (≤1B params) on 8 GPUs: **HF Trainer or a custom loop with DDP** is sufficient. You don't need DeepSpeed or Megatron until you hit multi-billion parameters.

---

## Stage 7 — Evaluate

### lm-evaluation-harness (the standard)

```bash
pip install lm-eval

# Evaluate on standard benchmarks
lm_eval --model hf \
    --model_args pretrained=./checkpoints \
    --tasks arc_easy,hellaswag,mmlu,winogrande \
    --batch_size 16
```

This is what produces CORE-like scores. It runs few-shot evaluation on standard tasks.

### Custom evaluation

```python
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

model = HFLM(pretrained="./checkpoints")
results = evaluator.simple_evaluate(
    model=model,
    tasks=["arc_easy", "hellaswag", "mmlu"],
    batch_size=16,
)
print(results["results"])
```

### Perplexity / BPB evaluation

```python
# Quick validation loss on held-out data
model.eval()
total_loss = 0
with torch.no_grad():
    for batch in val_dataloader:
        outputs = model(batch["input_ids"].cuda(), labels=batch["input_ids"].cuda())
        total_loss += outputs.loss.item()

avg_loss = total_loss / len(val_dataloader)
perplexity = math.exp(avg_loss)
bpb = avg_loss / math.log(2)  # bits per byte (approx)
print(f"Perplexity: {perplexity:.2f}, BPB: {bpb:.4f}")
```

---

## Stage 8 — Fine-tune (SFT + optional RLHF)

### Supervised Fine-Tuning with TRL

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Load a conversation dataset
dataset = load_dataset("HuggingFaceTB/smoltalk", split="train")

# SFT config
sft_config = SFTConfig(
    output_dir="checkpoints/sft",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,          # lower LR for fine-tuning
    num_train_epochs=3,
    bf16=True,
    max_seq_length=2048,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

### LoRA — fine-tune 1% of parameters (PEFT)

For large models, full fine-tuning is expensive. LoRA freezes the original weights and adds small trainable adapters:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                      # rank of the adapter
    lora_alpha=32,
    target_modules=["c_q", "c_k", "c_v", "c_proj"],  # which layers to adapt
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable: 2.4M (0.9%), total: 286M
```

### RLHF with TRL

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Wrap model with value head for RL
model = AutoModelForCausalLMWithValueHead.from_pretrained("checkpoints/sft")

ppo_config = PPOConfig(
    batch_size=16,
    learning_rate=1.41e-5,
)

# Requires a reward model — often another LLM trained on human preferences
ppo_trainer = PPOTrainer(
    model=model,
    config=ppo_config,
    tokenizer=tokenizer,
)

# Training loop: generate → score → update
for batch in dataloader:
    queries = batch["input_ids"]
    responses = ppo_trainer.generate(queries, max_new_tokens=128)
    rewards = reward_model(queries, responses)  # your reward signal
    ppo_trainer.step(queries, responses, rewards)
```

---

## Stage 9 — Deploy

### vLLM (fastest open source inference)

```bash
pip install vllm

# Serve your model as an OpenAI-compatible API
python -m vllm.entrypoints.openai.api_server \
    --model ./checkpoints/sft \
    --port 8000
```

```python
# Call it like OpenAI
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="./checkpoints/sft",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)
```

### HF Text Generation Inference (TGI)

```bash
docker run --gpus all -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id ./checkpoints/sft
```

### Push to Hugging Face Hub

```python
model.push_to_hub("arunma/my-nanochat")
tokenizer.push_to_hub("arunma/my-nanochat")

# Anyone can now use it:
# model = AutoModelForCausalLM.from_pretrained("arunma/my-nanochat")
```

---

## The complete tool stack — one table

| Stage | nanochat does it... | Open source alternative |
|-------|--------------------|-----------------------|
| Data collection | `nanochat/dataset.py` (downloads ClimbMix) | `datasets` library, datatrove |
| Data cleaning | Relies on pre-cleaned dataset | datatrove, text-dedup, trafilatura |
| Tokenizer | `nanochat/tokenizer.py` (custom BPE) | `tokenizers` (HF), sentencepiece |
| Tokenize corpus | `nanochat/dataloader.py` | `datasets.map()` |
| Model architecture | `nanochat/gpt.py` (512 lines) | `transformers` (LlamaForCausalLM), or custom |
| Pretraining | `scripts/base_train.py` (custom loop) | `Trainer`, Lightning, nanotron |
| Evaluation | `nanochat/core_eval.py` (custom) | `lm-evaluation-harness` |
| SFT | `scripts/chat_sft.py` | `trl.SFTTrainer` |
| RLHF | `scripts/chat_rl.py` | `trl.PPOTrainer` |
| Inference | `nanochat/engine.py` + web UI | vLLM, TGI, `model.generate()` |
| Sharing | N/A | `model.push_to_hub()` |

---

## Realistic cost and time estimates

| Scale | Params | Data | Hardware | Time | Cost |
|-------|--------|------|----------|------|------|
| **Toy** (learn) | 10-100M | Wikipedia | 1 GPU (A100/H100) | 1-4 hours | $5-20 |
| **GPT-2 grade** | 100M-1B | FineWeb 100B | 8 × H100 | 2-8 hours | $30-200 |
| **LLaMA-7B grade** | 7B | 1-2T tokens | 64-128 × H100 | 1-2 weeks | $50K-100K |
| **LLaMA-70B grade** | 70B | 2-15T tokens | 512-2048 × H100 | 2-4 weeks | $500K-2M |

Your nanochat speedrun sits at row 2 — achievable by an individual for ~$50.

---

*Back to [[Index]]*
