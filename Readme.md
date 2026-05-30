<div align="center">

# 🧠 LLM Step by Step

**A GPT-style language model built from scratch — every component implemented by hand.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 📖 Overview

This repository is an educational, ground-up implementation of a GPT-style decoder-only transformer. No `transformers` library — every layer, every attention head, every training loop is written by hand. The goal is to understand exactly what happens inside a large language model.

### GPT-1

GPT-1 parametrs:

- vocab_size: 40 000
- seq_len: 512
- max_seq_len: 512

- emb_size: 768
- num_layers: 12
- num_heads: 12
- head_size: 64
- hidden_size: 3072 (4 * emb_size)
- dropout: 0.1

- num_epoch: 100
- lr: 2.5e-4
- batch_size: 64

### GPT-2

---

## 🏗️ Project architecture

```
llm_step_by_step/
│
├── tokenizer/
│   └── BPE/                   # Byte-Pair Encoding tokenizer
│
├── embeddings/
│   ├── TokenEmbeddings/       # Learned token embeddings
│   └── PositionalEmbeddings/  # Learned positional embeddings
│
├── attention/
│   ├── MaskedHeadAttention/   # Single causal attention head
│   └── MultiHeadAttention/    # Multi-head attention block
│
├── FFN/                       # Position-wise Feed-Forward Network
├── Decoder/                   # Transformer decoder block (MHA + FFN + LayerNorm)
├── GPT/                       # Full GPT model (embeddings + N decoder blocks + LM head)
│
└── train/
    ├── dataset.py             # PyTorch Dataset for next-token prediction
    ├── trainer.py             # Custom Trainer (warmup, cosine decay, AMP, grad accum)
    ├── train.py               # Entry point
    └── configs/
        ├── model.yaml
        ├── training.yaml
        ├── dataset.yaml
        ├── logging.yaml
        └── checkpointing.yaml
```

---

## ✨ Features

| Feature | Details |
|---|---|
| **Tokenizer** | BPE from scratch — `fit`, `encode`, `decode` |
| **Attention** | Masked (causal) multi-head attention with `register_buffer` mask |
| **Training** | Warmup + cosine LR decay, gradient clipping, gradient accumulation |
| **Mixed Precision** | `torch.autocast` + `GradScaler` (CUDA only) |
| **Generation** | `top-k`, `top-p` (nucleus), temperature sampling |
| **Checkpointing** | Resume from any epoch, optimizer & scheduler state preserved |
| **Serialization** | HuggingFace-style `save_pretrained` / `from_pretrained` |
| **Weights format** | `.safetensors` — fast, safe, framework-agnostic |
| **Config** | `OmegaConf` — split YAML configs, dot-access |
| **Logging** | W&B integration (optional) |

---

## ⚡ Quick Start

### 1. Install

```bash
git clone https://github.com/your-username/llm_step_by_step.git
cd llm_step_by_step

pip install -e .

# optional: wandb logging
pip install -e ".[logging]"
```

### 2. Configure

Edit the configs in `train/configs/` to match your setup:

```yaml
# train/configs/model.yaml
model:
  vocab_size: 50257
  max_seq_len: 512
  emb_size: 256
  num_heads: 8
  num_layers: 6
  ...

# train/configs/training.yaml
training:
  device: cuda          # cpu / cuda / mps
  num_epochs: 10
  learning_rate: 3.0e-4
  use_amp: true         # true for cuda only
  ...
```

### 3. Train

```bash
python -m train.train --config-dir train/configs

# resume from checkpoint
python -m train.train --config-dir train/configs --resume checkpoints/checkpoint_epoch_3
```

### 4. Generate

```python
import torch
from GPT import GPT
from tokenizer.BPE import BPE

tokenizer = BPE.from_pretrained("my_tokenizer")
model = GPT.from_pretrained("my_model", device="cuda")

x = torch.tensor([tokenizer.encode("Once upon a time")]).to("cuda")
output = model.generate(x, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.8)
print(tokenizer.decode(output[0].tolist()))
```

---

## 💾 Save & Load

```python
# after training
model.save_pretrained("my_model")
tokenizer.save_pretrained("my_tokenizer")

# load anywhere
model = GPT.from_pretrained("my_model", device="cuda")
tokenizer = BPE.from_pretrained("my_tokenizer")
```

Saved format on disk:

```
my_model/
├── config.json           ← architecture hyperparameters
└── model.safetensors     ← weights

my_tokenizer/
├── tokenizer_config.json ← vocab_size, tokenizer_class
└── tokenizer.json        ← vocab & id mappings
```

---

## 🛠️ Requirements

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.8.0 | Core deep learning framework |
| `safetensors` | ≥ 0.4.0 | Safe & fast weight serialization |
| `omegaconf` | ≥ 2.3.0 | Structured YAML config management |
| `pyyaml` | ≥ 6.0 | YAML parsing |
| `wandb` *(optional)* | ≥ 0.17.0 | Experiment tracking |

---

## 🗺️ Roadmap

- [ ] Dataset pipeline (raw text → tokenized → DataLoader)
- [ ] Pre-training run on a real corpus
- [ ] SFT (Supervised Fine-Tuning) support
- [ ] Evaluation: perplexity, generation quality
- [ ] Multi-GPU training (DDP)

---

<div align="center">
Built for learning. Every line intentional.
</div>
