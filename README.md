# LLM from Scratch

A simple, educational implementation of a custom causal language model built from scratch in PyTorch, utilizing modern transformer architecture principles (similar to Llama).

## Features & Architecture

This repository contains all the building blocks to train and generate text from a custom language model:

- **Rotary Position Embeddings (RoPE)**: Implements complex frequency-based relative positional embeddings for query/key tensors.
- **RMSNorm**: Root Mean Square Layer Normalization used before attention and feed-forward blocks.
- **SwiGLU Activation**: Feed-forward networks using Swish-Gated Linear Units (SiLU-gated linear projections) instead of standard ReLU/GELU.
- **Custom BPE Tokenizer**: Helper scripts to train a Byte-Pair Encoding (BPE) tokenizer using the Hugging Face `tokenizers` library.
- **Top-K & Temperature Sampling**: Autoregressive text generation with customizable temperature scaling and top-k filtering.

---

## File Structure

- [model.py](file:///Users/elvinmarkmv/Development/Repositories/elvin-mark/llm-from-scratch/model.py): Core neural network components (`RMSNorm`, `Attention`, `FeedForward`, `TransformerBlock`, and `TinyLLM`).
- [data.py](file:///Users/elvinmarkmv/Development/Repositories/elvin-mark/llm-from-scratch/data.py): Tokenizer training pipeline and PyTorch dataset loader (`SentencesDataset`).
- [train.py](file:///Users/elvinmarkmv/Development/Repositories/elvin-mark/llm-from-scratch/train.py): Training loop configuration, optimizer setup, and checkpointing.
- [generate.py](file:///Users/elvinmarkmv/Development/Repositories/elvin-mark/llm-from-scratch/generate.py): Autoregressive text generator with Top-K and temperature scaling.
- [pyproject.toml](file:///Users/elvinmarkmv/Development/Repositories/elvin-mark/llm-from-scratch/pyproject.toml): Project metadata and dependencies.

---

## Getting Started

### 1. Installation

This project requires Python 3.13 or later. You can install the dependencies using your favorite package manager:

Using `uv` (recommended):
```bash
uv sync
```

Using standard `pip`:
```bash
pip install .
```

### 2. Prepare the Tokenizer and Corpus

To train the tokenizer, you need a dataset (such as a TSV file with format `id \t language \t sentence` from Tatoeba).

Example step to download and extract a Korean sentences dataset:
```bash
wget https://downloads.tatoeba.org/exports/per_language/kor/kor_sentences.tsv.bz2
bunzip2 ./kor_sentences.tsv.bz2
```

Then train the BPE tokenizer and create the `corpus.txt`:
```bash
python data.py ./kor_sentences.tsv
```
This produces:
- `corpus.txt`: Cleaned raw text.
- `tokenizer.json`: The trained BPE tokenizer configuration.

### 3. Train the Model

To train the `TinyLLM` model on the generated corpus:
```bash
python train.py
```
This will train the model for 10 epochs (by default) and output the weights to `tiny_llm.pth`.

### 4. Generate Text

Once trained, generate sentences autoregressively:
```bash
python generate.py
```
You can customize generation settings by importing and calling the `generate` function with custom parameters (e.g. `temperature`, `top_k`, `max_tokens`).
