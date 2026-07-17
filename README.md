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

This project uses **`uv`** as its package and environment manager to guarantee consistency and prevent python version/GIL conflicts (especially on macOS).

### 1. Installation

Install dependencies and synchronize your local virtual environment:
```bash
uv sync
```

### 2. Prepare the Tokenizer and Corpus

To train the tokenizer, download and extract a TSV sentences dataset (e.g. from Tatoeba):
```bash
wget https://downloads.tatoeba.org/exports/per_language/kor/kor_sentences.tsv.bz2
bunzip2 ./kor_sentences.tsv.bz2
```

Then train the BPE tokenizer and generate the text corpus using `uv run`:
```bash
uv run python data.py ./kor_sentences.tsv
```
This produces:
- `corpus.txt`: Cleaned raw text.
- `tokenizer.json`: The trained BPE tokenizer configuration.

### 3. Train the Model

To train the `TinyLLM` model on the generated corpus:
```bash
uv run python train.py
```
This will train the model for 10 epochs (by default) and save the weights to `tiny_llm.pth`.

### 4. Generate Text

Once trained, generate sentences autoregressively:
```bash
uv run python generate.py
```

---

## Mechanistic Interpretability Dashboard

An interactive dashboard is available to inspect the model's inner workings (Self-Attention maps, Logit Lens, FFN activation scans, and gradient-based word saliency).

To start the dashboard, run:
```bash
uv run streamlit run interpretability.py
```
