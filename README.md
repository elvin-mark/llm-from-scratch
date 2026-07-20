# LLM from Scratch

A simple, educational implementation of a custom causal language model built from scratch in PyTorch, utilizing modern transformer architecture principles (similar to Llama).

## Features & Architecture

This repository contains all the building blocks to train and generate text from a custom language model:

- **Rotary Position Embeddings (RoPE)**: Implements complex frequency-based relative positional embeddings for query/key tensors.
- **RMSNorm**: Root Mean Square Layer Normalization used before attention and feed-forward blocks.
- **SwiGLU Activation**: Feed-forward networks using Swish-Gated Linear Units (SiLU-gated linear projections) instead of standard ReLU/GELU.
- **Custom BPE Tokenizer**: Helper scripts to train a Byte-Pair Encoding (BPE) tokenizer using the Hugging Face `tokenizers` library, or an educational from-scratch pure Python BPE implementation.
- **Top-K & Temperature Sampling**: Autoregressive text generation with customizable temperature scaling and top-k filtering.
- **Standalone C & CUDA Inference**: Highly optimized, zero-allocation standalone inference engines written in C and CUDA, supporting OpenBLAS and Int8 Quantization.

---

## File Structure

- [model.py](model.py): Core neural network components (`RMSNorm`, `Attention`, `FeedForward`, `TransformerBlock`, and `TinyLLM`).
- [data.py](data.py): Tokenizer training pipeline and PyTorch dataset loader (`SentencesDataset`).
- [train.py](train.py): Training loop configuration, optimizer setup, and checkpointing.
- [generate.py](generate.py): Autoregressive text generator with Top-K and temperature scaling.
- [interpretability.py](interpretability.py): Streamlit dashboard for mechanistic interpretability.
- [export_onnx.py](export_onnx.py): Script to export PyTorch weights into optimized ONNX/Int8 formats.
- [tokenizer.py](tokenizer.py): Educational, from-scratch pure Python implementation of a BPE tokenizer.
- [c/](c/): Standalone inference engine implementations in pure C, OpenBLAS, CUDA, and Quantized Int8.
- [ui/](ui/): Frontend browser application utilizing ONNX Runtime Web.
- [docs/](docs/): Extensive documentation detailing the LLaMA-based architecture, tokenization, and training flow.

---

## Extensive Documentation

If you are interested in exactly how the mathematics and auto-regressive flows of this model work, we provide detailed markdown documentation with Mermaid flowchart diagrams:

- [Architecture Breakdown](docs/architecture.md): Deep dive into Pre-RMSNorm, RoPE, and SwiGLU FFNs.
- [Mathematical Foundations](docs/math.md): The theoretical mathematical formulas defining the entire forward pass.
- [Training Pipeline](docs/training.md): Overview of dataset ingestion, hyperparameter choices, and the CrossEntropy backward pass loop.
- [Tokenizer Architecture](docs/tokenizer.md): Explanation of the Byte-Pair Encoding (BPE) training and inference algorithms.
- [C Inference Architecture](c/ARCH.md): Explanation of the memory-mapped C inference engines and dynamic Int8 quantization.

---

## Getting Started

This project uses **`uv`** as its package and environment manager to guarantee consistency and prevent python version/GIL conflicts.

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
This produces `corpus.txt` and `tokenizer.json`.

*(Optional: You can train the tokenizer using our educational pure-Python BPE algorithm from scratch by appending `--scratch-tokenizer`)*

### 3. Train the Model

To train the `TinyLLM` model on the generated corpus:
```bash
uv run python train.py
```
This will train the model and save the weights to `tiny_llm.pth`.

### 4. Generate Text

Once trained, generate sentences autoregressively:
```bash
uv run python generate.py
```

*(Optional: Generate text using the pure-Python educational tokenizer by appending `--scratch-tokenizer`)*

### 5. NumPy-Only Inference (Pure NumPy)

For a hyper-compact, lightweight inference option that bypasses PyTorch dependencies and runs entirely on `numpy`, you can use `inference.py`:

```bash
uv run python inference.py --weights tiny_llm.pth --vocab-size 4000 --prompt "안녕하세요" --tokens 40
```

This script runs the entire transformer forward pass (Attention, RoPE, SwiGLU, RMSNorm) using bare-metal NumPy operations. It supports loading both `.pth` (PyTorch) checkpoints and `.npz` (NumPy compressed) weight packages.

---

## Standalone Native Inference (C & CUDA)

You can run the model entirely standalone without Python or PyTorch using our memory-mapped native inference engines located in the `c/` directory.

1. **Export the Weights**:
   ```bash
   cd c
   uv run python export.py
   ```
2. **Compile and Run**:
   Choose your hardware target:
   ```bash
   # Standard CPU (Naive Loops)
   make run
   
   # CPU with OpenBLAS Acceleration
   make run USE_BLAS=1
   
   # GPU with Custom CUDA Kernels
   make run_cu
   ```

3. **Int8 Dynamic Quantization**:
   For severe memory footprint reduction, export the model using row-wise Int8 quantization and run the dynamic quantizer:
   ```bash
   uv run python export_q8.py
   make runq
   ./runq
   ```
   
---

## Serverless Web UI Inference (ONNX)

You can run this model entirely inside any modern web browser utilizing **ONNX Runtime Web** (WebGL/WASM acceleration)—no backend API required!

1. **Export the Model**:
   Convert the `tiny_llm.pth` weights to `.onnx` and apply Int8 quantization:
   ```bash
   # Make sure onnx and onnxruntime are installed
   uv pip install onnx onnxruntime onnxscript
   uv run python export_onnx.py --quantize
   ```
2. **Launch the UI**:
   Since browsers block fetching local binary files, serve the directory locally:
   ```bash
   python3 -m http.server 8000
   ```
   Then open [http://localhost:8000/ui/](http://localhost:8000/ui/) to chat with your custom LLM right in the browser!

---

## Mechanistic Interpretability Dashboard

An interactive dashboard is available to inspect the model's inner workings (Self-Attention maps, Logit Lens, FFN activation scans, and gradient-based word saliency).

To start the dashboard, run:
```bash
uv run streamlit run interpretability.py
```
