# LLM from Scratch

A simple, educational implementation of a custom causal language model built from scratch in PyTorch, utilizing modern transformer architecture principles (similar to Llama).

🌐 **Live Web Demo**: [https://llm-from-scratch-edu.web.app/](https://llm-from-scratch-edu.web.app/)

## Features & Architecture

This repository contains all the building blocks to train and generate text from custom language models:

- **Multiple Architecture Models**:
  - **`TinyLLM`**: Standard dense Llama-style model (Multi-Head Attention + Dense SwiGLU).
  - **`MoELLM`**: Advanced Mixture-of-Experts model (Grouped Query Attention + MoE SwiGLU with Top-K Router).
- **Grouped Query Attention (GQA)**: Reduces memory footprint by sharing Key/Value heads across Query head groups.
- **Mixture-of-Experts (MoE)**: Gated routing mechanism that dispatches tokens dynamically to top-K expert MLPs.
- **Rotary Position Embeddings (RoPE)**: Implements complex frequency-based relative positional embeddings for query/key tensors.
- **RMSNorm**: Root Mean Square Layer Normalization used before attention and feed-forward blocks.
- **SwiGLU Activation**: Feed-forward networks using Swish-Gated Linear Units (SiLU-gated linear projections).
- **Custom BPE Tokenizer**: Helper scripts to train a Byte-Pair Encoding (BPE) tokenizer using Hugging Face `tokenizers` or an educational pure-Python BPE implementation.
- **Top-K & Temperature Sampling**: Autoregressive text generation with customizable temperature scaling and top-k filtering.
- **Standalone C & CUDA Engines**: Highly optimized, zero-allocation standalone inference and autograd training engines written in pure C and CUDA.
- **Mechanistic Interpretability Dashboard**: Streamlit app for attention map visualization, logit lens, FFN activations, and head ablation.
- **In-Browser WebAssembly / WebGL UI**: Deployed serverless Web UI running ONNX Runtime Web.

---

## File Structure

```text
llm-from-scratch/
├── src/
│   └── tiny_llm/             # Modular package
│       ├── configs.py        # Configuration dataclasses (TinyLLMConfig, MoELLMConfig)
│       ├── modules/          # Primitives (RMSNorm, RoPE, MHA, GQA, SwiGLU, MoE)
│       ├── models/           # Models (TinyLLM, MoELLM)
│       ├── tokenizer.py      # BPE Tokenizer implementation
│       └── data.py           # Dataset loaders
├── scripts/                  # High-level entrypoints (train, generate, inference, interpretability)
├── tools/
│   └── export/               # Export utilities (export_c, export_q8, export_onnx)
├── c/                        # Bare-metal C & CUDA engines (run.c, train.c, run.cu, train.cu)
├── ui/                       # Web interface & deployment configs
├── tests/                    # Unit testing suite (test_model.py, test_data.py, test_moe.py)
├── docs/                     # Documentation & architecture guides
├── data/                     # Training datasets (corpus.txt)
└── checkpoints/              # Model weights & tokenizer files
```

---

## Extensive Documentation

If you are interested in exactly how the mathematics and auto-regressive flows of this model work, we provide detailed markdown documentation with Mermaid flowchart diagrams:

- [Architecture Breakdown](docs/architecture.md): Deep dive into Pre-RMSNorm, RoPE, and SwiGLU FFNs.
- [Advanced Architecture (GQA & MoE)](docs/moe_gqa.md): Theoretical breakdown of Grouped Query Attention and Mixture-of-Experts routing.
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

### 2. Run the Unit Test Suite

Verify that all model modules, RoPE dot-product invariants, causal masking, and dataset loaders pass:
```bash
uv run pytest
```

### 3. Prepare the Tokenizer and Corpus

To train the tokenizer, download and extract a TSV sentences dataset (e.g. from Tatoeba):
```bash
wget https://downloads.tatoeba.org/exports/per_language/kor/kor_sentences.tsv.bz2
bunzip2 ./kor_sentences.tsv.bz2
```

Then train the BPE tokenizer and generate the text corpus using `uv run`:
```bash
uv run python scripts/prepare_data.py ./kor_sentences.tsv
```
This produces `data/corpus.txt` and `checkpoints/tokenizer.json`.

*(Optional: You can train the tokenizer using our educational pure-Python BPE algorithm from scratch by appending `--scratch-tokenizer`)*

### 4. Train the Model

To train the `TinyLLM` model on the generated corpus:
```bash
uv run python scripts/train.py
```
This will train the model and save the weights to `checkpoints/tiny_llm.pth`.

### 5. Generate Text

Once trained, generate sentences autoregressively:
```bash
uv run python scripts/generate.py
```

*(Optional: Generate text using the pure-Python educational tokenizer by appending `--scratch-tokenizer`)*

### 6. NumPy-Only Inference (Pure NumPy)

For a hyper-compact, lightweight inference option that bypasses PyTorch dependencies and runs entirely on `numpy`, you can use `scripts/inference.py`:

```bash
uv run python scripts/inference.py --weights checkpoints/tiny_llm.pth --vocab-size 4000 --prompt "안녕하세요" --tokens 40
```

This script runs the entire transformer forward pass (Attention, RoPE, SwiGLU, RMSNorm) using bare-metal NumPy operations. It supports loading both `.pth` (PyTorch) checkpoints and `.npz` (NumPy compressed) weight packages.

---

## Standalone Native C & CUDA Engines

You can run both **inference** and **training from scratch** entirely standalone without Python or PyTorch using our low-level C and CUDA engines in the `c/` directory.

### 1. C & CUDA Inference

1. **Export the Weights**:
   ```bash
   uv run python tools/export/export_c.py
   ```
   This generates `c/model.bin` and `c/vocab.bin`.

2. **Compile and Run**:
   ```bash
   cd c
   # Standard CPU (Naive Loops)
   make run && ./run
   
   # CPU with OpenBLAS Acceleration
   make run USE_BLAS=1 && ./run
   
   # GPU with Custom CUDA Kernels
   make run_cu && ./run_cu
   ```

3. **Int8 Dynamic Quantization**:
   For severe memory footprint reduction, export the model using row-wise Int8 quantization:
   ```bash
   uv run python tools/export/export_q8.py
   cd c && make runq && ./runq
   ```

### 2. C & CUDA Training (Autograd from Scratch)

You can also train the model directly in C or CUDA:
```bash
cd c
# Train on CPU (with OpenMP multi-threading and optional OpenBLAS)
make train && ./train

# Train on GPU (with custom CUDA kernels and cuBLAS)
make train_cu && ./train_cu
```

---

## Serverless Web UI Inference (ONNX)

You can test the deployed model in your browser or run it locally using **ONNX Runtime Web** (WebGL/WASM acceleration)—no backend server required!

* 🌐 **Live Web Demo**: [https://llm-from-scratch-edu.web.app/](https://llm-from-scratch-edu.web.app/)

### Local Web UI Execution:
1. **Export the Model**:
   Convert the `tiny_llm.pth` weights to `.onnx` and generate an Int8 quantized version:
   ```bash
   uv sync
   uv run python tools/export/export_onnx.py --quantize
   ```
2. **Launch the Local Server**:
   Serve the web directory locally:
   ```bash
   python3 -m http.server 8000
   ```
   Then open [http://localhost:8000/ui/](http://localhost:8000/ui/) to run inference right in your browser!

---

## Mechanistic Interpretability Dashboard

An interactive Streamlit dashboard is available to inspect the model's inner workings (Self-Attention maps, Logit Lens, FFN activation scans, weight heatmaps/histograms, and head ablation).

To start the dashboard, run:
```bash
uv run streamlit run scripts/interpretability.py
```

---

## Building the Python Package (`dist/`)

To compile the `tiny_llm` package into a redistributable wheel (`.whl`) and source tarball (`.tar.gz`):

```bash
uv run python scripts/build_package.py
# Or run native uv build:
uv build
```

The resulting packages will be generated inside the `dist/` directory:
* `dist/llm_from_scratch-0.1.0-py3-none-any.whl`
* `dist/llm_from_scratch-0.1.0.tar.gz`


