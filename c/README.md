# TinyLLM in Pure C

This directory contains a standalone, pure C inference engine for `TinyLLM`. 

Taking heavy inspiration from Andrej Karpathy's `llama2.c`, this implementation ditches the heavy PyTorch runtime and Python overhead. Instead, it runs the entire forward pass (RMSNorm, RoPE, SwiGLU, and Attention) using bare-metal C arrays and pointers.

## Features

- **Zero Dependencies**: Requires only standard C libraries (`<math.h>`, `<stdio.h>`, etc.).
- **Memory Mapped**: Uses POSIX `mmap()` to instantly map the model weights into RAM directly from the filesystem without huge memory allocations.
- **Blazing Fast**: Because there's no Python interpreter overhead or dynamic graph tracing, inference is extremely fast.

## How it Works

1. **`export.py`**: A Python script that loads your `tiny_llm.pth` PyTorch model and serializes all of its `float32` tensors directly into a contiguous binary sequence (`model.bin`). It also dumps the HuggingFace `tokenizer.json` into a binary format (`vocab.bin`) for the C code to read.
2. **`run.c`**: The main C executable. It defines the struct pointers into the memory-mapped `model.bin`, loads the vocabulary, allocates the KV cache, and runs the autoregressive generation loop.

## Usage

### 1. Build the Executable
First, compile the C code using GCC:
```bash
make
```

### 2. Export the Weights
Use Python to dump the PyTorch weights and vocabulary into binary format. Ensure you have your virtual environment active or use `uv run`:
```bash
uv run python export.py
```
*This will generate `model.bin` and `vocab.bin` in this directory.*

### 3. Run Inference
Execute the compiled binary:
```bash
./run
```

The model will immediately stream tokens autoregressively to `stdout`.
