# TinyLLM in Pure C & CUDA

This directory contains standalone inference engines for `TinyLLM` written in pure C and CUDA.

Taking heavy inspiration from Andrej Karpathy's `llama2.c`, this implementation ditches the heavy PyTorch runtime and Python overhead. It runs the entire forward pass (RMSNorm, RoPE, SwiGLU, and Attention) using bare-metal arrays, pointers, and optimized matrix multiplications.

## Implementations

1. **`run.c`**: A pure C implementation. By default, it runs single-threaded matrix multiplications. If `USE_BLAS=1` is provided during compilation, it uses hardware-accelerated OpenBLAS for massive CPU speedups.
2. **`run.cu`**: A CUDA C++ implementation designed for NVIDIA GPUs. It uses `cuBLAS` for matrix multiplications and custom `__global__` kernels for operations like `RMSNorm`, `RoPE`, and `SwiGLU` to keep execution entirely on the device.

## How it Works

- **`export.py`**: A Python script that loads your `tiny_llm.pth` PyTorch model and serializes all of its `float32` tensors directly into a contiguous binary sequence (`model.bin`). It also dumps the HuggingFace `tokenizer.json` into a binary format (`vocab.bin`) for the C code to read.
- Both executables define struct pointers into the memory-mapped `model.bin`, load the vocabulary, allocate the KV cache, and run the autoregressive generation loop.

## Usage

### 1. Export the Weights
Use Python to dump the PyTorch weights and vocabulary into binary format. Ensure you have your virtual environment active or use `uv run`:
```bash
uv run python export.py
```

### 2. CPU Inference (Pure C)
Compile the C code using GCC:
```bash
# Standard compile
make run

# Compile with OpenBLAS acceleration (requires libopenblas-dev)
make run USE_BLAS=1
```
Execute the binary:
```bash
./run
```

### 3. GPU Inference (CUDA)
If you have an NVIDIA GPU and the CUDA Toolkit (`nvcc`) installed (e.g., on Google Colab):
```bash
make run_cu
```
Execute the CUDA binary:
```bash
./run_cu
```
