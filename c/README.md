# TinyLLM in Pure C & CUDA

This directory contains standalone inference engines for `TinyLLM` written in pure C and CUDA.

Taking heavy inspiration from Andrej Karpathy's `llama2.c`, this implementation ditches the heavy PyTorch runtime and Python overhead. It runs the entire forward pass (RMSNorm, RoPE, SwiGLU, and Attention) using bare-metal arrays, pointers, and optimized matrix multiplications.

## Implementations

### Inference Engines
1. **`run.c`**: A pure C implementation. By default, it runs single-threaded matrix multiplications. If `USE_BLAS=1` is provided during compilation, it uses hardware-accelerated OpenBLAS for massive CPU speedups.
2. **`runq.c`**: A quantized C implementation. It performs dynamic INT8 matrix multiplications to reduce memory footprint and execute quantized inference on the CPU.
3. **`run.cu`**: A CUDA C++ implementation designed for NVIDIA GPUs. It uses `cuBLAS` for matrix multiplications and custom `__global__` kernels for operations like `RMSNorm`, `RoPE`, and `SwiGLU` to keep execution entirely on the device.

### Training Engines
1. **`train.c`**: A pure C training script. It implements the entire forward-backward autograd pass from scratch, including Attention causal softmax backward, RMSNorm backward, SwiGLU backward, CrossEntropyLoss backward, and AdamW optimizer updates. Supports OpenMP multi-threading and OpenBLAS acceleration.
2. **`train.cu`**: A GPU-optimized training script written in CUDA C++. It keeps all training states (parameters, activations, gradients, optimizer moments) in VRAM and launches custom forward/backward kernels for Attention, RoPE, RMSNorm, SwiGLU, and CrossEntropy, coupled with cuBLAS for matrix contractions.

## How it Works

- **`export.py`**: A Python script that loads your `tiny_llm.pth` PyTorch model and serializes all of its `float32` tensors directly into a contiguous binary sequence (`model.bin`). It also dumps the HuggingFace `tokenizer.json` into a binary format (`vocab.bin`) for the C/CUDA code to read.
- **`export_q8.py`**: A Python script that loads your `tiny_llm.pth` model and serializes the 2D weights as quantized INT8 values with associated FP32 scale factors per row into a binary sequence (`model_q8.bin`).
- The executables define struct pointers into the memory-mapped model files (`model.bin` or `model_q8.bin`), load the vocabulary, allocate the KV cache, and run the autoregressive generation loop.

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

### 4. Quantized CPU Inference (INT8)
To run the model with 8-bit dynamic quantization:

1. Export the model weights into the quantized binary format:
   ```bash
   uv run python export_q8.py
   ```
   This creates `model_q8.bin` in the current folder.

2. Compile the quantized C code:
   ```bash
   make runq
   ```

3. Execute the binary:
   ```bash
   ./runq
   ```

---

## Training from Scratch in C & CUDA

Both training scripts run training on a synthetic sequence over 40 steps, outputting the optimization loss reduction in real-time.

### 1. CPU Training (C)
Compile and run the training program on the CPU:
```bash
# Standard compile
make train

# Or compile with OpenBLAS acceleration (highly recommended)
make train USE_BLAS=1

# Execute the trainer
./train
```

### 2. GPU Training (CUDA)
If you have an NVIDIA GPU and NVCC installed, compile and execute the GPU trainer:
```bash
# Compile CUDA trainer
make train_cu

# Execute the GPU trainer
./train_cu
```


