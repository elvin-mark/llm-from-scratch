# TinyLLM C Implementation Architecture

This document details the underlying architecture, optimizations, and technical decisions made in the `c/` directory for performing standalone inference of the TinyLLM model.

## 1. Core Philosophy: Memory Mapping (mmap)
Across all CPU implementations, the most critical design decision is the use of POSIX `mmap()`.
When `export.py` or `export_q8.py` serializes the PyTorch model, it dumps the raw binary tensors directly into a contiguous file (`model.bin` or `model_q8.bin`).

Instead of using `malloc()` to allocate gigabytes of RAM and reading the file into memory, we use `mmap()` to memory-map the binary file directly into the process's virtual address space.
- **Zero-Copy**: The OS pages the weights directly from the disk into CPU caches when accessed.
- **Instant Load Times**: The model "loads" instantaneously, regardless of size, because the file isn't physically copied into RAM until the exact byte is touched.

The `Weights` struct in C simply acts as a collection of pointers offset into this memory-mapped region.

## 2. Standard FP32 Inference (`run.c`)
The baseline implementation executes the transformer forward pass using standard single-precision (`float32`) arithmetic.

- **Naive Loops**: Matrix multiplications (`matmul`) are performed using naive 3-level nested `for` loops. While simple and dependency-free, this is bottlenecked by CPU cache misses and lacks SIMD vectorization.
- **RunState**: All dynamically allocated memory for the forward pass (activations, logits, KV cache) is pre-allocated once into a `RunState` struct at startup. **Zero allocations** occur during the actual token generation loop, preventing memory fragmentation and OS overhead.

## 3. Hardware Acceleration (`USE_BLAS=1`)
To resolve the bottlenecks of the naive implementation, `run.c` can be conditionally compiled to link against **OpenBLAS**.

- **Optimized SGEMV**: The inner loops of the `matmul` function are replaced by `cblas_sgemv` (Single Precision General Matrix-Vector Multiplication).
- **SIMD Intrinsics**: OpenBLAS leverages architecture-specific assembly (like AVX2, AVX-512, or ARM NEON) to compute multiple floating-point operations simultaneously. It also optimizes memory access patterns to maximize cache locality.

## 4. CUDA GPU Inference (`run.cu`)
For environments with NVIDIA GPUs, `run.cu` ports the entire inference engine to device memory.

- **cuBLAS**: Replaces CPU matrix multiplications with `cublasSgemv`, utilizing massive parallel GPU cores.
- **Custom Global Kernels**: Operations that are traditionally memory-bound (like `RMSNorm`, `RoPE`, `SwiGLU`, and residual additions) are written as custom `__global__` CUDA kernels.
- **Device Residency**: By implementing these custom kernels, the intermediate activation vectors (`x`, `xb`) never leave the GPU during the transformer blocks. This avoids catastrophic PCIe bus transfer bottlenecks between the host and device.

## 5. INT8 Dynamic Quantization (`runq.c`)
To severely reduce memory bandwidth and footprint, `runq.c` implements `Q8_0` style dynamic quantization (heavily inspired by `llama.cpp`).

### Offline Weight Quantization (`export_q8.py`)
- We keep 1D tensors (token embeddings, norm weights) in FP32.
- For 2D matrices (Q, K, V, FFN), we perform **symmetric row-wise quantization**.
- We find the maximum absolute value (`amax`) in a row, compute a `scale = amax / 127.0`, and divide all row elements by this scale, casting them to 8-bit integers (`int8_t`).
- The `float` scales and `int8_t` weights are written sequentially.

### Dynamic Activation Quantization
During the forward pass, multiplying an FP32 activation vector (`x`) by an INT8 weight matrix requires alignment.
1. **Dynamic Quantization**: The `float* x` vector is dynamically quantized into an `int8_t qx[8192]` array located instantly on the stack.
2. **Integer Arithmetic**: The dot product is performed entirely using integer math (`int32_t += int8_t * int8_t`). CPU Integer ALUs handle this much faster than floating-point math, and moving `int8_t` memory is 4x faster than moving `float32`.
3. **Dequantization**: The final `int32_t` accumulation is multiplied by the combined scale (`weight_scale * activation_scale`) to return it to standard FP32 space before passing it to the next layer.
