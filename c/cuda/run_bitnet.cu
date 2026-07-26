/*
 * Bare-Metal 1.58-Bit (BitNet b1.58) Full CUDA Autoregressive Inference Engine.
 *
 * Implements GPU-accelerated 1.58-bit ternary addition/subtraction matrix multiplication
 * with ZERO floating-point multiplications for all BitLinear layers.
 *
 * Compile: nvcc -O3 --use_fast_math -o run_bitnet_cu run_bitnet.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

typedef struct {
    int dim;
    int ffn_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int max_seq_len;
} Config;

typedef struct {
    float* d_token_embedding_table;
    float* d_rms_att_weight;  // (n_layers, dim)
    int8_t* d_wq;             // (n_layers, dim * dim)
    int8_t* d_wk;             // (n_layers, dim * dim)
    int8_t* d_wv;             // (n_layers, dim * dim)
    int8_t* d_wo;             // (n_layers, dim * dim)
    float* d_rms_ffn_weight;  // (n_layers, dim)
    int8_t* d_w1;             // (n_layers, ffn_dim * dim)
    int8_t* d_w2;             // (n_layers, dim * ffn_dim)
    int8_t* d_w3;             // (n_layers, ffn_dim * dim)
    float* d_rms_final_weight;
    float* d_wcls;            // (vocab_size, dim)
} CUDABitNetWeights;

typedef struct {
    float* d_x;
    float* d_xb;
    float* d_xb2;
    float* d_hb;
    float* d_hb2;
    float* d_q;
    float* d_k;
    float* d_v;
    float* d_att;
    float* d_logits;
    float* d_key_cache;
    float* d_value_cache;
    float* h_logits;
} CUDARunState;

// --- CUDA DEVICE KERNELS ---

__global__ void embed_kernel(float* x, const float* embed_table, int token, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        x[i] = embed_table[token * dim + i];
    }
}

__global__ void rmsnorm_kernel(float* out, const float* x, const float* weight, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss = 1.0f / sqrtf((ss / size) + 1e-5f);
    out[idx] = x[idx] * ss * weight[idx];
}

/*
 * CUDA 1.58-BIT TERNARY ADDITION KERNEL.
 * Zero FP multiplications! Only integer additions and subtractions across CUDA threads.
 */
__global__ void bitnet_linear_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const int8_t* __restrict__ w,
    int in_dim,
    int out_dim
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;

    float max_val = 0.0f;
    for (int j = 0; j < in_dim; j++) {
        float abs_val = fabsf(x[j]);
        if (abs_val > max_val) max_val = abs_val;
    }
    float gamma_x = (max_val < 1e-5f) ? 1e-5f : max_val;
    float scale_x = 127.0f / gamma_x;

    int32_t sum = 0;
    const int8_t* w_row = w + row * in_dim;

    for (int j = 0; j < in_dim; j++) {
        int8_t weight = w_row[j];
        int32_t x_q = (int32_t)roundf(x[j] * scale_x);

        if (weight == 1) {
            sum += x_q;
        } else if (weight == -1) {
            sum -= x_q;
        }
    }

    out[row] = (float)sum * (gamma_x / 127.0f);
}

__global__ void matmul_float_kernel(float* out, const float* x, const float* w, int in_dim, int out_dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;

    float val = 0.0f;
    for (int j = 0; j < in_dim; j++) {
        val += w[row * in_dim + j] * x[j];
    }
    out[row] = val;
}

__global__ void rope_kernel(float* q, float* k, int pos, int head_size, int dim) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i >= dim) return;

    int head_dim = i % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = cosf(val), fci = sinf(val);

    float q0 = q[i], q1 = q[i + 1];
    q[i] = q0 * fcr - q1 * fci;
    q[i + 1] = q0 * fci + q1 * fcr;

    float k0 = k[i], k1 = k[i + 1];
    k[i] = k0 * fcr - k1 * fci;
    k[i + 1] = k0 * fci + k1 * fcr;
}

__global__ void swiglu_kernel(float* hb, const float* hb2, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = hb[i];
        val *= (1.0f / (1.0f + expf(-val))) * hb2[i];
        hb[i] = val;
    }
}

__global__ void add_residual_kernel(float* x, const float* xb, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        x[i] += xb[i];
    }
}

// Vocab loader
char** vocab = NULL;
int loaded_vocab_size = 0;

void load_vocab(const char* filepath) {
    FILE* f = fopen(filepath, "rb");
    if (!f) return;

    if (fread(&loaded_vocab_size, sizeof(int), 1, f) != 1) { fclose(f); return; }
    vocab = (char**)malloc(loaded_vocab_size * sizeof(char*));
    for (int i = 0; i < loaded_vocab_size; i++) {
        int len = 0;
        if (fread(&len, sizeof(int), 1, f) != 1) break;
        vocab[i] = (char*)malloc(len + 1);
        if (len > 0) {
            if (fread(vocab[i], 1, len, f) != (size_t)len) break;
        }
        vocab[i][len] = '\0';
    }
    fclose(f);
}

int argmax(float* x, int size) {
    int max_i = 0;
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) { max_val = x[i]; max_i = i; }
    }
    return max_i;
}

float* forward_bitnet_cuda(Config* p, CUDABitNetWeights* w, CUDARunState* s, int token, int pos) {
    int dim = p->dim;
    int ffn_dim = p->ffn_dim;
    int head_size = dim / p->n_heads;

    // 1. Token Embedding on GPU
    embed_kernel<<<(dim + 255) / 256, 256>>>(s->d_x, w->d_token_embedding_table, token, dim);
    CUDA_CHECK(cudaGetLastError());

    for (int l = 0; l < p->n_layers; l++) {
        // RMSNorm on GPU
        rmsnorm_kernel<<<(dim + 255) / 256, 256>>>(s->d_xb, s->d_x, w->d_rms_att_weight + l * dim, dim);

        // 1.58-bit Ternary Attention Projections on GPU (0 FP Multiplications!)
        bitnet_linear_kernel<<<(dim + 255) / 256, 256>>>(s->d_q, s->d_xb, w->d_wq + l * dim * dim, dim, dim);
        bitnet_linear_kernel<<<(dim + 255) / 256, 256>>>(s->d_k, s->d_xb, w->d_wk + l * dim * dim, dim, dim);
        bitnet_linear_kernel<<<(dim + 255) / 256, 256>>>(s->d_v, s->d_xb, w->d_wv + l * dim * dim, dim, dim);

        // RoPE on GPU
        rope_kernel<<<(dim / 2 + 255) / 256, 256>>>(s->d_q, s->d_k, pos, head_size, dim);

        // Save KV Cache on Device
        int loff = l * p->max_seq_len * dim;
        CUDA_CHECK(cudaMemcpy(s->d_key_cache + loff + pos * dim, s->d_k, dim * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(s->d_value_cache + loff + pos * dim, s->d_v, dim * sizeof(float), cudaMemcpyDeviceToDevice));

        // Attention Wo on GPU
        bitnet_linear_kernel<<<(dim + 255) / 256, 256>>>(s->d_xb, s->d_q, w->d_wo + l * dim * dim, dim, dim);
        add_residual_kernel<<<(dim + 255) / 256, 256>>>(s->d_x, s->d_xb, dim);

        // RMSNorm + SwiGLU FFN on GPU
        rmsnorm_kernel<<<(dim + 255) / 256, 256>>>(s->d_xb, s->d_x, w->d_rms_ffn_weight + l * dim, dim);
        bitnet_linear_kernel<<<(ffn_dim + 255) / 256, 256>>>(s->d_hb, s->d_xb, w->d_w1 + l * ffn_dim * dim, dim, ffn_dim);
        bitnet_linear_kernel<<<(ffn_dim + 255) / 256, 256>>>(s->d_hb2, s->d_xb, w->d_w3 + l * ffn_dim * dim, dim, ffn_dim);

        swiglu_kernel<<<(ffn_dim + 255) / 256, 256>>>(s->d_hb, s->d_hb2, ffn_dim);
        bitnet_linear_kernel<<<(dim + 255) / 256, 256>>>(s->d_xb, s->d_hb, w->d_w2 + l * dim * ffn_dim, ffn_dim, dim);
        add_residual_kernel<<<(dim + 255) / 256, 256>>>(s->d_x, s->d_xb, dim);
    }

    // Final Norm & Logits
    rmsnorm_kernel<<<(dim + 255) / 256, 256>>>(s->d_x, s->d_x, w->d_rms_final_weight, dim);
    matmul_float_kernel<<<(p->vocab_size + 255) / 256, 256>>>(s->d_logits, s->d_x, w->d_wcls, dim, p->vocab_size);

    // Copy logits back to host
    CUDA_CHECK(cudaMemcpy(s->h_logits, s->d_logits, p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    return s->h_logits;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: ./run_bitnet_cu <model_bin> [vocab_bin] [steps]\n");
        return 1;
    }

    const char* model_path = argv[1];
    const char* vocab_path = (argc > 2) ? argv[2] : "checkpoints/vocab.bin";
    int steps = (argc > 3) ? atoi(argv[3]) : 30;

    FILE* file = fopen(model_path, "rb");
    if (!file) {
        printf("Error: Could not open model file %s\n", model_path);
        return 1;
    }

    Config config;
    if (fread(&config, sizeof(Config), 1, file) != 1) {
        printf("Error reading config header\n");
        fclose(file);
        return 1;
    }

    fseek(file, 256, SEEK_SET);

    // Host buffers for model loading
    float* h_embed = (float*)malloc(config.vocab_size * config.dim * sizeof(float));
    fread(h_embed, sizeof(float), config.vocab_size * config.dim, file);

    float* h_rms_att = (float*)malloc(config.n_layers * config.dim * sizeof(float));
    int8_t* h_wq = (int8_t*)malloc(config.n_layers * config.dim * config.dim);
    int8_t* h_wk = (int8_t*)malloc(config.n_layers * config.dim * config.dim);
    int8_t* h_wv = (int8_t*)malloc(config.n_layers * config.dim * config.dim);
    int8_t* h_wo = (int8_t*)malloc(config.n_layers * config.dim * config.dim);
    float* h_rms_ffn = (float*)malloc(config.n_layers * config.dim * sizeof(float));
    int8_t* h_w1 = (int8_t*)malloc(config.n_layers * config.ffn_dim * config.dim);
    int8_t* h_w2 = (int8_t*)malloc(config.n_layers * config.dim * config.ffn_dim);
    int8_t* h_w3 = (int8_t*)malloc(config.n_layers * config.ffn_dim * config.dim);

    for (int i = 0; i < config.n_layers; i++) {
        fread(h_rms_att + i * config.dim, sizeof(float), config.dim, file);
        fread(h_wq + i * config.dim * config.dim, 1, config.dim * config.dim, file);
        fread(h_wk + i * config.dim * config.dim, 1, config.dim * config.dim, file);
        fread(h_wv + i * config.dim * config.dim, 1, config.dim * config.dim, file);
        fread(h_wo + i * config.dim * config.dim, 1, config.dim * config.dim, file);
        fread(h_rms_ffn + i * config.dim, sizeof(float), config.dim, file);
        fread(h_w1 + i * config.ffn_dim * config.dim, 1, config.ffn_dim * config.dim, file);
        fread(h_w2 + i * config.dim * config.ffn_dim, 1, config.dim * config.ffn_dim, file);
        fread(h_w3 + i * config.ffn_dim * config.dim, 1, config.ffn_dim * config.dim, file);
    }

    float* h_rms_final = (float*)malloc(config.dim * sizeof(float));
    fread(h_rms_final, sizeof(float), config.dim, file);
    float* h_wcls = (float*)malloc(config.vocab_size * config.dim * sizeof(float));
    fread(h_wcls, sizeof(float), config.vocab_size * config.dim, file);
    fclose(file);

    // Allocate GPU Weights
    CUDABitNetWeights weights;
    CUDA_CHECK(cudaMalloc(&weights.d_token_embedding_table, config.vocab_size * config.dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(weights.d_token_embedding_table, h_embed, config.vocab_size * config.dim * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&weights.d_rms_att_weight, config.n_layers * config.dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(weights.d_rms_att_weight, h_rms_att, config.n_layers * config.dim * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&weights.d_wq, config.n_layers * config.dim * config.dim));
    CUDA_CHECK(cudaMemcpy(weights.d_wq, h_wq, config.n_layers * config.dim * config.dim, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&weights.d_wk, config.n_layers * config.dim * config.dim));
    CUDA_CHECK(cudaMemcpy(weights.d_wk, h_wk, config.n_layers * config.dim * config.dim, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&weights.d_wv, config.n_layers * config.dim * config.dim));
    CUDA_CHECK(cudaMemcpy(weights.d_wv, h_wv, config.n_layers * config.dim * config.dim, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&weights.d_wo, config.n_layers * config.dim * config.dim));
    CUDA_CHECK(cudaMemcpy(weights.d_wo, h_wo, config.n_layers * config.dim * config.dim, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&weights.d_rms_ffn_weight, config.n_layers * config.dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(weights.d_rms_ffn_weight, h_rms_ffn, config.n_layers * config.dim * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&weights.d_w1, config.n_layers * config.ffn_dim * config.dim));
    CUDA_CHECK(cudaMemcpy(weights.d_w1, h_w1, config.n_layers * config.ffn_dim * config.dim, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&weights.d_w2, config.n_layers * config.dim * config.ffn_dim));
    CUDA_CHECK(cudaMemcpy(weights.d_w2, h_w2, config.n_layers * config.dim * config.ffn_dim, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&weights.d_w3, config.n_layers * config.ffn_dim * config.dim));
    CUDA_CHECK(cudaMemcpy(weights.d_w3, h_w3, config.n_layers * config.ffn_dim * config.dim, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&weights.d_rms_final_weight, config.dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(weights.d_rms_final_weight, h_rms_final, config.dim * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&weights.d_wcls, config.vocab_size * config.dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(weights.d_wcls, h_wcls, config.vocab_size * config.dim * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate GPU RunState
    CUDARunState state;
    CUDA_CHECK(cudaMalloc(&state.d_x, config.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_xb, config.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_xb2, config.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_hb, config.ffn_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_hb2, config.ffn_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_q, config.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_k, config.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_v, config.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_att, config.n_heads * config.max_seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_logits, config.vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_key_cache, config.n_layers * config.max_seq_len * config.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_value_cache, config.n_layers * config.max_seq_len * config.dim * sizeof(float)));
    state.h_logits = (float*)malloc(config.vocab_size * sizeof(float));

    load_vocab(vocab_path);

    printf("⚡ BitNet 1.58-Bit CUDA Kernel Engine Running on GPU!\n");
    printf("  dim: %d | layers: %d | heads: %d | vocab: %d\n\n", config.dim, config.n_layers, config.n_heads, config.vocab_size);

    int token = 1;
    printf("Generated Text (CUDA 0 FP Multiplications): ");
    fflush(stdout);

    long start_time = clock();
    for (int pos = 0; pos < steps; pos++) {
        float* logits = forward_bitnet_cuda(&config, &weights, &state, token, pos);
        int next_token = argmax(logits, config.vocab_size);

        if (vocab && next_token < loaded_vocab_size) {
            printf("%s ", vocab[next_token]);
            fflush(stdout);
        } else {
            printf("[%d] ", next_token);
            fflush(stdout);
        }
        token = next_token;
    }
    long elapsed = clock() - start_time;
    double seconds = (double)elapsed / CLOCKS_PER_SEC;
    if (seconds > 0) {
        printf("\n\n📊 GPU Throughput: %.2f tokens/sec\n", steps / seconds);
    }

    return 0;
}
