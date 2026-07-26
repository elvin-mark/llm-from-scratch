#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// Model Config
typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int max_seq_len;
} Config;

// Weights mapped to GPU Device Pointers
typedef struct {
    float* token_embedding_table; // [vocab_size, dim]
    float** rms_att_weight;       // [n_layers][dim]
    float** wq;                   // [n_layers][dim, dim]
    float** wk;                   // [n_layers][dim, dim]
    float** wv;                   // [n_layers][dim, dim]
    float** wo;                   // [n_layers][dim, dim]
    float** rms_ffn_weight;       // [n_layers][dim]
    float** w1;                   // [n_layers][hidden_dim, dim]
    float** w2;                   // [n_layers][dim, hidden_dim]
    float** w3;                   // [n_layers][hidden_dim, dim]
    float* rms_final_weight;      // [dim]
    float* wcls;                  // [vocab_size, dim]
} Weights;

// Activations on GPU Device
typedef struct {
    float* xb;      // [n_layers][B, T, dim]
    float* q;       // [n_layers][B, T, dim]
    float* k;       // [n_layers][B, T, dim]
    float* v;       // [n_layers][B, T, dim]
    float* q_rot;   // [n_layers][B, T, dim]
    float* k_rot;   // [n_layers][B, T, dim]
    float* att;     // [n_layers][B, n_heads, T, T]
    float* xb2;     // [n_layers][B, T, dim]
    float* ffn_xb;  // [n_layers][B, T, dim]
    float* hb;      // [n_layers][B, T, hidden_dim]
    float* hb2;     // [n_layers][B, T, hidden_dim]
    float* hb_act;  // [n_layers][B, T, hidden_dim]
} LayerActivations;

typedef struct {
    float* x;            // [n_layers + 1][B, T, dim]
    LayerActivations* l; // array of size n_layers
    float* x_final_norm; // [B, T, dim]
    float* logits;       // [B, T, vocab_size]
    float* probs;        // [B, T, vocab_size]
} BatchActivations;

// Gradients on GPU Device
typedef struct {
    float* d_x;           // [n_layers + 1][B, T, dim]
    float* d_x_final_norm;// [B, T, dim]
    float* d_logits;      // [B, T, vocab_size]
    
    // Weight gradients
    float* token_embedding_table; // [vocab_size, dim]
    float** rms_att_weight;       // [n_layers][dim]
    float** wq;                   // [n_layers][dim, dim]
    float** wk;                   // [n_layers][dim, dim]
    float** wv;                   // [n_layers][dim, dim]
    float** wo;                   // [n_layers][dim, dim]
    float** rms_ffn_weight;       // [n_layers][dim]
    float** w1;                   // [n_layers][hidden_dim, dim]
    float** w2;                   // [n_layers][dim, hidden_dim]
    float** w3;                   // [n_layers][hidden_dim, dim]
    float* rms_final_weight;      // [dim]
    float* wcls;                  // [vocab_size, dim]

    // Layer-wise activation gradients (temporary buffers)
    float* d_xb;     // [B, T, dim]
    float* d_q;      // [B, T, dim]
    float* d_k;      // [B, T, dim]
    float* d_v;      // [B, T, dim]
    float* d_q_rot;  // [B, T, dim]
    float* d_k_rot;  // [B, T, dim]
    float* d_att;    // [B, n_heads, T, T]
    float* d_xb2;    // [B, T, dim]
    float* d_ffn_xb; // [B, T, dim]
    float* d_hb;     // [B, T, hidden_dim]
    float* d_hb2;    // [B, T, hidden_dim]
    float* d_hb_act; // [B, T, hidden_dim]
} Gradients;

// -----------------------------------------------------------------------------
// CUDA Kernels for Forward and Backward Passes

__global__ void embedding_forward_kernel(float* x, float* table, int* inputs, int dim, int T) {
    int b = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T) {
        int token = inputs[b * T + t];
        for (int i = 0; i < dim; i++) {
            x[(b * T + t) * dim + i] = table[token * dim + i];
        }
    }
}

__global__ void embedding_backward_kernel(float* d_table, float* d_x, int* inputs, int dim, int T) {
    int b = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T) {
        int token = inputs[b * T + t];
        for (int i = 0; i < dim; i++) {
            atomicAdd(&d_table[token * dim + i], d_x[(b * T + t) * dim + i]);
        }
    }
}

__global__ void rmsnorm_forward_kernel(float* o, float* x, float* weight, int size, int total_rows, float* cache_std) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_rows) {
        float ss = 0.0f;
        for (int j = 0; j < size; j++) {
            float val = x[idx * size + j];
            ss += val * val;
        }
        ss = ss / size + 1e-6f;
        float rstd = rsqrtf(ss);
        cache_std[idx] = rstd;
        for (int j = 0; j < size; j++) {
            o[idx * size + j] = weight[j] * x[idx * size + j] * rstd;
        }
    }
}

__global__ void rmsnorm_backward_kernel(float* dx, float* dw, float* dy, float* x, float* weight, int size, int total_rows, float* cache_std) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_rows) {
        float rstd = cache_std[idx];
        float sum_dy_w_x = 0.0f;
        for (int j = 0; j < size; j++) {
            sum_dy_w_x += dy[idx * size + j] * weight[j] * x[idx * size + j];
        }
        float factor = sum_dy_w_x * rstd * rstd * rstd / size;
        for (int j = 0; j < size; j++) {
            dx[idx * size + j] = weight[j] * dy[idx * size + j] * rstd - x[idx * size + j] * factor;
            atomicAdd(&dw[j], dy[idx * size + j] * x[idx * size + j] * rstd);
        }
    }
}

__global__ void rope_forward_kernel(float* q, float* k, int B, int T, int dim, int head_size, int n_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int slices_per_token = n_heads * (head_size / 2);
    int total_slices = B * T * slices_per_token;
    if (idx < total_slices) {
        int b_t_idx = idx / slices_per_token;
        int t = b_t_idx % T;
        int h_i_idx = idx % slices_per_token;
        int h = h_i_idx / (head_size / 2);
        int i_pair = h_i_idx % (head_size / 2);
        int i = i_pair * 2;

        int global_idx = b_t_idx * dim + h * head_size + i;
        float freq = 1.0f / powf(10000.0f, i / (float)head_size);
        float val = t * freq;
        float fcr = cosf(val);
        float fci = sinf(val);

        float q0 = q[global_idx];
        float q1 = q[global_idx + 1];
        q[global_idx]     = q0 * fcr - q1 * fci;
        q[global_idx + 1] = q0 * fci + q1 * fcr;

        float k0 = k[global_idx];
        float k1 = k[global_idx + 1];
        k[global_idx]     = k0 * fcr - k1 * fci;
        k[global_idx + 1] = k0 * fci + k1 * fcr;
    }
}

__global__ void rope_backward_kernel(float* dq, float* dk, float* dq_rot, float* dk_rot, int B, int T, int dim, int head_size, int n_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int slices_per_token = n_heads * (head_size / 2);
    int total_slices = B * T * slices_per_token;
    if (idx < total_slices) {
        int b_t_idx = idx / slices_per_token;
        int t = b_t_idx % T;
        int h_i_idx = idx % slices_per_token;
        int h = h_i_idx / (head_size / 2);
        int i_pair = h_i_idx % (head_size / 2);
        int i = i_pair * 2;

        int global_idx = b_t_idx * dim + h * head_size + i;
        float freq = 1.0f / powf(10000.0f, i / (float)head_size);
        float val = t * freq;
        float fcr = cosf(val);
        float fci = sinf(val);

        float dq0 = dq_rot[global_idx];
        float dq1 = dq_rot[global_idx + 1];
        dq[global_idx]     = dq0 * fcr + dq1 * fci;
        dq[global_idx + 1] = -dq0 * fci + dq1 * fcr;

        float dk0 = dk_rot[global_idx];
        float dk1 = dk_rot[global_idx + 1];
        dk[global_idx]     = dk0 * fcr + dk1 * fci;
        dk[global_idx + 1] = -dk0 * fci + dk1 * fcr;
    }
}

__global__ void attention_forward_kernel(float* att, float* q, float* k, int B, int n_heads, int T, int head_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = B * n_heads * T * T;
    if (idx < total_cells) {
        int b_h_idx = idx / (T * T);
        int b = b_h_idx / n_heads;
        int h = b_h_idx % n_heads;
        int t_tau_idx = idx % (T * T);
        int t = t_tau_idx / T;
        int tau = t_tau_idx % T;

        if (tau <= t) {
            float* q_vec = q + b * T * dim + t * dim + h * head_size;
            float* k_vec = k + b * T * dim + tau * dim + h * head_size;
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += q_vec[i] * k_vec[i];
            }
            att[idx] = score / sqrtf((float)head_size);
        } else {
            att[idx] = -1e30f; // Causal mask
        }
    }
}

__global__ void softmax_forward_kernel(float* att, int B, int n_heads, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = B * n_heads * T;
    if (idx < total_rows) {
        int b = idx / (n_heads * T);
        int h_t_idx = idx % (n_heads * T);
        int h = h_t_idx / T;
        int t = h_t_idx % T;
        
        float* row = att + b * n_heads * T * T + h * T * T + t * T;
        float max_val = -1e30f;
        for (int col = 0; col <= t; col++) {
            if (row[col] > max_val) max_val = row[col];
        }
        float sum = 0.0f;
        for (int col = 0; col <= t; col++) {
            row[col] = expf(row[col] - max_val);
            sum += row[col];
        }
        for (int col = 0; col <= t; col++) {
            row[col] /= sum;
        }
        for (int col = t + 1; col < T; col++) {
            row[col] = 0.0f;
        }
    }
}

__global__ void attention_weighted_sum_kernel(float* xb2, float* att, float* v, int B, int n_heads, int T, int head_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * T * dim;
    if (idx < total_elements) {
        int b_t_idx = idx / dim;
        int b = b_t_idx / T;
        int t = b_t_idx % T;
        int h_i_idx = idx % dim;
        int h = h_i_idx / head_size;
        int i = h_i_idx % head_size;

        float* att_row = att + b * n_heads * T * T + h * T * T + t * T;
        float sum = 0.0f;
        for (int tau = 0; tau <= t; tau++) {
            float* v_vec = v + b * T * dim + tau * dim + h * head_size;
            sum += att_row[tau] * v_vec[i];
        }
        xb2[idx] = sum;
    }
}

__global__ void attention_weighted_sum_backward_kernel(float* d_v, float* d_att, float* d_xb2, float* att, float* v, int B, int n_heads, int T, int head_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = B * n_heads * T * T;
    if (idx < total_cells) {
        int b_h_idx = idx / (T * T);
        int b = b_h_idx / n_heads;
        int h = b_h_idx % n_heads;
        int t_tau_idx = idx % (T * T);
        int t = t_tau_idx / T;
        int tau = t_tau_idx % T;

        if (tau <= t) {
            float* d_out_vec = d_xb2 + b * T * dim + t * dim + h * head_size;
            float* v_vec = v + b * T * dim + tau * dim + h * head_size;
            float* d_v_vec = d_v + b * T * dim + tau * dim + h * head_size;
            float att_val = att[idx];

            float d_att_val = 0.0f;
            for (int i = 0; i < head_size; i++) {
                d_att_val += d_out_vec[i] * v_vec[i];
                atomicAdd(&d_v_vec[i], d_out_vec[i] * att_val);
            }
            d_att[idx] = d_att_val;
        } else {
            d_att[idx] = 0.0f;
        }
    }
}

__global__ void softmax_backward_kernel(float* d_att_logits, float* att, float* d_att, int B, int n_heads, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = B * n_heads * T;
    if (idx < total_rows) {
        int b = idx / (n_heads * T);
        int h_t_idx = idx % (n_heads * T);
        int h = h_t_idx / T;
        int t = h_t_idx % T;

        float* row_att = att + b * n_heads * T * T + h * T * T + t * T;
        float* row_d_att = d_att + b * n_heads * T * T + h * T * T + t * T;
        float* row_d_att_logits = d_att_logits + b * n_heads * T * T + h * T * T + t * T;

        float sum_ds_s = 0.0f;
        for (int col = 0; col <= t; col++) {
            sum_ds_s += row_d_att[col] * row_att[col];
        }
        for (int col = 0; col <= t; col++) {
            row_d_att_logits[col] = row_att[col] * (row_d_att[col] - sum_ds_s);
        }
        for (int col = t + 1; col < T; col++) {
            row_d_att_logits[col] = 0.0f;
        }
    }
}

__global__ void attention_backward_kernel(float* d_q, float* d_k, float* d_att_logits, float* q, float* k, int B, int n_heads, int T, int head_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = B * n_heads * T * T;
    if (idx < total_cells) {
        int b_h_idx = idx / (T * T);
        int b = b_h_idx / n_heads;
        int h = b_h_idx % n_heads;
        int t_tau_idx = idx % (T * T);
        int t = t_tau_idx / T;
        int tau = t_tau_idx % T;

        if (tau <= t) {
            float d_att_val = d_att_logits[idx] / sqrtf((float)head_size);
            float* q_vec = q + b * T * dim + t * dim + h * head_size;
            float* k_vec = k + b * T * dim + tau * dim + h * head_size;
            float* d_q_vec = d_q + b * T * dim + t * dim + h * head_size;
            float* d_k_vec = d_k + b * T * dim + tau * dim + h * head_size;

            for (int i = 0; i < head_size; i++) {
                atomicAdd(&d_q_vec[i], k_vec[i] * d_att_val);
                atomicAdd(&d_k_vec[i], q_vec[i] * d_att_val);
            }
        }
    }
}

__global__ void swiglu_forward_kernel(float* hb_act, float* hb, float* hb2, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = hb[idx];
        float silu = val * (1.0f / (1.0f + expf(-val)));
        hb_act[idx] = silu * hb2[idx];
    }
}

__global__ void swiglu_backward_kernel(float* d_hb, float* d_hb2, float* d_hb_act, float* hb, float* hb2, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = hb[idx];
        float sig = 1.0f / (1.0f + expf(-val));
        float silu = val * sig;
        float d_silu = sig * (1.0f + val * (1.0f - sig));

        d_hb2[idx] = d_hb_act[idx] * silu;
        d_hb[idx]  = d_hb_act[idx] * hb2[idx] * d_silu;
    }
}

__global__ void add_kernel(float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] += y[idx];
    }
}

__global__ void cross_entropy_forward_kernel(float* probs, float* logits, int* targets, float* loss_accum, int B, int T, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * T) {
        float* logit_row = logits + idx * V;
        float* prob_row = probs + idx * V;

        float max_val = logit_row[0];
        for (int v = 1; v < V; v++) {
            if (logit_row[v] > max_val) max_val = logit_row[v];
        }
        float sum = 0.0f;
        for (int v = 0; v < V; v++) {
            prob_row[v] = expf(logit_row[v] - max_val);
            sum += prob_row[v];
        }
        for (int v = 0; v < V; v++) {
            prob_row[v] /= sum;
        }

        int target = targets[idx];
        if (target >= 0 && target < V) {
            loss_accum[idx] = -logf(prob_row[target] + 1e-9f);
        } else {
            loss_accum[idx] = 0.0f;
        }
    }
}

__global__ void cross_entropy_backward_kernel(float* d_logits, float* probs, int* targets, int B, int T, int V, int active_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * T * V;
    if (idx < total_threads) {
        int row = idx / V;
        int v = idx % V;
        int target = targets[row];
        float p = probs[row * V + v];
        float scale = 1.0f / (active_count > 0 ? active_count : 1);
        if (target >= 0 && target < V) {
            d_logits[row * V + v] = scale * (p - (v == target ? 1.0f : 0.0f));
        } else {
            d_logits[row * V + v] = 0.0f;
        }
    }
}

__global__ void adamw_update_kernel(float* w, float* g, float* m, float* v, int size, float lr, float wd, float beta1, float beta2, float eps, int step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float correction1 = 1.0f - powf(beta1, step);
        float correction2 = 1.0f - powf(beta2, step);
        float step_lr = lr * sqrtf(correction2) / correction1;

        w[idx] -= lr * wd * w[idx]; // L2 Weight Decay
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g[idx];
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g[idx] * g[idx];

        w[idx] -= step_lr * m[idx] / (sqrtf(v[idx]) + eps);
    }
}

// -----------------------------------------------------------------------------
// cuBLAS Matrix Multiplication Wrappers
void matmul_forward_cublas(cublasHandle_t handle, float* out, float* x, float* w, int M, int N, int D) {
    float alpha = 1.0f;
    float beta = 0.0f;
    // cuBLAS performs column-major multiplications. 
    // To calculate Row-Major out = X * W^T (where X is M x N, W is D x N), we do:
    // W^T is shape N x D, so out^T = W * X^T (shape D x M)
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, D, M, N, &alpha, w, N, x, N, &beta, out, D);
}

void matmul_backward_input_cublas(cublasHandle_t handle, float* dx, float* dy, float* w, int M, int N, int D) {
    float alpha = 1.0f;
    float beta = 0.0f;
    // dx^T = W^T * dy^T (shape N x M, where W^T is N x D, dy^T is D x M)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, D, &alpha, w, N, dy, D, &beta, dx, N);
}

void matmul_backward_weights_cublas(cublasHandle_t handle, float* dw, float* dy, float* x, int M, int N, int D) {
    float alpha = 1.0f;
    float beta = 1.0f; // Accumulate gradients (dw += dy^T * x)
    // dw^T = x^T * dy (shape N x D, where x^T is N x M, dy is M x D)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, D, M, &alpha, x, N, dy, D, &beta, dw, N);
}

// -----------------------------------------------------------------------------
// GPU Forward Pass
void model_forward_gpu(cublasHandle_t handle, Config* c, Weights* w, BatchActivations* act, int* d_inputs, int B, int T, float* cache_rstd) {
    int dim = c->dim;
    int hidden_dim = c->hidden_dim;
    int n_heads = c->n_heads;
    int head_size = dim / n_heads;

    // 1. Embeddings copy
    dim3 block_emb(256);
    dim3 grid_emb((T + 255) / 256, B);
    embedding_forward_kernel<<<grid_emb, block_emb>>>(act->x, w->token_embedding_table, d_inputs, dim, T);
    CHECK_CUDA(cudaGetLastError());

    // 2. Layers Forward Loop
    for (int l = 0; l < c->n_layers; l++) {
        LayerActivations* la = &act->l[l];
        float* layer_in = act->x + l * B * T * dim;
        float* layer_out = act->x + (l + 1) * B * T * dim;

        // Attention Norm
        rmsnorm_forward_kernel<<<(B * T + 255) / 256, 256>>>(la->xb, layer_in, w->rms_att_weight[l], dim, B * T, cache_rstd + l * 2 * B * T);
        CHECK_CUDA(cudaGetLastError());

        // Projections
        matmul_forward_cublas(handle, la->q, la->xb, w->wq[l], B * T, dim, dim);
        matmul_forward_cublas(handle, la->k, la->xb, w->wk[l], B * T, dim, dim);
        matmul_forward_cublas(handle, la->v, la->xb, w->wv[l], B * T, dim, dim);

        // Apply RoPE
        CHECK_CUDA(cudaMemcpy(la->q_rot, la->q, B * T * dim * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(la->k_rot, la->k, B * T * dim * sizeof(float), cudaMemcpyDeviceToDevice));
        int total_rope_slices = B * T * n_heads * (head_size / 2);
        rope_forward_kernel<<<(total_rope_slices + 255) / 256, 256>>>(la->q_rot, la->k_rot, B, T, dim, head_size, n_heads);
        CHECK_CUDA(cudaGetLastError());

        // Attention matrix computation
        int total_cells = B * n_heads * T * T;
        attention_forward_kernel<<<(total_cells + 255) / 256, 256>>>(la->att, la->q_rot, la->k_rot, B, n_heads, T, head_size, dim);
        CHECK_CUDA(cudaGetLastError());

        // Softmax
        softmax_forward_kernel<<<(B * n_heads * T + 255) / 256, 256>>>(la->att, B, n_heads, T);
        CHECK_CUDA(cudaGetLastError());

        // Weighted Attention value sum (xb2)
        attention_weighted_sum_kernel<<<(B * T * dim + 255) / 256, 256>>>(la->xb2, la->att, la->v, B, n_heads, T, head_size, dim);
        CHECK_CUDA(cudaGetLastError());

        // Wo projection and residual addition
        matmul_forward_cublas(handle, layer_out, la->xb2, w->wo[l], B * T, dim, dim);
        add_kernel<<<(B * T * dim + 255) / 256, 256>>>(layer_out, layer_in, B * T * dim);
        CHECK_CUDA(cudaGetLastError());

        // FFN norm & MLP
        float* ffn_in = layer_out;
        rmsnorm_forward_kernel<<<(B * T + 255) / 256, 256>>>(la->ffn_xb, ffn_in, w->rms_ffn_weight[l], dim, B * T, cache_rstd + (l * 2 + 1) * B * T);
        CHECK_CUDA(cudaGetLastError());

        matmul_forward_cublas(handle, la->hb, la->ffn_xb, w->w1[l], B * T, dim, hidden_dim);
        matmul_forward_cublas(handle, la->hb2, la->ffn_xb, w->w3[l], B * T, dim, hidden_dim);

        swiglu_forward_kernel<<<(B * T * hidden_dim + 255) / 256, 256>>>(la->hb_act, la->hb, la->hb2, B * T * hidden_dim);
        CHECK_CUDA(cudaGetLastError());

        // w2 projection and final addition
        float* ffn_out = la->hb; // reuse hb
        matmul_forward_cublas(handle, ffn_out, la->hb_act, w->w2[l], B * T, hidden_dim, dim);
        add_kernel<<<(B * T * dim + 255) / 256, 256>>>(layer_out, ffn_out, B * T * dim);
        CHECK_CUDA(cudaGetLastError());
    }

    // 3. Final model output
    float* final_in = act->x + c->n_layers * B * T * dim;
    rmsnorm_forward_kernel<<<(B * T + 255) / 256, 256>>>(act->x_final_norm, final_in, w->rms_final_weight, dim, B * T, cache_rstd + c->n_layers * 2 * B * T);
    matmul_forward_cublas(handle, act->logits, act->x_final_norm, w->wcls, B * T, dim, c->vocab_size);
}

// -----------------------------------------------------------------------------
// GPU Backward Pass (Autograd)
void model_backward_gpu(cublasHandle_t handle, Config* c, Weights* w, Gradients* g, BatchActivations* act, int* d_inputs, int B, int T, float* cache_rstd) {
    int dim = c->dim;
    int hidden_dim = c->hidden_dim;
    int n_heads = c->n_heads;
    int head_size = dim / n_heads;

    // 1. Output logits projection backward
    matmul_backward_weights_cublas(handle, g->wcls, g->d_logits, act->x_final_norm, B * T, dim, c->vocab_size);
    matmul_backward_input_cublas(handle, g->d_x_final_norm, g->d_logits, w->wcls, B * T, dim, c->vocab_size);

    // Final RMSNorm backward
    float* final_in = act->x + c->n_layers * B * T * dim;
    float* d_final_in = g->d_x + c->n_layers * B * T * dim;
    rmsnorm_backward_kernel<<<(B * T + 255) / 256, 256>>>(d_final_in, g->rms_final_weight, g->d_x_final_norm, final_in, w->rms_final_weight, dim, B * T, cache_rstd + c->n_layers * 2 * B * T);
    CHECK_CUDA(cudaGetLastError());

    // 2. Transformer layers backward
    for (int l = c->n_layers - 1; l >= 0; l--) {
        LayerActivations* la = &act->l[l];
        float* layer_in = act->x + l * B * T * dim;
        float* d_layer_in = g->d_x + l * B * T * dim;
        float* d_layer_out = g->d_x + (l + 1) * B * T * dim;

        // FFN backward
        float* ffn_in = act->x + (l + 1) * B * T * dim;
        float* d_ffn_out = d_layer_out;

        matmul_backward_weights_cublas(handle, g->w2[l], d_ffn_out, la->hb_act, B * T, hidden_dim, dim);
        matmul_backward_input_cublas(handle, g->d_hb_act, d_ffn_out, w->w2[l], B * T, hidden_dim, dim);

        swiglu_backward_kernel<<<(B * T * hidden_dim + 255) / 256, 256>>>(g->d_hb, g->d_hb2, g->d_hb_act, la->hb, la->hb2, B * T * hidden_dim);
        CHECK_CUDA(cudaGetLastError());

        matmul_backward_weights_cublas(handle, g->w3[l], g->d_hb2, la->ffn_xb, B * T, dim, hidden_dim);
        matmul_backward_input_cublas(handle, g->d_ffn_xb, g->d_hb2, w->w3[l], B * T, dim, hidden_dim);

        matmul_backward_weights_cublas(handle, g->w1[l], g->d_hb, la->ffn_xb, B * T, dim, hidden_dim);
        
        float* d_ffn_xb_tmp = g->d_hb2; // reuse hb2 buffer
        matmul_backward_input_cublas(handle, d_ffn_xb_tmp, g->d_hb, w->w1[l], B * T, dim, hidden_dim);
        add_kernel<<<(B * T * dim + 255) / 256, 256>>>(g->d_ffn_xb, d_ffn_xb_tmp, B * T * dim);
        CHECK_CUDA(cudaGetLastError());

        // FFN RMSNorm backward
        float* d_ffn_in = g->d_hb;
        rmsnorm_backward_kernel<<<(B * T + 255) / 256, 256>>>(d_ffn_in, g->rms_ffn_weight[l], g->d_ffn_xb, ffn_in, w->rms_ffn_weight[l], dim, B * T, cache_rstd + (l * 2 + 1) * B * T);
        CHECK_CUDA(cudaGetLastError());

        // Intermediate gradients from FFN residual path
        float* d_h1 = g->d_xb;
        CHECK_CUDA(cudaMemcpy(d_h1, d_layer_out, B * T * dim * sizeof(float), cudaMemcpyDeviceToDevice));
        add_kernel<<<(B * T * dim + 255) / 256, 256>>>(d_h1, d_ffn_in, B * T * dim);
        CHECK_CUDA(cudaGetLastError());

        // Attention backward
        matmul_backward_weights_cublas(handle, g->wo[l], d_h1, la->xb2, B * T, dim, dim);
        matmul_backward_input_cublas(handle, g->d_xb2, d_h1, w->wo[l], B * T, dim, dim);

        // Value projection and Attention matrix backward
        CHECK_CUDA(cudaMemset(g->d_v, 0, B * T * dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(g->d_att, 0, B * n_heads * T * T * sizeof(float)));

        int total_cells = B * n_heads * T * T;
        attention_weighted_sum_backward_kernel<<<(total_cells + 255) / 256, 256>>>(g->d_v, g->d_att, g->d_xb2, la->att, la->v, B, n_heads, T, head_size, dim);
        CHECK_CUDA(cudaGetLastError());

        // Softmax backward
        float* d_att_logits = g->d_q; // reuse q buffer temporarily
        softmax_backward_kernel<<<(B * n_heads * T + 255) / 256, 256>>>(d_att_logits, la->att, g->d_att, B, n_heads, T);
        CHECK_CUDA(cudaGetLastError());

        // Q & K dot product backward
        CHECK_CUDA(cudaMemset(g->d_q_rot, 0, B * T * dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(g->d_k_rot, 0, B * T * dim * sizeof(float)));
        attention_backward_kernel<<<(total_cells + 255) / 256, 256>>>(g->d_q_rot, g->d_k_rot, d_att_logits, la->q_rot, la->k_rot, B, n_heads, T, head_size, dim);
        CHECK_CUDA(cudaGetLastError());

        // RoPE backward
        int total_rope_slices = B * T * n_heads * (head_size / 2);
        rope_backward_kernel<<<(total_rope_slices + 255) / 256, 256>>>(g->d_q, g->d_k, g->d_q_rot, g->d_k_rot, B, T, dim, head_size, n_heads);
        CHECK_CUDA(cudaGetLastError());

        // Attention weights backward
        matmul_backward_weights_cublas(handle, g->wq[l], g->d_q, la->xb, B * T, dim, dim);
        matmul_backward_weights_cublas(handle, g->wk[l], g->d_k, la->xb, B * T, dim, dim);
        matmul_backward_weights_cublas(handle, g->wv[l], g->d_v, la->xb, B * T, dim, dim);

        // Accumulate into d_xb
        matmul_backward_input_cublas(handle, g->d_xb, g->d_q, w->wq[l], B * T, dim, dim);
        
        float* d_xb_tmp = g->d_xb2; // reuse xb2 buffer
        matmul_backward_input_cublas(handle, d_xb_tmp, g->d_k, w->wk[l], B * T, dim, dim);
        add_kernel<<<(B * T * dim + 255) / 256, 256>>>(g->d_xb, d_xb_tmp, B * T * dim);
        CHECK_CUDA(cudaGetLastError());

        matmul_backward_input_cublas(handle, d_xb_tmp, g->d_v, w->wv[l], B * T, dim, dim);
        add_kernel<<<(B * T * dim + 255) / 256, 256>>>(g->d_xb, d_xb_tmp, B * T * dim);
        CHECK_CUDA(cudaGetLastError());

        // Attention RMSNorm backward
        rmsnorm_backward_kernel<<<(B * T + 255) / 256, 256>>>(d_layer_in, g->rms_att_weight[l], g->d_xb, layer_in, w->rms_att_weight[l], dim, B * T, cache_rstd + l * 2 * B * T);
        CHECK_CUDA(cudaGetLastError());

        // Add back layer input residual
        add_kernel<<<(B * T * dim + 255) / 256, 256>>>(d_layer_in, d_h1, B * T * dim);
        CHECK_CUDA(cudaGetLastError());
    }

    // 3. Accumulate token embeddings gradient
    dim3 block_emb(256);
    dim3 grid_emb((T + 255) / 256, B);
    embedding_backward_kernel<<<grid_emb, block_emb>>>(g->token_embedding_table, g->d_x, d_inputs, dim, T);
    CHECK_CUDA(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// Main execution / Synthetic Training Loop
int main() {
    // Model architecture setup
    Config c = {
        .dim = 64,
        .hidden_dim = 128,
        .n_layers = 2,
        .n_heads = 2,
        .n_kv_heads = 2,
        .vocab_size = 256,
        .max_seq_len = 32
    };

    int B = 4;
    int T = 16;

    printf("Initializing CUDA-based TinyLLM training configuration:\n");
    printf("- dim: %d, layers: %d, heads: %d, vocab: %d\n", c.dim, c.n_layers, c.n_heads, c.vocab_size);
    printf("- batch: %d, seqlen: %d\n", B, T);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Host parameters & gradients allocation
    Weights h_w;
    h_w.token_embedding_table = (float*)malloc(c.vocab_size * c.dim * sizeof(float));
    h_w.rms_att_weight = (float**)malloc(c.n_layers * sizeof(float*));
    h_w.wq = (float**)malloc(c.n_layers * sizeof(float*));
    h_w.wk = (float**)malloc(c.n_layers * sizeof(float*));
    h_w.wv = (float**)malloc(c.n_layers * sizeof(float*));
    h_w.wo = (float**)malloc(c.n_layers * sizeof(float*));
    h_w.rms_ffn_weight = (float**)malloc(c.n_layers * sizeof(float*));
    h_w.w1 = (float**)malloc(c.n_layers * sizeof(float*));
    h_w.w2 = (float**)malloc(c.n_layers * sizeof(float*));
    h_w.w3 = (float**)malloc(c.n_layers * sizeof(float*));

    for (int l = 0; l < c.n_layers; l++) {
        h_w.rms_att_weight[l] = (float*)malloc(c.dim * sizeof(float));
        h_w.wq[l] = (float*)malloc(c.dim * c.dim * sizeof(float));
        h_w.wk[l] = (float*)malloc(c.dim * c.dim * sizeof(float));
        h_w.wv[l] = (float*)malloc(c.dim * c.dim * sizeof(float));
        h_w.wo[l] = (float*)malloc(c.dim * c.dim * sizeof(float));
        h_w.rms_ffn_weight[l] = (float*)malloc(c.dim * sizeof(float));
        h_w.w1[l] = (float*)malloc(c.hidden_dim * c.dim * sizeof(float));
        h_w.w2[l] = (float*)malloc(c.dim * c.hidden_dim * sizeof(float));
        h_w.w3[l] = (float*)malloc(c.hidden_dim * c.dim * sizeof(float));
    }
    h_w.rms_final_weight = (float*)malloc(c.dim * sizeof(float));
    h_w.wcls = (float*)malloc(c.vocab_size * c.dim * sizeof(float));

    // Initialize Host weights (Kaiming Normal)
    srand(42);
    float scale = 0.02f;
    #define INIT_NORMAL(arr, len) do { for(int i=0; i<len; i++) { float u1 = (float)rand()/RAND_MAX; float u2 = (float)rand()/RAND_MAX; arr[i] = scale * sqrtf(-2.0f * logf(u1 + 1e-9f)) * cosf(2.0f * M_PI * u2); } } while(0)
    #define INIT_CONSTANT(arr, len, val) do { for(int i=0; i<len; i++) arr[i] = val; } while(0)

    INIT_NORMAL(h_w.token_embedding_table, c.vocab_size * c.dim);
    for (int l = 0; l < c.n_layers; l++) {
        INIT_CONSTANT(h_w.rms_att_weight[l], c.dim, 1.0f);
        INIT_NORMAL(h_w.wq[l], c.dim * c.dim);
        INIT_NORMAL(h_w.wk[l], c.dim * c.dim);
        INIT_NORMAL(h_w.wv[l], c.dim * c.dim);
        INIT_NORMAL(h_w.wo[l], c.dim * c.dim);
        INIT_CONSTANT(h_w.rms_ffn_weight[l], c.dim, 1.0f);
        INIT_NORMAL(h_w.w1[l], c.hidden_dim * c.dim);
        INIT_NORMAL(h_w.w2[l], c.dim * c.hidden_dim);
        INIT_NORMAL(h_w.w3[l], c.hidden_dim * c.dim);
    }
    INIT_CONSTANT(h_w.rms_final_weight, c.dim, 1.0f);
    INIT_NORMAL(h_w.wcls, c.vocab_size * c.dim);

    // Device Parameters & Gradients allocation
    Weights d_w;
    Gradients d_g;
    Weights d_opt_m, d_opt_v;

    size_t embed_size = c.vocab_size * c.dim * sizeof(float);
    size_t proj_size = c.dim * c.dim * sizeof(float);
    size_t ffn_size = c.hidden_dim * c.dim * sizeof(float);

    // Malloc Device
    CHECK_CUDA(cudaMalloc(&d_w.token_embedding_table, embed_size));
    CHECK_CUDA(cudaMalloc(&d_g.token_embedding_table, embed_size));
    CHECK_CUDA(cudaMalloc(&d_opt_m.token_embedding_table, embed_size));
    CHECK_CUDA(cudaMalloc(&d_opt_v.token_embedding_table, embed_size));
    CHECK_CUDA(cudaMemset(d_opt_m.token_embedding_table, 0, embed_size));
    CHECK_CUDA(cudaMemset(d_opt_v.token_embedding_table, 0, embed_size));

    CHECK_CUDA(cudaMemcpy(d_w.token_embedding_table, h_w.token_embedding_table, embed_size, cudaMemcpyHostToDevice));

    d_w.rms_att_weight = (float**)malloc(c.n_layers * sizeof(float*));
    d_w.wq = (float**)malloc(c.n_layers * sizeof(float*));
    d_w.wk = (float**)malloc(c.n_layers * sizeof(float*));
    d_w.wv = (float**)malloc(c.n_layers * sizeof(float*));
    d_w.wo = (float**)malloc(c.n_layers * sizeof(float*));
    d_w.rms_ffn_weight = (float**)malloc(c.n_layers * sizeof(float*));
    d_w.w1 = (float**)malloc(c.n_layers * sizeof(float*));
    d_w.w2 = (float**)malloc(c.n_layers * sizeof(float*));
    d_w.w3 = (float**)malloc(c.n_layers * sizeof(float*));

    d_g.rms_att_weight = (float**)malloc(c.n_layers * sizeof(float*));
    d_g.wq = (float**)malloc(c.n_layers * sizeof(float*));
    d_g.wk = (float**)malloc(c.n_layers * sizeof(float*));
    d_g.wv = (float**)malloc(c.n_layers * sizeof(float*));
    d_g.wo = (float**)malloc(c.n_layers * sizeof(float*));
    d_g.rms_ffn_weight = (float**)malloc(c.n_layers * sizeof(float*));
    d_g.w1 = (float**)malloc(c.n_layers * sizeof(float*));
    d_g.w2 = (float**)malloc(c.n_layers * sizeof(float*));
    d_g.w3 = (float**)malloc(c.n_layers * sizeof(float*));

    d_opt_m.rms_att_weight = (float**)malloc(c.n_layers * sizeof(float*));
    d_opt_m.wq = (float**)malloc(c.n_layers * sizeof(float*));
    d_opt_m.wk = (float**)malloc(c.n_layers * sizeof(float*));
    d_opt_m.wv = (float**)malloc(c.n_layers * sizeof(float*));
    d_opt_m.wo = (float**)malloc(c.n_layers * sizeof(float*));
    d_opt_m.rms_ffn_weight = (float**)malloc(c.n_layers * sizeof(float*));
    d_opt_m.w1 = (float**)malloc(c.n_layers * sizeof(float*));
    d_opt_m.w2 = (float**)malloc(c.n_layers * sizeof(float*));
    d_opt_m.w3 = (float**)malloc(c.n_layers * sizeof(float*));

    d_opt_v.rms_att_weight = (float**)malloc(c.n_layers * sizeof(float*));
    d_opt_v.wq = (float**)malloc(c.n_layers * sizeof(float*));
    d_opt_v.wk = (float**)malloc(c.n_layers * sizeof(float*));
    d_opt_v.wv = (float**)malloc(c.n_layers * sizeof(float*));
    d_opt_v.wo = (float**)malloc(c.n_layers * sizeof(float*));
    d_opt_v.rms_ffn_weight = (float**)malloc(c.n_layers * sizeof(float*));
    d_opt_v.w1 = (float**)malloc(c.n_layers * sizeof(float*));
    d_opt_v.w2 = (float**)malloc(c.n_layers * sizeof(float*));
    d_opt_v.w3 = (float**)malloc(c.n_layers * sizeof(float*));

    for (int l = 0; l < c.n_layers; l++) {
        CHECK_CUDA(cudaMalloc(&d_w.rms_att_weight[l], c.dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_w.wq[l], proj_size));
        CHECK_CUDA(cudaMalloc(&d_w.wk[l], proj_size));
        CHECK_CUDA(cudaMalloc(&d_w.wv[l], proj_size));
        CHECK_CUDA(cudaMalloc(&d_w.wo[l], proj_size));
        CHECK_CUDA(cudaMalloc(&d_w.rms_ffn_weight[l], c.dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_w.w1[l], ffn_size));
        CHECK_CUDA(cudaMalloc(&d_w.w2[l], ffn_size));
        CHECK_CUDA(cudaMalloc(&d_w.w3[l], ffn_size));

        CHECK_CUDA(cudaMemcpy(d_w.rms_att_weight[l], h_w.rms_att_weight[l], c.dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_w.wq[l], h_w.wq[l], proj_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_w.wk[l], h_w.wk[l], proj_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_w.wv[l], h_w.wv[l], proj_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_w.wo[l], h_w.wo[l], proj_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_w.rms_ffn_weight[l], h_w.rms_ffn_weight[l], c.dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_w.w1[l], h_w.w1[l], ffn_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_w.w2[l], h_w.w2[l], ffn_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_w.w3[l], h_w.w3[l], ffn_size, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMalloc(&d_g.rms_att_weight[l], c.dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_g.wq[l], proj_size));
        CHECK_CUDA(cudaMalloc(&d_g.wk[l], proj_size));
        CHECK_CUDA(cudaMalloc(&d_g.wv[l], proj_size));
        CHECK_CUDA(cudaMalloc(&d_g.wo[l], proj_size));
        CHECK_CUDA(cudaMalloc(&d_g.rms_ffn_weight[l], c.dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_g.w1[l], ffn_size));
        CHECK_CUDA(cudaMalloc(&d_g.w2[l], ffn_size));
        CHECK_CUDA(cudaMalloc(&d_g.w3[l], ffn_size));

        CHECK_CUDA(cudaMalloc(&d_opt_m.rms_att_weight[l], c.dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_opt_m.wq[l], proj_size));
        CHECK_CUDA(cudaMalloc(&d_opt_m.wk[l], proj_size));
        CHECK_CUDA(cudaMalloc(&d_opt_m.wv[l], proj_size));
        CHECK_CUDA(cudaMalloc(&d_opt_m.wo[l], proj_size));
        CHECK_CUDA(cudaMalloc(&d_opt_m.rms_ffn_weight[l], c.dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_opt_m.w1[l], ffn_size));
        CHECK_CUDA(cudaMalloc(&d_opt_m.w2[l], ffn_size));
        CHECK_CUDA(cudaMalloc(&d_opt_m.w3[l], ffn_size));

        CHECK_CUDA(cudaMemset(d_opt_m.rms_att_weight[l], 0, c.dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_opt_m.wq[l], 0, proj_size));
        CHECK_CUDA(cudaMemset(d_opt_m.wk[l], 0, proj_size));
        CHECK_CUDA(cudaMemset(d_opt_m.wv[l], 0, proj_size));
        CHECK_CUDA(cudaMemset(d_opt_m.wo[l], 0, proj_size));
        CHECK_CUDA(cudaMemset(d_opt_m.rms_ffn_weight[l], 0, c.dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_opt_m.w1[l], 0, ffn_size));
        CHECK_CUDA(cudaMemset(d_opt_m.w2[l], 0, ffn_size));
        CHECK_CUDA(cudaMemset(d_opt_m.w3[l], 0, ffn_size));

        CHECK_CUDA(cudaMalloc(&d_opt_v.rms_att_weight[l], c.dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_opt_v.wq[l], proj_size));
        CHECK_CUDA(cudaMalloc(&d_opt_v.wk[l], proj_size));
        CHECK_CUDA(cudaMalloc(&d_opt_v.wv[l], proj_size));
        CHECK_CUDA(cudaMalloc(&d_opt_v.wo[l], proj_size));
        CHECK_CUDA(cudaMalloc(&d_opt_v.rms_ffn_weight[l], c.dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_opt_v.w1[l], ffn_size));
        CHECK_CUDA(cudaMalloc(&d_opt_v.w2[l], ffn_size));
        CHECK_CUDA(cudaMalloc(&d_opt_v.w3[l], ffn_size));

        CHECK_CUDA(cudaMemset(d_opt_v.rms_att_weight[l], 0, c.dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_opt_v.wq[l], 0, proj_size));
        CHECK_CUDA(cudaMemset(d_opt_v.wk[l], 0, proj_size));
        CHECK_CUDA(cudaMemset(d_opt_v.wv[l], 0, proj_size));
        CHECK_CUDA(cudaMemset(d_opt_v.wo[l], 0, proj_size));
        CHECK_CUDA(cudaMemset(d_opt_v.rms_ffn_weight[l], 0, c.dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_opt_v.w1[l], 0, ffn_size));
        CHECK_CUDA(cudaMemset(d_opt_v.w2[l], 0, ffn_size));
        CHECK_CUDA(cudaMemset(d_opt_v.w3[l], 0, ffn_size));
    }

    CHECK_CUDA(cudaMalloc(&d_w.rms_final_weight, c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w.wcls, c.vocab_size * c.dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_w.rms_final_weight, h_w.rms_final_weight, c.dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w.wcls, h_w.wcls, c.vocab_size * c.dim * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&d_g.rms_final_weight, c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g.wcls, c.vocab_size * c.dim * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_opt_m.rms_final_weight, c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_opt_m.wcls, c.vocab_size * c.dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_opt_m.rms_final_weight, 0, c.dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_opt_m.wcls, 0, c.vocab_size * c.dim * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_opt_v.rms_final_weight, c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_opt_v.wcls, c.vocab_size * c.dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_opt_v.rms_final_weight, 0, c.dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_opt_v.wcls, 0, c.vocab_size * c.dim * sizeof(float)));

    // Allocate Device Batch Activations
    BatchActivations d_act;
    CHECK_CUDA(cudaMalloc(&d_act.x, (c.n_layers + 1) * B * T * c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_act.x_final_norm, B * T * c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_act.logits, B * T * c.vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_act.probs, B * T * c.vocab_size * sizeof(float)));
    d_act.l = (LayerActivations*)malloc(c.n_layers * sizeof(LayerActivations));

    for (int l = 0; l < c.n_layers; l++) {
        CHECK_CUDA(cudaMalloc(&d_act.l[l].xb, B * T * c.dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_act.l[l].q, B * T * c.dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_act.l[l].k, B * T * c.dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_act.l[l].v, B * T * c.dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_act.l[l].q_rot, B * T * c.dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_act.l[l].k_rot, B * T * c.dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_act.l[l].att, B * c.n_heads * T * T * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_act.l[l].xb2, B * T * c.dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_act.l[l].ffn_xb, B * T * c.dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_act.l[l].hb, B * T * c.hidden_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_act.l[l].hb2, B * T * c.hidden_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_act.l[l].hb_act, B * T * c.hidden_dim * sizeof(float)));
    }

    // Allocate Device Gradients activation buffers
    CHECK_CUDA(cudaMalloc(&d_g.d_x, (c.n_layers + 1) * B * T * c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g.d_x_final_norm, B * T * c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g.d_logits, B * T * c.vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g.d_xb, B * T * c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g.d_q, B * T * c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g.d_k, B * T * c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g.d_v, B * T * c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g.d_q_rot, B * T * c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g.d_k_rot, B * T * c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g.d_att, B * c.n_heads * T * T * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g.d_xb2, B * T * c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g.d_ffn_xb, B * T * c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g.d_hb, B * T * c.hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g.d_hb2, B * T * c.hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g.d_hb_act, B * T * c.hidden_dim * sizeof(float)));

    // Allocate Device normalization caches
    float* d_cache_rstd;
    CHECK_CUDA(cudaMalloc(&d_cache_rstd, (c.n_layers * 2 + 1) * B * T * sizeof(float)));

    // Prepare synthetic data
    int* h_inputs = (int*)malloc(B * T * sizeof(int));
    int* h_targets = (int*)malloc(B * T * sizeof(int));

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            h_inputs[b * T + t] = ((b * 5 + t) % 100) + 1;
            h_targets[b * T + t] = ((b * 5 + t + 1) % 100) + 1;
        }
    }

    int* d_inputs;
    int* d_targets;
    CHECK_CUDA(cudaMalloc(&d_inputs, B * T * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_targets, B * T * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_inputs, h_inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_targets, h_targets, B * T * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate loss tracking buffers
    float* d_loss_accum;
    float* h_loss_accum = (float*)malloc(B * T * sizeof(float));
    CHECK_CUDA(cudaMalloc(&d_loss_accum, B * T * sizeof(float)));

    // Training Parameters
    float lr = 1e-2f;
    float wd = 1e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;

    printf("Starting GPU training optimization loop...\n");

    // 40 training steps to verify loss minimization
    for (int step = 1; step <= 40; step++) {
        // Zero all gradients
        CHECK_CUDA(cudaMemset(d_g.token_embedding_table, 0, embed_size));
        for (int l = 0; l < c.n_layers; l++) {
            CHECK_CUDA(cudaMemset(d_g.rms_att_weight[l], 0, c.dim * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_g.wq[l], 0, proj_size));
            CHECK_CUDA(cudaMemset(d_g.wk[l], 0, proj_size));
            CHECK_CUDA(cudaMemset(d_g.wv[l], 0, proj_size));
            CHECK_CUDA(cudaMemset(d_g.wo[l], 0, proj_size));
            CHECK_CUDA(cudaMemset(d_g.rms_ffn_weight[l], 0, c.dim * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_g.w1[l], 0, ffn_size));
            CHECK_CUDA(cudaMemset(d_g.w2[l], 0, ffn_size));
            CHECK_CUDA(cudaMemset(d_g.w3[l], 0, ffn_size));
        }
        CHECK_CUDA(cudaMemset(d_g.rms_final_weight, 0, c.dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_g.wcls, 0, c.vocab_size * c.dim * sizeof(float)));

        // 1. GPU Forward Pass
        model_forward_gpu(handle, &c, &d_w, &d_act, d_inputs, B, T, d_cache_rstd);

        // 2. Loss computation
        cross_entropy_forward_kernel<<<(B * T + 255) / 256, 256>>>(d_act.probs, d_act.logits, d_targets, d_loss_accum, B, T, c.vocab_size);
        CHECK_CUDA(cudaGetLastError());

        if (step % 5 == 0 || step == 1) {
            CHECK_CUDA(cudaMemcpy(h_loss_accum, d_loss_accum, B * T * sizeof(float), cudaMemcpyDeviceToHost));
            float mean_loss = 0.0f;
            int active_count = 0;
            for (int i = 0; i < B * T; i++) {
                if (h_targets[i] >= 0 && h_targets[i] < c.vocab_size) {
                    mean_loss += h_loss_accum[i];
                    active_count++;
                }
            }
            mean_loss /= (active_count > 0 ? active_count : 1);
            printf("Step %d | Loss: %.5f\n", step, mean_loss);
        }

        // 3. GPU Backward Pass
        int active_count = B * T; // all synthetic tokens are active
        cross_entropy_backward_kernel<<<(B * T * c.vocab_size + 255) / 256, 256>>>(d_g.d_logits, d_act.probs, d_targets, B, T, c.vocab_size, active_count);
        CHECK_CUDA(cudaGetLastError());

        model_backward_gpu(handle, &c, &d_w, &d_g, &d_act, d_inputs, B, T, d_cache_rstd);

        // 4. AdamW Parameter updates on GPU
        #define UPDATE_GPU(w_dev, g_dev, m_dev, v_dev, s_val) adamw_update_kernel<<<(s_val + 255)/256, 256>>>(w_dev, g_dev, m_dev, v_dev, s_val, lr, wd, beta1, beta2, eps, step)
        
        UPDATE_GPU(d_w.token_embedding_table, d_g.token_embedding_table, d_opt_m.token_embedding_table, d_opt_v.token_embedding_table, c.vocab_size * c.dim);
        for (int l = 0; l < c.n_layers; l++) {
            UPDATE_GPU(d_w.rms_att_weight[l], d_g.rms_att_weight[l], d_opt_m.rms_att_weight[l], d_opt_v.rms_att_weight[l], c.dim);
            UPDATE_GPU(d_w.wq[l], d_g.wq[l], d_opt_m.wq[l], d_opt_v.wq[l], c.dim * c.dim);
            UPDATE_GPU(d_w.wk[l], d_g.wk[l], d_opt_m.wk[l], d_opt_v.wk[l], c.dim * c.dim);
            UPDATE_GPU(d_w.wv[l], d_g.wv[l], d_opt_m.wv[l], d_opt_v.wv[l], c.dim * c.dim);
            UPDATE_GPU(d_w.wo[l], d_g.wo[l], d_opt_m.wo[l], d_opt_v.wo[l], c.dim * c.dim);
            UPDATE_GPU(d_w.rms_ffn_weight[l], d_g.rms_ffn_weight[l], d_opt_m.rms_ffn_weight[l], d_opt_v.rms_ffn_weight[l], c.dim);
            UPDATE_GPU(d_w.w1[l], d_g.w1[l], d_opt_m.w1[l], d_opt_v.w1[l], c.hidden_dim * c.dim);
            UPDATE_GPU(d_w.w2[l], d_g.w2[l], d_opt_m.w2[l], d_opt_v.w2[l], c.dim * c.hidden_dim);
            UPDATE_GPU(d_w.w3[l], d_g.w3[l], d_opt_m.w3[l], d_opt_v.w3[l], c.hidden_dim * c.dim);
        }
        UPDATE_GPU(d_w.rms_final_weight, d_g.rms_final_weight, d_opt_m.rms_final_weight, d_opt_v.rms_final_weight, c.dim);
        UPDATE_GPU(d_w.wcls, d_g.wcls, d_opt_m.wcls, d_opt_v.wcls, c.vocab_size * c.dim);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    printf("GPU Training completed successfully!\n");

    // Clean up host memory
    free(h_w.token_embedding_table);
    for (int l = 0; l < c.n_layers; l++) {
        free(h_w.rms_att_weight[l]);
        free(h_w.wq[l]); free(h_w.wk[l]); free(h_w.wv[l]); free(h_w.wo[l]);
        free(h_w.rms_ffn_weight[l]);
        free(h_w.w1[l]); free(h_w.w2[l]); free(h_w.w3[l]);
    }
    free(h_w.rms_final_weight); free(h_w.wcls);
    free(h_inputs); free(h_targets); free(h_loss_accum);

    // Clean up device memory
    CHECK_CUDA(cudaFree(d_w.token_embedding_table));
    CHECK_CUDA(cudaFree(d_g.token_embedding_table));
    CHECK_CUDA(cudaFree(d_opt_m.token_embedding_table));
    CHECK_CUDA(cudaFree(d_opt_v.token_embedding_table));

    for (int l = 0; l < c.n_layers; l++) {
        CHECK_CUDA(cudaFree(d_w.rms_att_weight[l]));
        CHECK_CUDA(cudaFree(d_w.wq[l])); CHECK_CUDA(cudaFree(d_w.wk[l]));
        CHECK_CUDA(cudaFree(d_w.wv[l])); CHECK_CUDA(cudaFree(d_w.wo[l]));
        CHECK_CUDA(cudaFree(d_w.rms_ffn_weight[l]));
        CHECK_CUDA(cudaFree(d_w.w1[l])); CHECK_CUDA(cudaFree(d_w.w2[l])); CHECK_CUDA(cudaFree(d_w.w3[l]));

        CHECK_CUDA(cudaFree(d_g.rms_att_weight[l]));
        CHECK_CUDA(cudaFree(d_g.wq[l])); CHECK_CUDA(cudaFree(d_g.wk[l]));
        CHECK_CUDA(cudaFree(d_g.wv[l])); CHECK_CUDA(cudaFree(d_g.wo[l]));
        CHECK_CUDA(cudaFree(d_g.rms_ffn_weight[l]));
        CHECK_CUDA(cudaFree(d_g.w1[l])); CHECK_CUDA(cudaFree(d_g.w2[l])); CHECK_CUDA(cudaFree(d_g.w3[l]));

        CHECK_CUDA(cudaFree(d_opt_m.rms_att_weight[l]));
        CHECK_CUDA(cudaFree(d_opt_m.wq[l])); CHECK_CUDA(cudaFree(d_opt_m.wk[l]));
        CHECK_CUDA(cudaFree(d_opt_m.wv[l])); CHECK_CUDA(cudaFree(d_opt_m.wo[l]));
        CHECK_CUDA(cudaFree(d_opt_m.rms_ffn_weight[l]));
        CHECK_CUDA(cudaFree(d_opt_m.w1[l])); CHECK_CUDA(cudaFree(d_opt_m.w2[l])); CHECK_CUDA(cudaFree(d_opt_m.w3[l]));

        CHECK_CUDA(cudaFree(d_opt_v.rms_att_weight[l]));
        CHECK_CUDA(cudaFree(d_opt_v.wq[l])); CHECK_CUDA(cudaFree(d_opt_v.wk[l]));
        CHECK_CUDA(cudaFree(d_opt_v.wv[l])); CHECK_CUDA(cudaFree(d_opt_v.wo[l]));
        CHECK_CUDA(cudaFree(d_opt_v.rms_ffn_weight[l]));
        CHECK_CUDA(cudaFree(d_opt_v.w1[l])); CHECK_CUDA(cudaFree(d_opt_v.w2[l])); CHECK_CUDA(cudaFree(d_opt_v.w3[l]));

        CHECK_CUDA(cudaFree(d_act.l[l].xb));
        CHECK_CUDA(cudaFree(d_act.l[l].q)); CHECK_CUDA(cudaFree(d_act.l[l].k)); CHECK_CUDA(cudaFree(d_act.l[l].v));
        CHECK_CUDA(cudaFree(d_act.l[l].q_rot)); CHECK_CUDA(cudaFree(d_act.l[l].k_rot));
        CHECK_CUDA(cudaFree(d_act.l[l].att)); CHECK_CUDA(cudaFree(d_act.l[l].xb2));
        CHECK_CUDA(cudaFree(d_act.l[l].ffn_xb));
        CHECK_CUDA(cudaFree(d_act.l[l].hb)); CHECK_CUDA(cudaFree(d_act.l[l].hb2)); CHECK_CUDA(cudaFree(d_act.l[l].hb_act));
    }

    CHECK_CUDA(cudaFree(d_w.rms_final_weight)); CHECK_CUDA(cudaFree(d_w.wcls));
    CHECK_CUDA(cudaFree(d_g.rms_final_weight)); CHECK_CUDA(cudaFree(d_g.wcls));
    CHECK_CUDA(cudaFree(d_opt_m.rms_final_weight)); CHECK_CUDA(cudaFree(d_opt_m.wcls));
    CHECK_CUDA(cudaFree(d_opt_v.rms_final_weight)); CHECK_CUDA(cudaFree(d_opt_v.wcls));

    CHECK_CUDA(cudaFree(d_act.x)); CHECK_CUDA(cudaFree(d_act.x_final_norm)); CHECK_CUDA(cudaFree(d_act.logits)); CHECK_CUDA(cudaFree(d_act.probs));
    free(d_act.l);

    CHECK_CUDA(cudaFree(d_g.d_x)); CHECK_CUDA(cudaFree(d_g.d_x_final_norm)); CHECK_CUDA(cudaFree(d_g.d_logits));
    CHECK_CUDA(cudaFree(d_g.d_xb)); CHECK_CUDA(cudaFree(d_g.d_q)); CHECK_CUDA(cudaFree(d_g.d_k)); CHECK_CUDA(cudaFree(d_g.d_v));
    CHECK_CUDA(cudaFree(d_g.d_q_rot)); CHECK_CUDA(cudaFree(d_g.d_k_rot)); CHECK_CUDA(cudaFree(d_g.d_att)); CHECK_CUDA(cudaFree(d_g.d_xb2));
    CHECK_CUDA(cudaFree(d_g.d_ffn_xb)); CHECK_CUDA(cudaFree(d_g.d_hb)); CHECK_CUDA(cudaFree(d_g.d_hb2)); CHECK_CUDA(cudaFree(d_g.d_hb_act));

    CHECK_CUDA(cudaFree(d_cache_rstd));
    CHECK_CUDA(cudaFree(d_inputs)); CHECK_CUDA(cudaFree(d_targets)); CHECK_CUDA(cudaFree(d_loss_accum));

    free(d_w.rms_att_weight); free(d_w.wq); free(d_w.wk); free(d_w.wv); free(d_w.wo);
    free(d_w.rms_ffn_weight); free(d_w.w1); free(d_w.w2); free(d_w.w3);

    free(d_g.rms_att_weight); free(d_g.wq); free(d_g.wk); free(d_g.wv); free(d_g.wo);
    free(d_g.rms_ffn_weight); free(d_g.w1); free(d_g.w2); free(d_g.w3);

    free(d_opt_m.rms_att_weight); free(d_opt_m.wq); free(d_opt_m.wk); free(d_opt_m.wv); free(d_opt_m.wo);
    free(d_opt_m.rms_ffn_weight); free(d_opt_m.w1); free(d_opt_m.w2); free(d_opt_m.w3);

    free(d_opt_v.rms_att_weight); free(d_opt_v.wq); free(d_opt_v.wk); free(d_opt_v.wv); free(d_opt_v.wo);
    free(d_opt_v.rms_ffn_weight); free(d_opt_v.w1); free(d_opt_v.w2); free(d_opt_v.w3);

    cublasDestroy(handle);

    return 0;
}
