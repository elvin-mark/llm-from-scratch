#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef USE_BLAS
#include <cblas.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

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

// Weights Structure
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

// Activations for a batch of size B and sequence length T
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
    float* x;            // [n_layers + 1][B, T, dim] (0 is embeddings, l+1 is layer output)
    LayerActivations* l; // array of size n_layers
    float* x_final_norm; // [B, T, dim]
    float* logits;       // [B, T, vocab_size]
    float* probs;        // [B, T, vocab_size]
} BatchActivations;

// Gradients Structure
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

// -----------------------------------------------------------------------------
// Helper Memory Allocators
void* malloc_align(size_t size) {
    void* ptr = NULL;
#if defined(_MSC_VER)
    ptr = _aligned_malloc(size, 64);
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    ptr = aligned_alloc(64, (size + 63) & ~63);
#else
    posix_memalign(&ptr, 64, size);
#endif
    return ptr;
}

void free_align(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Optimized MatMul Forward: out = X * W^T
// X is shape [M, N], W is shape [D, N] -> out is shape [M, D]
void matmul_forward(float* out, float* x, float* w, int M, int N, int D) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, D, N, 1.0f, x, N, w, N, 0.0f, out, D);
#else
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < D; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += x[i * N + k] * w[j * N + k];
            }
            out[i * D + j] = sum;
        }
    }
#endif
}

// Optimized MatMul Backward Input: dx = dy * W
// dy is shape [M, D], W is shape [D, N] -> dx is shape [M, N]
void matmul_backward_input(float* dx, float* dy, float* w, int M, int N, int D) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, D, 1.0f, dy, D, w, N, 0.0f, dx, N);
#else
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < N; k++) {
            float sum = 0.0f;
            for (int j = 0; j < D; j++) {
                sum += dy[i * D + j] * w[j * N + k];
            }
            dx[i * N + k] = sum;
        }
    }
#endif
}

// Optimized MatMul Backward Weights: dw += dy^T * X
// dy is shape [M, D], X is shape [M, N] -> dw is shape [D, N]
void matmul_backward_weights(float* dw, float* dy, float* x, int M, int N, int D) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, D, N, M, 1.0f, dy, D, x, N, 1.0f, dw, N);
#else
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < D; j++) {
        for (int k = 0; k < N; k++) {
            float sum = 0.0f;
            for (int i = 0; i < M; i++) {
                sum += dy[i * D + j] * x[i * N + k];
            }
            dw[j * N + k] += sum;
        }
    }
#endif
}

// RMSNorm Forward
void rmsnorm_forward(float* o, float* x, float* weight, int size, int M, float* cache_std) {
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        float ss = 0.0f;
        for (int j = 0; j < size; j++) {
            ss += x[i * size + j] * x[i * size + j];
        }
        ss = ss / size + 1e-6f;
        float rstd = 1.0f / sqrtf(ss);
        cache_std[i] = rstd;
        for (int j = 0; j < size; j++) {
            o[i * size + j] = weight[j] * x[i * size + j] * rstd;
        }
    }
}

// RMSNorm Backward
void rmsnorm_backward(float* dx, float* dw, float* dy, float* x, float* weight, int size, int M, float* cache_std) {
    // 1. Parameter gradient accumulation
    #pragma omp parallel for
    for (int j = 0; j < size; j++) {
        float sum = 0.0f;
        for (int i = 0; i < M; i++) {
            sum += dy[i * size + j] * x[i * size + j] * cache_std[i];
        }
        dw[j] += sum;
    }

    // 2. Input gradient computation
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        float rstd = cache_std[i];
        float sum_dy_w_x = 0.0f;
        for (int j = 0; j < size; j++) {
            sum_dy_w_x += dy[i * size + j] * weight[j] * x[i * size + j];
        }
        float factor = sum_dy_w_x * rstd * rstd * rstd / size;
        for (int j = 0; j < size; j++) {
            dx[i * size + j] = weight[j] * dy[i * size + j] * rstd - x[i * size + j] * factor;
        }
    }
}

// Softmax Forward (causal masking applied)
void softmax_forward(float* att, int B, int n_heads, int T) {
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < n_heads; h++) {
            for (int t = 0; t < T; t++) {
                float* head_att = att + b * n_heads * T * T + h * T * T + t * T;
                float max_val = -1e30f;
                for (int col = 0; col <= t; col++) {
                    if (head_att[col] > max_val) max_val = head_att[col];
                }
                float sum = 0.0f;
                for (int col = 0; col <= t; col++) {
                    head_att[col] = expf(head_att[col] - max_val);
                    sum += head_att[col];
                }
                for (int col = 0; col <= t; col++) {
                    head_att[col] /= sum;
                }
                for (int col = t + 1; col < T; col++) {
                    head_att[col] = 0.0f; // Causal mask
                }
            }
        }
    }
}

// Softmax Backward
void softmax_backward(float* d_att, float* att, float* d_s, int B, int n_heads, int T) {
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < n_heads; h++) {
            for (int t = 0; t < T; t++) {
                float* head_att = att + b * n_heads * T * T + h * T * T + t * T;
                float* head_d_s = d_s + b * n_heads * T * T + h * T * T + t * T;
                float* head_d_att = d_att + b * n_heads * T * T + h * T * T + t * T;
                
                float sum_ds_s = 0.0f;
                for (int col = 0; col <= t; col++) {
                    sum_ds_s += head_d_s[col] * head_att[col];
                }
                for (int col = 0; col <= t; col++) {
                    head_d_att[col] = head_att[col] * (head_d_s[col] - sum_ds_s);
                }
                for (int col = t + 1; col < T; col++) {
                    head_d_att[col] = 0.0f;
                }
            }
        }
    }
}

// RoPE Forward
void rope_forward(float* q, float* k, int B, int T, int dim, int head_size, int n_heads) {
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < n_heads; h++) {
                for (int i = 0; i < head_size; i += 2) {
                    int global_idx = b * T * dim + t * dim + h * head_size + i;
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
        }
    }
}

// RoPE Backward (Adjoint transposition)
void rope_backward(float* dq, float* dk, float* dq_rot, float* dk_rot, int B, int T, int dim, int head_size, int n_heads) {
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < n_heads; h++) {
                for (int i = 0; i < head_size; i += 2) {
                    int global_idx = b * T * dim + t * dim + h * head_size + i;
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
        }
    }
}

// SwiGLU Forward
void swiglu_forward(float* hb_act, float* hb, float* hb2, int total_elements) {
    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) {
        float val = hb[i];
        float silu = val * (1.0f / (1.0f + expf(-val))); // Silu(hb)
        hb_act[i] = silu * hb2[i];                      // Silu(hb) * hb3
    }
}

// SwiGLU Backward
void swiglu_backward(float* d_hb, float* d_hb2, float* d_hb_act, float* hb, float* hb2, int total_elements) {
    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) {
        float val = hb[i];
        float sig = 1.0f / (1.0f + expf(-val));
        float silu = val * sig;
        float d_silu = sig * (1.0f + val * (1.0f - sig)); // d(SiLU)/dx

        d_hb2[i] = d_hb_act[i] * silu;
        d_hb[i]  = d_hb_act[i] * hb2[i] * d_silu;
    }
}

// Cross Entropy Loss & Probs Forward
float cross_entropy_forward(float* probs, float* logits, int* targets, int B, int T, int V) {
    float total_loss = 0.0f;
    int count = 0;

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* logit_row = logits + (b * T + t) * V;
            float* prob_row = probs + (b * T + t) * V;

            // Softmax
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

            // Loss accumulation
            int target = targets[b * T + t];
            if (target >= 0 && target < V) {
                total_loss += -logf(prob_row[target] + 1e-9f);
                count++;
            }
        }
    }
    return count > 0 ? (total_loss / count) : 0.0f;
}

// Cross Entropy Loss Backward
void cross_entropy_backward(float* d_logits, float* probs, int* targets, int B, int T, int V) {
    int count = 0;
    for (int i = 0; i < B * T; i++) {
        if (targets[i] >= 0 && targets[i] < V) count++;
    }
    float scale = 1.0f / (count > 0 ? count : 1);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < B * T; i++) {
        for (int v = 0; v < V; v++) {
            int target = targets[i];
            float p = probs[i * V + v];
            if (target >= 0 && target < V) {
                d_logits[i * V + v] = scale * (p - (v == target ? 1.0f : 0.0f));
            } else {
                d_logits[i * V + v] = 0.0f; // ignored tokens
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Forward Pass Core
void model_forward(Config* c, Weights* w, BatchActivations* act, int* inputs, int B, int T, float* cache_rstd) {
    int dim = c->dim;
    int hidden_dim = c->hidden_dim;
    int n_heads = c->n_heads;
    int head_size = dim / n_heads;

    // 1. Embeddings copy
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int tok = inputs[b * T + t];
            memcpy(act->x + (b * T + t) * dim, w->token_embedding_table + tok * dim, dim * sizeof(float));
        }
    }

    // 2. Transformer layers forward
    for (int l = 0; l < c->n_layers; l++) {
        LayerActivations* la = &act->l[l];
        float* layer_in = act->x + l * B * T * dim;
        float* layer_out = act->x + (l + 1) * B * T * dim;

        // RMSNorm (Attention Input)
        rmsnorm_forward(la->xb, layer_in, w->rms_att_weight[l], dim, B * T, cache_rstd + l * 2 * B * T);

        // Attention Projections
        matmul_forward(la->q, la->xb, w->wq[l], B * T, dim, dim);
        matmul_forward(la->k, la->xb, w->wk[l], B * T, dim, dim);
        matmul_forward(la->v, la->xb, w->wv[l], B * T, dim, dim);

        // Apply RoPE positional rotation
        memcpy(la->q_rot, la->q, B * T * dim * sizeof(float));
        memcpy(la->k_rot, la->k, B * T * dim * sizeof(float));
        rope_forward(la->q_rot, la->k_rot, B, T, dim, head_size, n_heads);

        // Batch Attention Computation
        #pragma omp parallel for collapse(3)
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < n_heads; h++) {
                for (int t = 0; t < T; t++) {
                    float* q_vec = la->q_rot + b * T * dim + t * dim + h * head_size;
                    float* att_row = la->att + b * n_heads * T * T + h * T * T + t * T;
                    for (int tau = 0; tau < T; tau++) {
                        float* k_vec = la->k_rot + b * T * dim + tau * dim + h * head_size;
                        float score = 0.0f;
                        for (int i = 0; i < head_size; i++) {
                            score += q_vec[i] * k_vec[i];
                        }
                        att_row[tau] = score / sqrtf((float)head_size);
                    }
                }
            }
        }

        // Apply Causal Softmax Mask
        softmax_forward(la->att, B, n_heads, T);

        // Weighted Attention sum to get post-attention embeddings (xb2)
        #pragma omp parallel for collapse(3)
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < n_heads; h++) {
                for (int t = 0; t < T; t++) {
                    float* att_row = la->att + b * n_heads * T * T + h * T * T + t * T;
                    float* out_vec = la->xb2 + b * T * dim + t * dim + h * head_size;
                    memset(out_vec, 0, head_size * sizeof(float));
                    for (int tau = 0; tau <= t; tau++) {
                        float* v_vec = la->v + b * T * dim + tau * dim + h * head_size;
                        float score = att_row[tau];
                        for (int i = 0; i < head_size; i++) {
                            out_vec[i] += score * v_vec[i];
                        }
                    }
                }
            }
        }

        // Wo projection and residual addition
        matmul_forward(layer_out, la->xb2, w->wo[l], B * T, dim, dim);
        #pragma omp parallel for
        for (int i = 0; i < B * T * dim; i++) {
            layer_out[i] += layer_in[i];
        }

        // FFN norm & MLP
        float* ffn_in = layer_out; // residual update is current layer output
        rmsnorm_forward(la->ffn_xb, ffn_in, w->rms_ffn_weight[l], dim, B * T, cache_rstd + (l * 2 + 1) * B * T);

        matmul_forward(la->hb, la->ffn_xb, w->w1[l], B * T, dim, hidden_dim);
        matmul_forward(la->hb2, la->ffn_xb, w->w3[l], B * T, dim, hidden_dim);

        swiglu_forward(la->hb_act, la->hb, la->hb2, B * T * hidden_dim);

        // w2 projection and final block output
        float* ffn_out = la->hb; // reuse hb buffer
        matmul_forward(ffn_out, la->hb_act, w->w2[l], B * T, hidden_dim, dim);

        #pragma omp parallel for
        for (int i = 0; i < B * T * dim; i++) {
            layer_out[i] += ffn_out[i];
        }
    }

    // 3. Final model output
    float* final_in = act->x + c->n_layers * B * T * dim;
    rmsnorm_forward(act->x_final_norm, final_in, w->rms_final_weight, dim, B * T, cache_rstd + c->n_layers * 2 * B * T);
    matmul_forward(act->logits, act->x_final_norm, w->wcls, B * T, dim, c->vocab_size);
}

// -----------------------------------------------------------------------------
// Backward Pass Core
void model_backward(Config* c, Weights* w, Gradients* g, BatchActivations* act, int* inputs, int B, int T, float* cache_rstd) {
    int dim = c->dim;
    int hidden_dim = c->hidden_dim;
    int n_heads = c->n_heads;
    int head_size = dim / n_heads;

    // 1. Output logits projection backward
    matmul_backward_weights(g->wcls, g->d_logits, act->x_final_norm, B * T, dim, c->vocab_size);
    matmul_backward_input(g->d_x_final_norm, g->d_logits, w->wcls, B * T, dim, c->vocab_size);

    // Final RMSNorm backward
    float* final_in = act->x + c->n_layers * B * T * dim;
    float* d_final_in = g->d_x + c->n_layers * B * T * dim;
    rmsnorm_backward(d_final_in, g->rms_final_weight, g->d_x_final_norm, final_in, w->rms_final_weight, dim, B * T, cache_rstd + c->n_layers * 2 * B * T);

    // 2. Transformer layers backward
    for (int l = c->n_layers - 1; l >= 0; l--) {
        LayerActivations* la = &act->l[l];
        float* layer_in = act->x + l * B * T * dim;
        float* d_layer_in = g->d_x + l * B * T * dim;
        float* d_layer_out = g->d_x + (l + 1) * B * T * dim;

        // FFN backward
        float* ffn_in = act->x + (l + 1) * B * T * dim; // Layer output is the FFN output destination
        float* d_ffn_out = d_layer_out;
        
        matmul_backward_weights(g->w2[l], d_ffn_out, la->hb_act, B * T, hidden_dim, dim);
        matmul_backward_input(g->d_hb_act, d_ffn_out, w->w2[l], B * T, hidden_dim, dim);

        swiglu_backward(g->d_hb, g->d_hb2, g->d_hb_act, la->hb, la->hb2, B * T * hidden_dim);

        matmul_backward_weights(g->w3[l], g->d_hb2, la->ffn_xb, B * T, dim, hidden_dim);
        matmul_backward_input(g->d_ffn_xb, g->d_hb2, w->w3[l], B * T, dim, hidden_dim);

        matmul_backward_weights(g->w1[l], g->d_hb, la->ffn_xb, B * T, dim, hidden_dim);
        // Accumulate into d_ffn_xb
        float* d_ffn_xb_tmp = g->d_hb2; // reuse hb2 buffer to avoid allocation
        matmul_backward_input(d_ffn_xb_tmp, g->d_hb, w->w1[l], B * T, dim, hidden_dim);
        #pragma omp parallel for
        for (int i = 0; i < B * T * dim; i++) {
            g->d_ffn_xb[i] += d_ffn_xb_tmp[i];
        }

        // FFN RMSNorm backward
        float* d_ffn_in = g->d_hb; // reuse hb buffer
        rmsnorm_backward(d_ffn_in, g->rms_ffn_weight[l], g->d_ffn_xb, ffn_in, w->rms_ffn_weight[l], dim, B * T, cache_rstd + (l * 2 + 1) * B * T);

        // Intermediate gradients from FFN residual path
        float* d_h1 = g->d_xb; // reuse xb buffer
        #pragma omp parallel for
        for (int i = 0; i < B * T * dim; i++) {
            d_h1[i] = d_layer_out[i] + d_ffn_in[i];
        }

        // Attention backward
        matmul_backward_weights(g->wo[l], d_h1, la->xb2, B * T, dim, dim);
        matmul_backward_input(g->d_xb2, d_h1, w->wo[l], B * T, dim, dim);

        // Value projection and Attention matrix backward
        memset(g->d_v, 0, B * T * dim * sizeof(float));
        memset(g->d_att, 0, B * n_heads * T * T * sizeof(float));

        #pragma omp parallel for collapse(3)
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < n_heads; h++) {
                for (int t = 0; t < T; t++) {
                    float* d_out_vec = g->d_xb2 + b * T * dim + t * dim + h * head_size;
                    float* att_row = la->att + b * n_heads * T * T + h * T * T + t * T;
                    float* d_att_row = g->d_att + b * n_heads * T * T + h * T * T + t * T;
                    
                    for (int tau = 0; tau <= t; tau++) {
                        float* v_vec = la->v + b * T * dim + tau * dim + h * head_size;
                        float* d_v_vec = g->d_v + b * T * dim + tau * dim + h * head_size;
                        float att_val = att_row[tau];
                        
                        for (int i = 0; i < head_size; i++) {
                            d_att_row[tau] += d_out_vec[i] * v_vec[i];
                            // Accumulate d_v
                            #pragma omp atomic
                            d_v_vec[i] += d_out_vec[i] * att_val;
                        }
                    }
                }
            }
        }

        // Softmax backward
        float* d_att_logits = g->d_q; // reuse q buffer temporarily
        softmax_backward(d_att_logits, la->att, g->d_att, B, n_heads, T);

        // Q & K dot product backward
        memset(g->d_q_rot, 0, B * T * dim * sizeof(float));
        memset(g->d_k_rot, 0, B * T * dim * sizeof(float));

        #pragma omp parallel for collapse(3)
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < n_heads; h++) {
                for (int t = 0; t < T; t++) {
                    float* q_vec = la->q_rot + b * T * dim + t * dim + h * head_size;
                    float* d_q_vec = g->d_q_rot + b * T * dim + t * dim + h * head_size;
                    float* d_att_row = d_att_logits + b * n_heads * T * T + h * T * T + t * T;
                    
                    for (int tau = 0; tau <= t; tau++) {
                        float* k_vec = la->k_rot + b * T * dim + tau * dim + h * head_size;
                        float* d_k_vec = g->d_k_rot + b * T * dim + tau * dim + h * head_size;
                        float scale_factor = d_att_row[tau] / sqrtf((float)head_size);
                        
                        for (int i = 0; i < head_size; i++) {
                            #pragma omp atomic
                            d_q_vec[i] += k_vec[i] * scale_factor;
                            #pragma omp atomic
                            d_k_vec[i] += q_vec[i] * scale_factor;
                        }
                    }
                }
            }
        }

        // RoPE backward
        rope_backward(g->d_q, g->d_k, g->d_q_rot, g->d_k_rot, B, T, dim, head_size, n_heads);

        // Attention weights backward
        matmul_backward_weights(g->wq[l], g->d_q, la->xb, B * T, dim, dim);
        matmul_backward_weights(g->wk[l], g->d_k, la->xb, B * T, dim, dim);
        matmul_backward_weights(g->wv[l], g->d_v, la->xb, B * T, dim, dim);

        // Accumulate into d_xb
        matmul_backward_input(g->d_xb, g->d_q, w->wq[l], B * T, dim, dim);
        
        float* d_xb_tmp = g->d_xb2; // reuse xb2 buffer
        matmul_backward_input(d_xb_tmp, g->d_k, w->wk[l], B * T, dim, dim);
        #pragma omp parallel for
        for (int i = 0; i < B * T * dim; i++) {
            g->d_xb[i] += d_xb_tmp[i];
        }

        matmul_backward_input(d_xb_tmp, g->d_v, w->wv[l], B * T, dim, dim);
        #pragma omp parallel for
        for (int i = 0; i < B * T * dim; i++) {
            g->d_xb[i] += d_xb_tmp[i];
        }

        // Attention RMSNorm backward
        rmsnorm_backward(d_layer_in, g->rms_att_weight[l], g->d_xb, layer_in, w->rms_att_weight[l], dim, B * T, cache_rstd + l * 2 * B * T);

        // Add back layer input residual
        #pragma omp parallel for
        for (int i = 0; i < B * T * dim; i++) {
            d_layer_in[i] += d_h1[i];
        }
    }

    // 3. Accumulate token embeddings gradient
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int token = inputs[b * T + t];
            float* d_x_emb = g->d_x + (b * T + t) * dim;
            float* grad_emb_row = g->token_embedding_table + token * dim;
            for (int i = 0; i < dim; i++) {
                #pragma omp atomic
                grad_emb_row[i] += d_x_emb[i];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// AdamW Optimizer parameter update
void adamw_update(float* w, float* g, float* m, float* v, int size, float lr, float wd, float beta1, float beta2, float eps, int step) {
    float correction1 = 1.0f - powf(beta1, step);
    float correction2 = 1.0f - powf(beta2, step);
    float step_lr = lr * sqrtf(correction2) / correction1;

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        // L2 Regularization (Weight decay)
        w[i] -= lr * wd * w[i];

        // Momentum updates
        m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];

        w[i] -= step_lr * m[i] / (sqrtf(v[i]) + eps);
    }
}

// -----------------------------------------------------------------------------
// Main execution / Synthetic Training Loop
int main() {
#ifdef _OPENMP
    printf("OpenMP initialized with %d threads.\n", omp_get_max_threads());
#endif

    // Hyperparameters
    Config c = {
        .dim = 64,
        .hidden_dim = 128,
        .n_layers = 2,
        .n_heads = 2,
        .n_kv_heads = 2,
        .vocab_size = 256,
        .max_seq_len = 32
    };

    int B = 4; // Batch size
    int T = 16; // Seq length

    printf("Initializing TinyLLM Model training configuration:\n");
    printf("- dim: %d, layers: %d, heads: %d, vocab: %d\n", c.dim, c.n_layers, c.n_heads, c.vocab_size);
    printf("- batch: %d, seqlen: %d\n", B, T);

    // Allocate parameters & gradients
    Weights w;
    Gradients g;
    Weights opt_m, opt_v;

    size_t embed_size = c.vocab_size * c.dim * sizeof(float);
    size_t proj_size = c.dim * c.dim * sizeof(float);
    size_t ffn_size = c.hidden_dim * c.dim * sizeof(float);
    
    // Allocate Host buffers
    w.token_embedding_table = malloc_align(embed_size);
    g.token_embedding_table = malloc_align(embed_size);
    opt_m.token_embedding_table = calloc(c.vocab_size * c.dim, sizeof(float));
    opt_v.token_embedding_table = calloc(c.vocab_size * c.dim, sizeof(float));

    w.rms_att_weight = malloc(c.n_layers * sizeof(float*));
    w.wq = malloc(c.n_layers * sizeof(float*));
    w.wk = malloc(c.n_layers * sizeof(float*));
    w.wv = malloc(c.n_layers * sizeof(float*));
    w.wo = malloc(c.n_layers * sizeof(float*));
    w.rms_ffn_weight = malloc(c.n_layers * sizeof(float*));
    w.w1 = malloc(c.n_layers * sizeof(float*));
    w.w2 = malloc(c.n_layers * sizeof(float*));
    w.w3 = malloc(c.n_layers * sizeof(float*));

    g.rms_att_weight = malloc(c.n_layers * sizeof(float*));
    g.wq = malloc(c.n_layers * sizeof(float*));
    g.wk = malloc(c.n_layers * sizeof(float*));
    g.wv = malloc(c.n_layers * sizeof(float*));
    g.wo = malloc(c.n_layers * sizeof(float*));
    g.rms_ffn_weight = malloc(c.n_layers * sizeof(float*));
    g.w1 = malloc(c.n_layers * sizeof(float*));
    g.w2 = malloc(c.n_layers * sizeof(float*));
    g.w3 = malloc(c.n_layers * sizeof(float*));

    opt_m.rms_att_weight = malloc(c.n_layers * sizeof(float*));
    opt_m.wq = malloc(c.n_layers * sizeof(float*));
    opt_m.wk = malloc(c.n_layers * sizeof(float*));
    opt_m.wv = malloc(c.n_layers * sizeof(float*));
    opt_m.wo = malloc(c.n_layers * sizeof(float*));
    opt_m.rms_ffn_weight = malloc(c.n_layers * sizeof(float*));
    opt_m.w1 = malloc(c.n_layers * sizeof(float*));
    opt_m.w2 = malloc(c.n_layers * sizeof(float*));
    opt_m.w3 = malloc(c.n_layers * sizeof(float*));

    opt_v.rms_att_weight = malloc(c.n_layers * sizeof(float*));
    opt_v.wq = malloc(c.n_layers * sizeof(float*));
    opt_v.wk = malloc(c.n_layers * sizeof(float*));
    opt_v.wv = malloc(c.n_layers * sizeof(float*));
    opt_v.wo = malloc(c.n_layers * sizeof(float*));
    opt_v.rms_ffn_weight = malloc(c.n_layers * sizeof(float*));
    opt_v.w1 = malloc(c.n_layers * sizeof(float*));
    opt_v.w2 = malloc(c.n_layers * sizeof(float*));
    opt_v.w3 = malloc(c.n_layers * sizeof(float*));

    for (int l = 0; l < c.n_layers; l++) {
        w.rms_att_weight[l] = malloc_align(c.dim * sizeof(float));
        w.wq[l] = malloc_align(proj_size);
        w.wk[l] = malloc_align(proj_size);
        w.wv[l] = malloc_align(proj_size);
        w.wo[l] = malloc_align(proj_size);
        w.rms_ffn_weight[l] = malloc_align(c.dim * sizeof(float));
        w.w1[l] = malloc_align(ffn_size);
        w.w2[l] = malloc_align(ffn_size);
        w.w3[l] = malloc_align(ffn_size);

        g.rms_att_weight[l] = malloc_align(c.dim * sizeof(float));
        g.wq[l] = malloc_align(proj_size);
        g.wk[l] = malloc_align(proj_size);
        g.wv[l] = malloc_align(proj_size);
        g.wo[l] = malloc_align(proj_size);
        g.rms_ffn_weight[l] = malloc_align(c.dim * sizeof(float));
        g.w1[l] = malloc_align(ffn_size);
        g.w2[l] = malloc_align(ffn_size);
        g.w3[l] = malloc_align(ffn_size);

        opt_m.rms_att_weight[l] = calloc(c.dim, sizeof(float));
        opt_m.wq[l] = calloc(c.dim * c.dim, sizeof(float));
        opt_m.wk[l] = calloc(c.dim * c.dim, sizeof(float));
        opt_m.wv[l] = calloc(c.dim * c.dim, sizeof(float));
        opt_m.wo[l] = calloc(c.dim * c.dim, sizeof(float));
        opt_m.rms_ffn_weight[l] = calloc(c.dim, sizeof(float));
        opt_m.w1[l] = calloc(c.hidden_dim * c.dim, sizeof(float));
        opt_m.w2[l] = calloc(c.dim * c.hidden_dim, sizeof(float));
        opt_m.w3[l] = calloc(c.hidden_dim * c.dim, sizeof(float));

        opt_v.rms_att_weight[l] = calloc(c.dim, sizeof(float));
        opt_v.wq[l] = calloc(c.dim * c.dim, sizeof(float));
        opt_v.wk[l] = calloc(c.dim * c.dim, sizeof(float));
        opt_v.wv[l] = calloc(c.dim * c.dim, sizeof(float));
        opt_v.wo[l] = calloc(c.dim * c.dim, sizeof(float));
        opt_v.rms_ffn_weight[l] = calloc(c.dim, sizeof(float));
        opt_v.w1[l] = calloc(c.hidden_dim * c.dim, sizeof(float));
        opt_v.w2[l] = calloc(c.dim * c.hidden_dim, sizeof(float));
        opt_v.w3[l] = calloc(c.hidden_dim * c.dim, sizeof(float));
    }

    w.rms_final_weight = malloc_align(c.dim * sizeof(float));
    w.wcls = malloc_align(c.vocab_size * c.dim * sizeof(float));

    g.rms_final_weight = malloc_align(c.dim * sizeof(float));
    g.wcls = malloc_align(c.vocab_size * c.dim * sizeof(float));

    opt_m.rms_final_weight = calloc(c.dim, sizeof(float));
    opt_m.wcls = calloc(c.vocab_size * c.dim, sizeof(float));

    opt_v.rms_final_weight = calloc(c.dim, sizeof(float));
    opt_v.wcls = calloc(c.vocab_size * c.dim, sizeof(float));

    // Initialize Weights with standard Kaiming Normal distribution (mean=0, std=0.02)
    srand(42);
    float scale = 0.02f;
    #define INIT_NORMAL(arr, len) do { for(int i=0; i<len; i++) { float u1 = (float)rand()/RAND_MAX; float u2 = (float)rand()/RAND_MAX; arr[i] = scale * sqrtf(-2.0f * logf(u1 + 1e-9f)) * cosf(2.0f * M_PI * u2); } } while(0)
    #define INIT_CONSTANT(arr, len, val) do { for(int i=0; i<len; i++) arr[i] = val; } while(0)

    INIT_NORMAL(w.token_embedding_table, c.vocab_size * c.dim);
    for (int l = 0; l < c.n_layers; l++) {
        INIT_CONSTANT(w.rms_att_weight[l], c.dim, 1.0f);
        INIT_NORMAL(w.wq[l], c.dim * c.dim);
        INIT_NORMAL(w.wk[l], c.dim * c.dim);
        INIT_NORMAL(w.wv[l], c.dim * c.dim);
        INIT_NORMAL(w.wo[l], c.dim * c.dim);
        INIT_CONSTANT(w.rms_ffn_weight[l], c.dim, 1.0f);
        INIT_NORMAL(w.w1[l], c.hidden_dim * c.dim);
        INIT_NORMAL(w.w2[l], c.dim * c.hidden_dim);
        INIT_NORMAL(w.w3[l], c.hidden_dim * c.dim);
    }
    INIT_CONSTANT(w.rms_final_weight, c.dim, 1.0f);
    INIT_NORMAL(w.wcls, c.vocab_size * c.dim);

    // Allocate Batch Activations
    BatchActivations act;
    act.x = malloc_align((c.n_layers + 1) * B * T * c.dim * sizeof(float));
    act.x_final_norm = malloc_align(B * T * c.dim * sizeof(float));
    act.logits = malloc_align(B * T * c.vocab_size * sizeof(float));
    act.probs = malloc_align(B * T * c.vocab_size * sizeof(float));
    act.l = malloc(c.n_layers * sizeof(LayerActivations));

    for (int l = 0; l < c.n_layers; l++) {
        act.l[l].xb = malloc_align(B * T * c.dim * sizeof(float));
        act.l[l].q = malloc_align(B * T * c.dim * sizeof(float));
        act.l[l].k = malloc_align(B * T * c.dim * sizeof(float));
        act.l[l].v = malloc_align(B * T * c.dim * sizeof(float));
        act.l[l].q_rot = malloc_align(B * T * c.dim * sizeof(float));
        act.l[l].k_rot = malloc_align(B * T * c.dim * sizeof(float));
        act.l[l].att = malloc_align(B * c.n_heads * T * T * sizeof(float));
        act.l[l].xb2 = malloc_align(B * T * c.dim * sizeof(float));
        act.l[l].ffn_xb = malloc_align(B * T * c.dim * sizeof(float));
        act.l[l].hb = malloc_align(B * T * c.hidden_dim * sizeof(float));
        act.l[l].hb2 = malloc_align(B * T * c.hidden_dim * sizeof(float));
        act.l[l].hb_act = malloc_align(B * T * c.hidden_dim * sizeof(float));
    }

    // Allocate Gradients activation buffers
    g.d_x = malloc_align((c.n_layers + 1) * B * T * c.dim * sizeof(float));
    g.d_x_final_norm = malloc_align(B * T * c.dim * sizeof(float));
    g.d_logits = malloc_align(B * T * c.vocab_size * sizeof(float));
    g.d_xb = malloc_align(B * T * c.dim * sizeof(float));
    g.d_q = malloc_align(B * T * c.dim * sizeof(float));
    g.d_k = malloc_align(B * T * c.dim * sizeof(float));
    g.d_v = malloc_align(B * T * c.dim * sizeof(float));
    g.d_q_rot = malloc_align(B * T * c.dim * sizeof(float));
    g.d_k_rot = malloc_align(B * T * c.dim * sizeof(float));
    g.d_att = malloc_align(B * c.n_heads * T * T * sizeof(float));
    g.d_xb2 = malloc_align(B * T * c.dim * sizeof(float));
    g.d_ffn_xb = malloc_align(B * T * c.dim * sizeof(float));
    g.d_hb = malloc_align(B * T * c.hidden_dim * sizeof(float));
    g.d_hb2 = malloc_align(B * T * c.hidden_dim * sizeof(float));
    g.d_hb_act = malloc_align(B * T * c.hidden_dim * sizeof(float));

    // Allocate standard normalization caches
    float* cache_rstd = malloc_align((c.n_layers * 2 + 1) * B * T * sizeof(float));

    // Define Synthetic inputs and targets (representing target shifting)
    int* inputs = malloc(B * T * sizeof(int));
    int* targets = malloc(B * T * sizeof(int));

    // Synthetic repeating sequence: 10, 20, 30, 40...
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            inputs[b * T + t] = ((b * 5 + t) % 100) + 1;
            targets[b * T + t] = ((b * 5 + t + 1) % 100) + 1;
        }
    }

    // Training Parameters
    float lr = 1e-2f;
    float wd = 1e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;

    printf("Starting training optimization loop...\n");
    clock_t start_time = clock();

    // 40 training steps to show loss decreasing
    for (int step = 1; step <= 40; step++) {
        // Zero all gradients
        memset(g.token_embedding_table, 0, embed_size);
        for (int l = 0; l < c.n_layers; l++) {
            memset(g.rms_att_weight[l], 0, c.dim * sizeof(float));
            memset(g.wq[l], 0, proj_size);
            memset(g.wk[l], 0, proj_size);
            memset(g.wv[l], 0, proj_size);
            memset(g.wo[l], 0, proj_size);
            memset(g.rms_ffn_weight[l], 0, c.dim * sizeof(float));
            memset(g.w1[l], 0, ffn_size);
            memset(g.w2[l], 0, ffn_size);
            memset(g.w3[l], 0, ffn_size);
        }
        memset(g.rms_final_weight, 0, c.dim * sizeof(float));
        memset(g.wcls, 0, c.vocab_size * c.dim * sizeof(float));

        // 1. Forward Pass
        model_forward(&c, &w, &act, inputs, B, T, cache_rstd);

        // 2. Cross Entropy Loss calculation
        float loss = cross_entropy_forward(act.probs, act.logits, targets, B, T, c.vocab_size);

        if (step % 5 == 0 || step == 1) {
            printf("Step %d | Loss: %.5f\n", step, loss);
        }

        // 3. Backward Pass (Autograd)
        cross_entropy_backward(g.d_logits, act.probs, targets, B, T, c.vocab_size);
        model_backward(&c, &w, &g, &act, inputs, B, T, cache_rstd);

        // 4. AdamW Parameter updates
        adamw_update(w.token_embedding_table, g.token_embedding_table, opt_m.token_embedding_table, opt_v.token_embedding_table, c.vocab_size * c.dim, lr, wd, beta1, beta2, eps, step);
        for (int l = 0; l < c.n_layers; l++) {
            adamw_update(w.rms_att_weight[l], g.rms_att_weight[l], opt_m.rms_att_weight[l], opt_v.rms_att_weight[l], c.dim, lr, wd, beta1, beta2, eps, step);
            adamw_update(w.wq[l], g.wq[l], opt_m.wq[l], opt_v.wq[l], c.dim * c.dim, lr, wd, beta1, beta2, eps, step);
            adamw_update(w.wk[l], g.wk[l], opt_m.wk[l], opt_v.wk[l], c.dim * c.dim, lr, wd, beta1, beta2, eps, step);
            adamw_update(w.wv[l], g.wv[l], opt_m.wv[l], opt_v.wv[l], c.dim * c.dim, lr, wd, beta1, beta2, eps, step);
            adamw_update(w.wo[l], g.wo[l], opt_m.wo[l], opt_v.wo[l], c.dim * c.dim, lr, wd, beta1, beta2, eps, step);
            adamw_update(w.rms_ffn_weight[l], g.rms_ffn_weight[l], opt_m.rms_ffn_weight[l], opt_v.rms_ffn_weight[l], c.dim, lr, wd, beta1, beta2, eps, step);
            adamw_update(w.w1[l], g.w1[l], opt_m.w1[l], opt_v.w1[l], c.hidden_dim * c.dim, lr, wd, beta1, beta2, eps, step);
            adamw_update(w.w2[l], g.w2[l], opt_m.w2[l], opt_v.w2[l], c.dim * c.hidden_dim, lr, wd, beta1, beta2, eps, step);
            adamw_update(w.w3[l], g.w3[l], opt_m.w3[l], opt_v.w3[l], c.hidden_dim * c.dim, lr, wd, beta1, beta2, eps, step);
        }
        adamw_update(w.rms_final_weight, g.rms_final_weight, opt_m.rms_final_weight, opt_v.rms_final_weight, c.dim, lr, wd, beta1, beta2, eps, step);
        adamw_update(w.wcls, g.wcls, opt_m.wcls, opt_v.wcls, c.vocab_size * c.dim, lr, wd, beta1, beta2, eps, step);
    }

    double elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    printf("Training optimization completed in %.4f seconds.\n", elapsed_time);

    // Free all memory
    free_align(w.token_embedding_table);
    free_align(g.token_embedding_table);
    free(opt_m.token_embedding_table);
    free(opt_v.token_embedding_table);

    for (int l = 0; l < c.n_layers; l++) {
        free_align(w.rms_att_weight[l]);
        free_align(w.wq[l]); free_align(w.wk[l]); free_align(w.wv[l]); free_align(w.wo[l]);
        free_align(w.rms_ffn_weight[l]);
        free_align(w.w1[l]); free_align(w.w2[l]); free_align(w.w3[l]);

        free_align(g.rms_att_weight[l]);
        free_align(g.wq[l]); free_align(g.wk[l]); free_align(g.wv[l]); free_align(g.wo[l]);
        free_align(g.rms_ffn_weight[l]);
        free_align(g.w1[l]); free_align(g.w2[l]); free_align(g.w3[l]);

        free(opt_m.rms_att_weight[l]);
        free(opt_m.wq[l]); free(opt_m.wk[l]); free(opt_m.wv[l]); free(opt_m.wo[l]);
        free(opt_m.rms_ffn_weight[l]);
        free(opt_m.w1[l]); free(opt_m.w2[l]); free(opt_m.w3[l]);

        free(opt_v.rms_att_weight[l]);
        free(opt_v.wq[l]); free(opt_v.wk[l]); free(opt_v.wv[l]); free(opt_v.wo[l]);
        free(opt_v.rms_ffn_weight[l]);
        free(opt_v.w1[l]); free(opt_v.w2[l]); free(opt_v.w3[l]);

        free_align(act.l[l].xb);
        free_align(act.l[l].q); free_align(act.l[l].k); free_align(act.l[l].v);
        free_align(act.l[l].q_rot); free_align(act.l[l].k_rot);
        free_align(act.l[l].att); free_align(act.l[l].xb2);
        free_align(act.l[l].ffn_xb);
        free_align(act.l[l].hb); free_align(act.l[l].hb2); free_align(act.l[l].hb_act);
    }
    
    free_align(w.rms_final_weight); free_align(w.wcls);
    free_align(g.rms_final_weight); free_align(g.wcls);
    free(opt_m.rms_final_weight); free(opt_m.wcls);
    free(opt_v.rms_final_weight); free(opt_v.wcls);

    free_align(act.x); free_align(act.x_final_norm); free_align(act.logits); free_align(act.probs);
    free(act.l);
    
    free_align(g.d_x); free_align(g.d_x_final_norm); free_align(g.d_logits);
    free_align(g.d_xb); free_align(g.d_q); free_align(g.d_k); free_align(g.d_v);
    free_align(g.d_q_rot); free_align(g.d_k_rot); free_align(g.d_att); free_align(g.d_xb2);
    free_align(g.d_ffn_xb); free_align(g.d_hb); free_align(g.d_hb2); free_align(g.d_hb_act);

    free_align(cache_rstd);
    free(inputs); free(targets);

    return 0;
}
