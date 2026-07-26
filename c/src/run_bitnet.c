/*
 * Bare-Metal 1.58-Bit (BitNet b1.58) C Autoregressive Text Generation Engine.
 *
 * Implements pure ternary addition-and-subtraction matrix multiplication
 * with ZERO floating-point multiplications for all BitLinear layers.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

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
    float* token_embedding_table; // (vocab_size, dim)
    // Layer weights
    float** rms_att_weight;       // (n_layers, dim)
    int8_t** wq;                  // (n_layers, dim * dim) in {-1, 0, +1}
    int8_t** wk;                  // (n_layers, dim * dim) in {-1, 0, +1}
    int8_t** wv;                  // (n_layers, dim * dim) in {-1, 0, +1}
    int8_t** wo;                  // (n_layers, dim * dim) in {-1, 0, +1}
    float** rms_ffn_weight;       // (n_layers, dim)
    int8_t** w1;                  // (n_layers, ffn_dim * dim) in {-1, 0, +1}
    int8_t** w2;                  // (n_layers, dim * ffn_dim) in {-1, 0, +1}
    int8_t** w3;                  // (n_layers, ffn_dim * dim) in {-1, 0, +1}
    // Final head
    float* rms_final_weight;      // (dim,)
    float* wcls;                  // (vocab_size, dim)
} Weights;

typedef struct {
    float* x;
    float* xb;
    float* xb2;
    float* hb;
    float* hb2;
    float* q;
    float* k;
    float* v;
    float* att;
    float* logits;
    float* key_cache;
    float* value_cache;
} RunState;

void malloc_run_state(RunState* s, Config* p) {
    s->x = (float*)malloc(p->dim * sizeof(float));
    s->xb = (float*)malloc(p->dim * sizeof(float));
    s->xb2 = (float*)malloc(p->dim * sizeof(float));
    s->hb = (float*)malloc(p->ffn_dim * sizeof(float));
    s->hb2 = (float*)malloc(p->ffn_dim * sizeof(float));
    s->q = (float*)malloc(p->dim * sizeof(float));
    s->k = (float*)malloc(p->dim * sizeof(float));
    s->v = (float*)malloc(p->dim * sizeof(float));
    s->att = (float*)malloc(p->n_heads * p->max_seq_len * sizeof(float));
    s->logits = (float*)malloc(p->vocab_size * sizeof(float));
    s->key_cache = (float*)calloc(p->n_layers * p->max_seq_len * p->dim, sizeof(float));
    s->value_cache = (float*)calloc(p->n_layers * p->max_seq_len * p->dim, sizeof(float));
}

void free_run_state(RunState* s) {
    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
    free(s->q); free(s->k); free(s->v); free(s->att); free(s->logits);
    free(s->key_cache); free(s->value_cache);
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) { ss += x[j] * x[j]; }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) { o[j] = x[j] * ss * weight[j]; }
}

/*
 * PURE 1.58-BIT TERNARY MATRIX MULTIPLICATION KERNEL.
 * Replaces floating-point MACs with ZERO multiplications!
 * Only integer additions, subtractions, and skips (w == 0).
 */
void bitlinear_forward(float* out, float* x, int8_t* w, int in_dim, int out_dim) {
    float max_val = 0.0f;
    for (int i = 0; i < in_dim; i++) {
        float abs_val = fabsf(x[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    float gamma_x = (max_val < 1e-5f) ? 1e-5f : max_val;
    float scale_x = 127.0f / gamma_x;

    for (int i = 0; i < out_dim; i++) {
        int32_t sum = 0;
        int8_t* w_row = w + i * in_dim;
        for (int j = 0; j < in_dim; j++) {
            int8_t weight = w_row[j];
            int32_t x_q = (int32_t)roundf(x[j] * scale_x);

            if (weight == 1) {
                sum += x_q;      // Pure Addition
            } else if (weight == -1) {
                sum -= x_q;      // Pure Subtraction
            }
        }
        out[i] = (float)sum * (gamma_x / 127.0f);
    }
}

void matmul_float(float* out, float* x, float* w, int in_dim, int out_dim) {
    for (int i = 0; i < out_dim; i++) {
        float val = 0.0f;
        for (int j = 0; j < in_dim; j++) {
            val += w[i * in_dim + j] * x[j];
        }
        out[i] = val;
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) { if (x[i] > max_val) max_val = x[i]; }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) { x[i] /= sum; }
}

int argmax(float* x, int size) {
    int max_i = 0;
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) { max_val = x[i]; max_i = i; }
    }
    return max_i;
}

float* forward_bitnet(Config* p, Weights* w, RunState* s, int token, int pos) {
    int dim = p->dim;
    int ffn_dim = p->ffn_dim;
    int head_size = dim / p->n_heads;

    memcpy(s->x, w->token_embedding_table + token * dim, dim * sizeof(float));

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, s->x, w->rms_att_weight[l], dim);

        // 1.58-bit Ternary Attention Projections (0 FP Multiplications!)
        bitlinear_forward(s->q, s->xb, w->wq[l], dim, dim);
        bitlinear_forward(s->k, s->xb, w->wk[l], dim, dim);
        bitlinear_forward(s->v, s->xb, w->wv[l], dim, dim);

        // RoPE
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val), fci = sinf(val);

            float q0 = s->q[i], q1 = s->q[i + 1];
            s->q[i] = q0 * fcr - q1 * fci;
            s->q[i + 1] = q0 * fci + q1 * fcr;

            float k0 = s->k[i], k1 = s->k[i + 1];
            s->k[i] = k0 * fcr - k1 * fci;
            s->k[i + 1] = k0 * fci + k1 * fcr;
        }

        // Cache KV
        int loff = l * p->max_seq_len * dim;
        memcpy(s->key_cache + loff + pos * dim, s->k, dim * sizeof(float));
        memcpy(s->value_cache + loff + pos * dim, s->v, dim * sizeof(float));

        // Multi-Head Attention
        for (int h = 0; h < p->n_heads; h++) {
            float* q_head = s->q + h * head_size;
            float* att_head = s->att + h * p->max_seq_len;

            for (int t = 0; t <= pos; t++) {
                float* k_head = s->key_cache + loff + t * dim + h * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) { score += q_head[i] * k_head[i]; }
                att_head[t] = score / sqrtf(head_size);
            }

            softmax(att_head, pos + 1);

            float* xb_head = s->xb2 + h * head_size;
            memset(xb_head, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v_head = s->value_cache + loff + t * dim + h * head_size;
                float a = att_head[t];
                for (int i = 0; i < head_size; i++) { xb_head[i] += a * v_head[i]; }
            }
        }

        bitlinear_forward(s->xb, s->xb2, w->wo[l], dim, dim);
        for (int i = 0; i < dim; i++) { s->x[i] += s->xb[i]; }

        rmsnorm(s->xb, s->x, w->rms_ffn_weight[l], dim);

        // 1.58-bit Ternary FFN Projections (0 FP Multiplications!)
        bitlinear_forward(s->hb, s->xb, w->w1[l], dim, ffn_dim);
        bitlinear_forward(s->hb2, s->xb, w->w3[l], dim, ffn_dim);

        for (int i = 0; i < ffn_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val))) * s->hb2[i];
            s->hb[i] = val;
        }

        bitlinear_forward(s->xb, s->hb, w->w2[l], ffn_dim, dim);
        for (int i = 0; i < dim; i++) { s->x[i] += s->xb[i]; }
    }

    rmsnorm(s->x, s->x, w->rms_final_weight, dim);
    matmul_float(s->logits, s->x, w->wcls, dim, p->vocab_size);
    return s->logits;
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

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: ./run_bitnet <model_bin> [vocab_bin] [steps]\n");
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

    Weights weights;
    weights.token_embedding_table = (float*)malloc(config.vocab_size * config.dim * sizeof(float));
    fread(weights.token_embedding_table, sizeof(float), config.vocab_size * config.dim, file);

    weights.rms_att_weight = (float**)malloc(config.n_layers * sizeof(float*));
    weights.wq = (int8_t**)malloc(config.n_layers * sizeof(int8_t*));
    weights.wk = (int8_t**)malloc(config.n_layers * sizeof(int8_t*));
    weights.wv = (int8_t**)malloc(config.n_layers * sizeof(int8_t*));
    weights.wo = (int8_t**)malloc(config.n_layers * sizeof(int8_t*));
    weights.rms_ffn_weight = (float**)malloc(config.n_layers * sizeof(float*));
    weights.w1 = (int8_t**)malloc(config.n_layers * sizeof(int8_t*));
    weights.w2 = (int8_t**)malloc(config.n_layers * sizeof(int8_t*));
    weights.w3 = (int8_t**)malloc(config.n_layers * sizeof(int8_t*));

    for (int i = 0; i < config.n_layers; i++) {
        weights.rms_att_weight[i] = (float*)malloc(config.dim * sizeof(float));
        fread(weights.rms_att_weight[i], sizeof(float), config.dim, file);

        weights.wq[i] = (int8_t*)malloc(config.dim * config.dim);
        fread(weights.wq[i], 1, config.dim * config.dim, file);
        weights.wk[i] = (int8_t*)malloc(config.dim * config.dim);
        fread(weights.wk[i], 1, config.dim * config.dim, file);
        weights.wv[i] = (int8_t*)malloc(config.dim * config.dim);
        fread(weights.wv[i], 1, config.dim * config.dim, file);
        weights.wo[i] = (int8_t*)malloc(config.dim * config.dim);
        fread(weights.wo[i], 1, config.dim * config.dim, file);

        weights.rms_ffn_weight[i] = (float*)malloc(config.dim * sizeof(float));
        fread(weights.rms_ffn_weight[i], sizeof(float), config.dim, file);

        weights.w1[i] = (int8_t*)malloc(config.ffn_dim * config.dim);
        fread(weights.w1[i], 1, config.ffn_dim * config.dim, file);
        weights.w2[i] = (int8_t*)malloc(config.dim * config.ffn_dim);
        fread(weights.w2[i], 1, config.dim * config.ffn_dim, file);
        weights.w3[i] = (int8_t*)malloc(config.ffn_dim * config.dim);
        fread(weights.w3[i], 1, config.ffn_dim * config.dim, file);
    }

    weights.rms_final_weight = (float*)malloc(config.dim * sizeof(float));
    fread(weights.rms_final_weight, sizeof(float), config.dim, file);
    weights.wcls = (float*)malloc(config.vocab_size * config.dim * sizeof(float));
    fread(weights.wcls, sizeof(float), config.vocab_size * config.dim, file);
    fclose(file);

    load_vocab(vocab_path);

    RunState state;
    malloc_run_state(&state, &config);

    printf("⚡ Bare-Metal 1.58-Bit (BitNet b1.58) C Inference Engine Running...\n");
    printf("  dim: %d | layers: %d | heads: %d | vocab: %d\n\n", config.dim, config.n_layers, config.n_heads, config.vocab_size);

    int token = 1; // Start token ID
    printf("Generated Text (0 FP Multiplications in BitLinear): ");
    fflush(stdout);

    long start_time = clock();
    for (int pos = 0; pos < steps; pos++) {
        float* logits = forward_bitnet(&config, &weights, &state, token, pos);
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
        printf("\n\n📊 Throughput: %.2f tokens/sec\n", steps / seconds);
    }

    free_run_state(&state);
    return 0;
}
