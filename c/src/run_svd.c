/*
 * Bare-Metal C Inference Engine for Hybrid SVD + Int8 Quantized LLMs (c/src/run_svd.c).
 *
 * Implements low-rank factorization matrix-vector multiplications:
 *   h = x @ W_B^T  (shape: [1, in_dim] -> [1, r])
 *   out = h @ W_A^T (shape: [1, r] -> [1, out_dim])
 * where W_A and W_B are dequantized on-the-fly from Int8 vectors.
 *
 * Compile: make run_svd
 * Run:     ./run_svd checkpoints/model_svd.bin checkpoints/vocab.bin 40
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>

#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

typedef struct {
    int dim;
    int ffn_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int max_seq_len;
    int rank;
} Config;

typedef struct {
    int8_t* w_a;
    float scale_a;
    int8_t* w_b;
    float scale_b;
} SVDLayer;

typedef struct {
    float* rms_att;
    SVDLayer wq;
    SVDLayer wk;
    SVDLayer wv;
    SVDLayer wo;
    float* rms_ffn;
    SVDLayer w1;
    SVDLayer w2;
    SVDLayer w3;
} SVDTransformerBlock;

typedef struct {
    float* tok_embeddings;
    SVDTransformerBlock* layers;
    float* rms_final;
    float* wcls;
} SVDWeights;

typedef struct {
    float* x;
    float* xb;
    float* h_rank; // Buffer for intermediate rank r vector
    float* q;
    float* k;
    float* v;
    float* att;
    float* logits;
    float* key_cache;
    float* value_cache;
} RunState;

// Low-Rank SVD Int8 Matrix-Vector Multiply: out = x @ W_B.T @ W_A.T
void mat_vec_svd_int8(float* out, float* x, SVDLayer* layer, int in_dim, int out_dim, int rank, float* h_buf) {
    // 1. Low-Rank Projection: h_buf [r] = x [in_dim] @ W_B [r, in_dim].T
    for (int r_idx = 0; r_idx < rank; r_idx++) {
        float sum = 0.0f;
        int8_t* b_row = layer->w_b + r_idx * in_dim;
        for (int k = 0; k < in_dim; k++) {
            sum += x[k] * ((float)b_row[k] * layer->scale_b);
        }
        h_buf[r_idx] = sum;
    }

    // 2. Output Projection: out [out_dim] = h_buf [r] @ W_A [out_dim, r].T
    for (int i = 0; i < out_dim; i++) {
        float sum = 0.0f;
        int8_t* a_row = layer->w_a + i * rank;
        for (int r_idx = 0; r_idx < rank; r_idx++) {
            sum += h_buf[r_idx] * ((float)a_row[r_idx] * layer->scale_a);
        }
        out[i] = sum;
    }
}

// Standard FP32 Matrix-Vector Multiply for Head & Embeddings
void mat_vec_fp32(float* out, float* x, float* w, int in_dim, int out_dim) {
    for (int i = 0; i < out_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < in_dim; j++) {
            sum += x[j] * w[i * in_dim + j];
        }
        out[i] = sum;
    }
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_p = probabilities[i];
            max_i = i;
        }
    }
    return max_i;
}

// Forward Pass for a single token position using Hybrid SVD + Int8 Layers
float* forward_svd(Config* p, SVDWeights* w, RunState* s, int token, int pos) {
    float* x = s->x;
    int dim = p->dim;
    int ffn_dim = p->ffn_dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int head_size = dim / p->n_heads;

    // 1. Embedding lookup
    memcpy(x, w->tok_embeddings + token * dim, dim * sizeof(float));

    // 2. Loop through SVD Transformer Blocks
    for (int l = 0; l < p->n_layers; l++) {
        SVDTransformerBlock* block = &w->layers[l];

        // Attention RMSNorm
        rmsnorm(s->xb, x, block->rms_att, dim);

        // SVD Int8 Attention Projections
        mat_vec_svd_int8(s->q, s->xb, &block->wq, dim, dim, p->rank, s->h_rank);
        mat_vec_svd_int8(s->k, s->xb, &block->wk, dim, kv_dim, p->rank, s->h_rank);
        mat_vec_svd_int8(s->v, s->xb, &block->wv, dim, kv_dim, p->rank, s->h_rank);

        // Apply RoPE
        for (int i = 0; i < dim; i += 2) {
            int head_dim_idx = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim_idx / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // Cache Key & Value
        int loff = l * p->max_seq_len * kv_dim;
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(float));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(float));

        // Multi-Head Attention Calculation
        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->max_seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / (p->n_heads / p->n_kv_heads)) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }
            softmax(att, pos + 1);

            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / (p->n_heads / p->n_kv_heads)) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // SVD Int8 Output Projection wo
        mat_vec_svd_int8(s->q, s->xb, &block->wo, dim, dim, p->rank, s->h_rank);
        for (int i = 0; i < dim; i++) x[i] += s->q[i];

        // FFN RMSNorm
        rmsnorm(s->xb, x, block->rms_ffn, dim);

        // SVD Int8 SwiGLU FFN Projections (w1, w2, w3)
        // w1 (gate): [dim] -> [ffn_dim], w3 (up): [dim] -> [ffn_dim]
        mat_vec_svd_int8(s->q, s->xb, &block->w1, dim, ffn_dim, p->rank, s->h_rank);
        mat_vec_svd_int8(s->k, s->xb, &block->w3, dim, ffn_dim, p->rank, s->h_rank);

        // SwiGLU activation: q = SiLU(w1) * w3
        for (int i = 0; i < ffn_dim; i++) {
            float val = s->q[i];
            s->q[i] = (val / (1.0f + expf(-val))) * s->k[i];
        }

        // SVD Int8 Down-projection w2: [ffn_dim] -> [dim]
        mat_vec_svd_int8(s->xb, s->q, &block->w2, ffn_dim, dim, p->rank, s->h_rank);
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];
    }

    // Final RMSNorm & Output Logits
    rmsnorm(x, x, w->rms_final, dim);
    mat_vec_fp32(s->logits, x, w->wcls, dim, p->vocab_size);
    return s->logits;
}

static void read_svd_layer(FILE* f, SVDLayer* l, int in_d, int out_d, int rank) {
    l->w_a = (int8_t*)malloc(out_d * rank * sizeof(int8_t));
    if (fread(l->w_a, sizeof(int8_t), out_d * rank, f) != (size_t)(out_d * rank)) {
        printf("Error reading w_a\n");
    }
    if (fread(&l->scale_a, sizeof(float), 1, f) != 1) {
        printf("Error reading scale_a\n");
    }

    l->w_b = (int8_t*)malloc(rank * in_d * sizeof(int8_t));
    if (fread(l->w_b, sizeof(int8_t), rank * in_d, f) != (size_t)(rank * in_d)) {
        printf("Error reading w_b\n");
    }
    if (fread(&l->scale_b, sizeof(float), 1, f) != 1) {
        printf("Error reading scale_b\n");
    }
}

// Memory-map / Load SVD Model Binary
void load_svd_model(char* checkpoint_path, Config* config, SVDWeights* weights) {
    FILE* file = fopen(checkpoint_path, "rb");
    if (!file) {
        printf("❌ Failed to open SVD model checkpoint file: %s\n", checkpoint_path);
        exit(1);
    }

    // Read 256-byte header
    if (fread(config, sizeof(Config), 1, file) != 1) {
        printf("❌ Failed to read header from SVD model checkpoint\n");
        exit(1);
    }
    fseek(file, 256, SEEK_SET);

    // Read Token Embeddings
    int emb_size = config->vocab_size * config->dim;
    weights->tok_embeddings = (float*)malloc(emb_size * sizeof(float));
    fread(weights->tok_embeddings, sizeof(float), emb_size, file);

    // Read Transformer Blocks
    weights->layers = (SVDTransformerBlock*)malloc(config->n_layers * sizeof(SVDTransformerBlock));
    int dim = config->dim;
    int ffn_dim = config->ffn_dim;
    int r = config->rank;

    for (int i = 0; i < config->n_layers; i++) {
        SVDTransformerBlock* b = &weights->layers[i];
        b->rms_att = (float*)malloc(dim * sizeof(float));
        fread(b->rms_att, sizeof(float), dim, file);

        read_svd_layer(file, &b->wq, dim, dim, r);
        read_svd_layer(file, &b->wk, dim, dim, r);
        read_svd_layer(file, &b->wv, dim, dim, r);
        read_svd_layer(file, &b->wo, dim, dim, r);

        b->rms_ffn = (float*)malloc(dim * sizeof(float));
        fread(b->rms_ffn, sizeof(float), dim, file);

        read_svd_layer(file, &b->w1, dim, ffn_dim, r);
        read_svd_layer(file, &b->w2, ffn_dim, dim, r);
        read_svd_layer(file, &b->w3, dim, ffn_dim, r);
    }

    // Read Final RMSNorm & Output Head
    weights->rms_final = (float*)malloc(dim * sizeof(float));
    fread(weights->rms_final, sizeof(float), dim, file);

    int wcls_size = config->vocab_size * dim;
    weights->wcls = (float*)malloc(wcls_size * sizeof(float));
    fread(weights->wcls, sizeof(float), wcls_size, file);

    fclose(file);
}

int main(int argc, char* argv[]) {
    char* model_path = (argc > 1) ? argv[1] : "c/model_svd.bin";
    char* vocab_path = (argc > 2) ? argv[2] : "c/vocab.bin";
    int num_steps = (argc > 3) ? atoi(argv[3]) : 30;

    printf("🍏 Bare-Metal C SVD+Int8 Inference Engine Initializing...\n");
    printf("  Model Binary: %s\n", model_path);
    printf("  Vocab Binary: %s\n", vocab_path);

    Config config;
    SVDWeights weights;
    load_svd_model(model_path, &config, &weights);

    printf("  dim: %d | layers: %d | heads: %d | vocab: %d | SVD rank: %d\n\n",
           config.dim, config.n_layers, config.n_heads, config.vocab_size, config.rank);

    // Read Vocabulary
    char** vocab = (char**)malloc(config.vocab_size * sizeof(char*));
    FILE* vfile = fopen(vocab_path, "rb");
    if (vfile) {
        for (int i = 0; i < config.vocab_size; i++) {
            unsigned char len = 0;
            fread(&len, 1, 1, vfile);
            if (len > 31) len = 31;
            vocab[i] = (char*)malloc(32);
            fread(vocab[i], 1, 31, vfile);
            vocab[i][len] = '\0';
        }
        fclose(vfile);
    } else {
        for (int i = 0; i < config.vocab_size; i++) {
            vocab[i] = (char*)malloc(16);
            sprintf(vocab[i], "[%d]", i);
        }
    }

    // Allocate RunState Buffers
    RunState state;
    state.x = (float*)calloc(config.dim, sizeof(float));
    state.xb = (float*)calloc(config.dim, sizeof(float));
    state.h_rank = (float*)calloc(config.rank, sizeof(float));
    state.q = (float*)calloc(config.dim, sizeof(float));
    state.k = (float*)calloc(config.dim, sizeof(float));
    state.v = (float*)calloc(config.dim, sizeof(float));
    state.att = (float*)calloc(config.n_heads * config.max_seq_len, sizeof(float));
    state.logits = (float*)calloc(config.vocab_size, sizeof(float));

    int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    state.key_cache = (float*)calloc(config.n_layers * config.max_seq_len * kv_dim, sizeof(float));
    state.value_cache = (float*)calloc(config.n_layers * config.max_seq_len * kv_dim, sizeof(float));

    // Generation Loop
    int token = 1; // SOS token
    printf("Generated Text (SVD Low-Rank + Int8 Quantization): ");

    clock_t start = clock();
    for (int pos = 0; pos < num_steps; pos++) {
        float* logits = forward_svd(&config, &weights, &state, token, pos);
        int next_token = sample_argmax(logits, config.vocab_size);
        printf("%s ", vocab[next_token]);
        fflush(stdout);
        token = next_token;
    }
    clock_t end = clock();

    double total_time = (double)(end - start) / CLOCKS_PER_SEC;
    double tok_per_sec = num_steps / total_time;
    printf("\n\n📊 SVD+Int8 Engine Throughput: %.2f tokens/sec\n", tok_per_sec);

    return 0;
}
