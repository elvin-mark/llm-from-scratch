#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Model Config
typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int max_seq_len;
    int is_quantized;
} Config;

// Quantized Weights mapped from memory
typedef struct {
    float* token_embedding_table;
    
    float** rms_att_weight;
    int8_t** wq; float** wq_s;
    int8_t** wk; float** wk_s;
    int8_t** wv; float** wv_s;
    int8_t** wo; float** wo_s;
    
    float** rms_ffn_weight;
    int8_t** w1; float** w1_s;
    int8_t** w2; float** w2_s;
    int8_t** w3; float** w3_s;
    
    float* rms_final_weight;
    int8_t* wcls; float* wcls_s;
} WeightsQ8;

// RunState buffers
typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float *key_cache;
    float *value_cache;
} RunState;

// Math kernels
void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= size;
    ss += 1e-6f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) o[j] = weight[j] * (x[j] * ss);
}

// INT8 Matrix Multiplication kernel
void matmul_q8(float* xout, float* x, int8_t* w, float* scales, int n, int d) {
    // 1. Quantize the input activation vector `x` dynamically
    float amax = 0.0f;
    for (int j = 0; j < n; j++) {
        if (fabsf(x[j]) > amax) amax = fabsf(x[j]);
    }
    float scale_x = amax / 127.0f;
    if (scale_x == 0.0f) scale_x = 1.0f; // avoid division by zero
    float inv_scale_x = 1.0f / scale_x;
    
    int8_t qx[8192]; // assuming n <= 8192 for TinyLLM
    for (int j = 0; j < n; j++) {
        qx[j] = (int8_t)roundf(x[j] * inv_scale_x);
    }

    // 2. Compute integer dot product and dequantize
    for (int i = 0; i < d; i++) {
        int32_t ival = 0;
        for (int j = 0; j < n; j++) {
            ival += ((int32_t)w[i * n + j]) * ((int32_t)qx[j]);
        }
        xout[i] = ((float)ival) * (scale_x * scales[i]);
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

int argmax(float* x, int size) {
    int max_i = 0;
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
            max_i = i;
        }
    }
    return max_i;
}

// Forward pass for Quantized Model
float* forward(Config* p, WeightsQ8* w, RunState* s, int token, int pos) {
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int head_size = dim / p->n_heads;

    // 1. Copy token embedding (kept as FP32)
    memcpy(s->x, w->token_embedding_table + token * dim, dim * sizeof(float));

    // For each layer
    for(int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, s->x, w->rms_att_weight[l], dim);

        // Q, K, V using INT8
        matmul_q8(s->q, s->xb, w->wq[l], w->wq_s[l], dim, dim);
        matmul_q8(s->k, s->xb, w->wk[l], w->wk_s[l], dim, kv_dim);
        matmul_q8(s->v, s->xb, w->wv[l], w->wv_s[l], dim, kv_dim);

        // RoPE
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);

            float q0 = s->q[i]; float q1 = s->q[i+1];
            s->q[i]   = q0 * fcr - q1 * fci;
            s->q[i+1] = q0 * fci + q1 * fcr;
            
            if (i < kv_dim) {
                float k0 = s->k[i]; float k1 = s->k[i+1];
                s->k[i]   = k0 * fcr - k1 * fci;
                s->k[i+1] = k0 * fci + k1 * fcr;
            }
        }

        // Cache KV
        int loff = l * p->max_seq_len * kv_dim;
        memcpy(s->key_cache + loff + pos * kv_dim, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + loff + pos * kv_dim, s->v, kv_dim * sizeof(float));

        // Attention
        for (int h = 0; h < p->n_heads; h++) {
            float* q_head = s->q + h * head_size;
            float* att_head = s->att + h * p->max_seq_len;
            
            for (int t = 0; t <= pos; t++) {
                float* k_head = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += q_head[i] * k_head[i];
                att_head[t] = score / sqrtf(head_size);
            }

            softmax(att_head, pos + 1);

            float* xb_head = s->xb2 + h * head_size;
            memset(xb_head, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v_head = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att_head[t];
                for (int i = 0; i < head_size; i++) xb_head[i] += a * v_head[i];
            }
        }

        // Wo using INT8
        matmul_q8(s->xb, s->xb2, w->wo[l], w->wo_s[l], dim, dim);

        // Residual
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];

        // FFN norm
        rmsnorm(s->xb, s->x, w->rms_ffn_weight[l], dim);

        // FFN using INT8
        matmul_q8(s->hb, s->xb, w->w1[l], w->w1_s[l], dim, hidden_dim);
        matmul_q8(s->hb2, s->xb, w->w3[l], w->w3_s[l], dim, hidden_dim);
        
        // SwiGLU
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        matmul_q8(s->xb, s->hb, w->w2[l], w->w2_s[l], hidden_dim, dim);

        // Residual
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
    }

    // Final Norm & Logits (INT8)
    rmsnorm(s->x, s->x, w->rms_final_weight, dim);
    matmul_q8(s->logits, s->x, w->wcls, w->wcls_s, dim, p->vocab_size);

    return s->logits;
}

// Tokenizer and Vocab
char** vocab;
int vocab_size;

void load_vocab(const char* filepath) {
    FILE* f = fopen(filepath, "rb");
    if (!f) exit(1);
    int read_count = fread(&vocab_size, sizeof(int), 1, f);
    if (read_count != 1) exit(1);
    vocab = (char**)malloc(vocab_size * sizeof(char*));
    for (int i = 0; i < vocab_size; i++) {
        int len;
        read_count = fread(&len, sizeof(int), 1, f);
        vocab[i] = (char*)malloc(len + 1);
        if(len > 0) read_count = fread(vocab[i], 1, len, f);
        vocab[i][len] = '\0';
    }
    fclose(f);
}

int main() {
    int fd = open("model_q8.bin", O_RDONLY);
    if (fd == -1) { printf("Cannot open model_q8.bin. Run export_q8.py first!\n"); return 1; }
    
    struct stat sb;
    fstat(fd, &sb);
    int8_t* data = (int8_t*)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) return 1;

    Config c;
    int* idata = (int*)data;
    c.dim = idata[0]; c.hidden_dim = idata[1]; c.n_layers = idata[2];
    c.n_heads = idata[3]; c.n_kv_heads = idata[4]; c.vocab_size = idata[5]; c.max_seq_len = idata[6];
    c.is_quantized = idata[7];

    if (!c.is_quantized) { printf("Model is not quantized! Please run export_q8.py\n"); return 1; }

    int8_t* w_ptr = data + 256; // Header is 256 bytes

    // Macros for easy pointer mapping
    #define ADVANCE_FP32(ptr, size) do { ptr = (float*)w_ptr; w_ptr += (size)*sizeof(float); } while(0)
    #define ADVANCE_Q8(q, s, d, n) do { s = (float*)w_ptr; w_ptr += (d)*sizeof(float); q = (int8_t*)w_ptr; w_ptr += (d)*(n)*sizeof(int8_t); } while(0)

    WeightsQ8 w;
    ADVANCE_FP32(w.token_embedding_table, c.vocab_size * c.dim);

    w.rms_att_weight = malloc(c.n_layers * sizeof(float*));
    w.wq = malloc(c.n_layers * sizeof(int8_t*)); w.wq_s = malloc(c.n_layers * sizeof(float*));
    w.wk = malloc(c.n_layers * sizeof(int8_t*)); w.wk_s = malloc(c.n_layers * sizeof(float*));
    w.wv = malloc(c.n_layers * sizeof(int8_t*)); w.wv_s = malloc(c.n_layers * sizeof(float*));
    w.wo = malloc(c.n_layers * sizeof(int8_t*)); w.wo_s = malloc(c.n_layers * sizeof(float*));
    w.rms_ffn_weight = malloc(c.n_layers * sizeof(float*));
    w.w1 = malloc(c.n_layers * sizeof(int8_t*)); w.w1_s = malloc(c.n_layers * sizeof(float*));
    w.w2 = malloc(c.n_layers * sizeof(int8_t*)); w.w2_s = malloc(c.n_layers * sizeof(float*));
    w.w3 = malloc(c.n_layers * sizeof(int8_t*)); w.w3_s = malloc(c.n_layers * sizeof(float*));

    int kv_dim = (c.dim * c.n_kv_heads) / c.n_heads;

    for (int l = 0; l < c.n_layers; l++) {
        ADVANCE_FP32(w.rms_att_weight[l], c.dim);
        ADVANCE_Q8(w.wq[l], w.wq_s[l], c.dim, c.dim);
        ADVANCE_Q8(w.wk[l], w.wk_s[l], kv_dim, c.dim);
        ADVANCE_Q8(w.wv[l], w.wv_s[l], kv_dim, c.dim);
        ADVANCE_Q8(w.wo[l], w.wo_s[l], c.dim, c.dim);
        ADVANCE_FP32(w.rms_ffn_weight[l], c.dim);
        ADVANCE_Q8(w.w1[l], w.w1_s[l], c.hidden_dim, c.dim);
        ADVANCE_Q8(w.w2[l], w.w2_s[l], c.dim, c.hidden_dim);
        ADVANCE_Q8(w.w3[l], w.w3_s[l], c.hidden_dim, c.dim);
    }
    
    ADVANCE_FP32(w.rms_final_weight, c.dim);
    ADVANCE_Q8(w.wcls, w.wcls_s, c.vocab_size, c.dim);

    load_vocab("vocab.bin");

    RunState s;
    s.x = calloc(c.dim, sizeof(float));
    s.xb = calloc(c.dim, sizeof(float));
    s.xb2 = calloc(c.dim, sizeof(float));
    s.hb = calloc(c.hidden_dim, sizeof(float));
    s.hb2 = calloc(c.hidden_dim, sizeof(float));
    s.q = calloc(c.dim, sizeof(float));
    s.k = calloc(kv_dim, sizeof(float));
    s.v = calloc(kv_dim, sizeof(float));
    s.att = calloc(c.n_heads * c.max_seq_len, sizeof(float));
    s.logits = calloc(c.vocab_size, sizeof(float));
    s.key_cache = calloc(c.n_layers * c.max_seq_len * kv_dim, sizeof(float));
    s.value_cache = calloc(c.n_layers * c.max_seq_len * kv_dim, sizeof(float));

    printf("Starting Quantized Generation:\n");
    int token = 1; // [CLS] token id
    printf("%s", vocab[token]);
    
    int pos = 0;
    while (pos < c.max_seq_len - 1) {
        float* logits = forward(&c, &w, &s, token, pos);
        
        int next_token = argmax(logits, c.vocab_size);
        char* token_str = vocab[next_token];
        
        if (strncmp(token_str, "Ġ", 3) == 0) printf(" %s", token_str + 3);
        else if (strncmp(token_str, " ", 1) == 0) printf(" %s", token_str + 1);
        else if (strcmp(token_str, "[PAD]") == 0 || strcmp(token_str, "[SEP]") == 0 || strcmp(token_str, "[EOS]") == 0) break;
        else printf("%s", token_str);
        
        fflush(stdout);
        token = next_token;
        pos++;
    }
    printf("\n");

    return 0;
}
