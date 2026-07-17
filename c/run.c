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
} Config;

// Weights mapped from memory
typedef struct {
    float* token_embedding_table;
    float** rms_att_weight;
    float** wq;
    float** wk;
    float** wv;
    float** wo;
    float** rms_ffn_weight;
    float** w1;
    float** w2;
    float** w3;
    float* rms_final_weight;
    float* wcls;
} Weights;

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
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-6f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (x[j] * ss);
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W is shape (d, n) -> transposed logic based on PyTorch Linear
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
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

// Forward pass
float* forward(Config* p, Weights* w, RunState* s, int token, int pos) {
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int head_size = dim / p->n_heads;

    // 1. Copy token embedding
    memcpy(s->x, w->token_embedding_table + token * dim, dim * sizeof(float));

    // For each layer
    for(int l = 0; l < p->n_layers; l++) {
        // Attention norm
        rmsnorm(s->xb, s->x, w->rms_att_weight[l], dim);

        // Q, K, V
        matmul(s->q, s->xb, w->wq[l], dim, dim);
        matmul(s->k, s->xb, w->wk[l], dim, kv_dim);
        matmul(s->v, s->xb, w->wv[l], dim, kv_dim);

        // RoPE
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);

            // Q
            float q0 = s->q[i];
            float q1 = s->q[i+1];
            s->q[i]   = q0 * fcr - q1 * fci;
            s->q[i+1] = q0 * fci + q1 * fcr;
            
            // K
            if (i < kv_dim) {
                float k0 = s->k[i];
                float k1 = s->k[i+1];
                s->k[i]   = k0 * fcr - k1 * fci;
                s->k[i+1] = k0 * fci + k1 * fcr;
            }
        }

        // Cache KV
        int loff = l * p->max_seq_len * kv_dim;
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(float));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(float));

        // Attention
        for (int h = 0; h < p->n_heads; h++) {
            float* q_head = s->q + h * head_size;
            float* att_head = s->att + h * p->max_seq_len;
            
            for (int t = 0; t <= pos; t++) {
                float* k_head = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q_head[i] * k_head[i];
                }
                score /= sqrtf(head_size);
                att_head[t] = score;
            }

            softmax(att_head, pos + 1);

            float* xb_head = s->xb2 + h * head_size;
            memset(xb_head, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v_head = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att_head[t];
                for (int i = 0; i < head_size; i++) {
                    xb_head[i] += a * v_head[i];
                }
            }
        }

        // Wo
        matmul(s->xb, s->xb2, w->wo[l], dim, dim);

        // Residual
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i];
        }

        // FFN norm
        rmsnorm(s->xb, s->x, w->rms_ffn_weight[l], dim);

        // FFN
        matmul(s->hb, s->xb, w->w1[l], dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3[l], dim, hidden_dim);
        
        // SwiGLU
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        matmul(s->xb, s->hb, w->w2[l], hidden_dim, dim);

        // Residual
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i];
        }
    }

    // Final Norm & Logits
    rmsnorm(s->x, s->x, w->rms_final_weight, dim);
    matmul(s->logits, s->x, w->wcls, dim, p->vocab_size);

    return s->logits;
}

// Tokenizer and Vocab
char** vocab;
int vocab_size;

void load_vocab(const char* filepath) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        printf("Error opening %s\n", filepath);
        exit(1);
    }
    fread(&vocab_size, sizeof(int), 1, f);
    vocab = (char**)malloc(vocab_size * sizeof(char*));
    for (int i = 0; i < vocab_size; i++) {
        int len;
        fread(&len, sizeof(int), 1, f);
        vocab[i] = (char*)malloc(len + 1);
        if(len > 0) fread(vocab[i], 1, len, f);
        vocab[i][len] = '\0';
    }
    fclose(f);
}

int main() {
    // Load weights
    int fd = open("model.bin", O_RDONLY);
    if (fd == -1) { printf("Cannot open model.bin. Run export.py first!\n"); return 1; }
    
    struct stat sb;
    fstat(fd, &sb);
    float* data = (float*)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { printf("mmap failed\n"); return 1; }

    Config c;
    int* idata = (int*)data;
    c.dim = idata[0];
    c.hidden_dim = idata[1];
    c.n_layers = idata[2];
    c.n_heads = idata[3];
    c.n_kv_heads = idata[4];
    c.vocab_size = idata[5];
    c.max_seq_len = idata[6];

    float* weights_ptr = data + (256 / sizeof(float)); // Header is 256 bytes

    Weights w;
    w.token_embedding_table = weights_ptr; weights_ptr += c.vocab_size * c.dim;

    w.rms_att_weight = malloc(c.n_layers * sizeof(float*));
    w.wq = malloc(c.n_layers * sizeof(float*));
    w.wk = malloc(c.n_layers * sizeof(float*));
    w.wv = malloc(c.n_layers * sizeof(float*));
    w.wo = malloc(c.n_layers * sizeof(float*));
    w.rms_ffn_weight = malloc(c.n_layers * sizeof(float*));
    w.w1 = malloc(c.n_layers * sizeof(float*));
    w.w2 = malloc(c.n_layers * sizeof(float*));
    w.w3 = malloc(c.n_layers * sizeof(float*));

    for (int l = 0; l < c.n_layers; l++) {
        w.rms_att_weight[l] = weights_ptr; weights_ptr += c.dim;
        w.wq[l] = weights_ptr; weights_ptr += c.dim * c.dim;
        w.wk[l] = weights_ptr; weights_ptr += c.dim * c.dim;
        w.wv[l] = weights_ptr; weights_ptr += c.dim * c.dim;
        w.wo[l] = weights_ptr; weights_ptr += c.dim * c.dim;
        w.rms_ffn_weight[l] = weights_ptr; weights_ptr += c.dim;
        w.w1[l] = weights_ptr; weights_ptr += c.hidden_dim * c.dim;
        w.w2[l] = weights_ptr; weights_ptr += c.dim * c.hidden_dim;
        w.w3[l] = weights_ptr; weights_ptr += c.hidden_dim * c.dim;
    }
    
    w.rms_final_weight = weights_ptr; weights_ptr += c.dim;
    w.wcls = weights_ptr; weights_ptr += c.vocab_size * c.dim;

    // Load Vocab
    load_vocab("vocab.bin");

    // Alloc RunState
    RunState s;
    s.x = calloc(c.dim, sizeof(float));
    s.xb = calloc(c.dim, sizeof(float));
    s.xb2 = calloc(c.dim, sizeof(float));
    s.hb = calloc(c.hidden_dim, sizeof(float));
    s.hb2 = calloc(c.hidden_dim, sizeof(float));
    s.q = calloc(c.dim, sizeof(float));
    s.k = calloc(c.dim, sizeof(float));
    s.v = calloc(c.dim, sizeof(float));
    s.att = calloc(c.n_heads * c.max_seq_len, sizeof(float));
    s.logits = calloc(c.vocab_size, sizeof(float));
    s.key_cache = calloc(c.n_layers * c.max_seq_len * c.dim, sizeof(float));
    s.value_cache = calloc(c.n_layers * c.max_seq_len * c.dim, sizeof(float));

    printf("Starting Generation:\n");
    int token = 1; // [CLS] token id is 1
    printf("%s", vocab[token]);
    
    int pos = 0;
    while (pos < c.max_seq_len - 1) {
        float* logits = forward(&c, &w, &s, token, pos);
        
        // Next token (greedy argmax)
        int next_token = argmax(logits, c.vocab_size);
        
        char* token_str = vocab[next_token];
        
        // HuggingFace BPE cleanup logic for display
        if (strncmp(token_str, "Ġ", 3) == 0) printf(" %s", token_str + 3); // Ġ represents space
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
