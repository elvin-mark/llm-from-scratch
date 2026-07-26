#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
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

// Weights mapped from memory (Host struct containing Device pointers)
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

// RunState buffers (Device)
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

// CUDA Kernels
__global__ void rmsnorm_kernel(float* o, float* x, float* weight, int size) {
    __shared__ float ss;
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += x[i] * x[i];
        }
        sum /= size;
        sum += 1e-6f;
        ss = 1.0f / sqrtf(sum);
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        o[idx] = weight[idx] * (x[idx] * ss);
    }
}

__global__ void rope_kernel(float* q, float* k, int dim, int kv_dim, int head_size, int pos) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i < dim) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);

        float q0 = q[i];
        float q1 = q[i+1];
        q[i]   = q0 * fcr - q1 * fci;
        q[i+1] = q0 * fci + q1 * fcr;
        
        if (i < kv_dim) {
            float k0 = k[i];
            float k1 = k[i+1];
            k[i]   = k0 * fcr - k1 * fci;
            k[i+1] = k0 * fci + k1 * fcr;
        }
    }
}

__global__ void swiglu_kernel(float* hb, float* hb2, int hidden_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        float val = hb[idx];
        val *= (1.0f / (1.0f + expf(-val)));
        val *= hb2[idx];
        hb[idx] = val;
    }
}

__global__ void add_kernel(float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] += y[idx];
    }
}

int argmax(float* logits, int size) {
    float* h_x = (float*)malloc(size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_x, logits, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    int max_i = 0;
    float max_val = h_x[0];
    for (int i = 1; i < size; i++) {
        if (h_x[i] > max_val) {
            max_val = h_x[i];
            max_i = i;
        }
    }
    free(h_x);
    return max_i;
}

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

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <prompt>\n", argv[0]);
        return 1;
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    int fd = open("model.bin", O_RDONLY);
    if (fd == -1) { printf("Compile and run python export.py first to generate model.bin\n"); return 1; }
    
    struct stat sb;
    fstat(fd, &sb);
    float* data = (float*)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) return 1;

    Config c;
    int* idata = (int*)data;
    c.dim = idata[0]; c.hidden_dim = idata[1]; c.n_layers = idata[2];
    c.n_heads = idata[3]; c.n_kv_heads = idata[4]; c.vocab_size = idata[5]; c.max_seq_len = idata[6];

    float* d_weights_data;
    size_t weights_size = sb.st_size - 256;
    CHECK_CUDA(cudaMalloc(&d_weights_data, weights_size));
    CHECK_CUDA(cudaMemcpy(d_weights_data, data + (256/sizeof(float)), weights_size, cudaMemcpyHostToDevice));

    float* w_ptr = d_weights_data;
    Weights w;
    w.token_embedding_table = w_ptr; w_ptr += c.vocab_size * c.dim;

    w.rms_att_weight = (float**)malloc(c.n_layers * sizeof(float*));
    w.wq = (float**)malloc(c.n_layers * sizeof(float*));
    w.wk = (float**)malloc(c.n_layers * sizeof(float*));
    w.wv = (float**)malloc(c.n_layers * sizeof(float*));
    w.wo = (float**)malloc(c.n_layers * sizeof(float*));
    w.rms_ffn_weight = (float**)malloc(c.n_layers * sizeof(float*));
    w.w1 = (float**)malloc(c.n_layers * sizeof(float*));
    w.w2 = (float**)malloc(c.n_layers * sizeof(float*));
    w.w3 = (float**)malloc(c.n_layers * sizeof(float*));

    for (int l = 0; l < c.n_layers; l++) {
        w.rms_att_weight[l] = w_ptr; w_ptr += c.dim;
        w.wq[l] = w_ptr; w_ptr += c.dim * c.dim;
        w.wk[l] = w_ptr; w_ptr += c.dim * c.dim;
        w.wv[l] = w_ptr; w_ptr += c.dim * c.dim;
        w.wo[l] = w_ptr; w_ptr += c.dim * c.dim;
        w.rms_ffn_weight[l] = w_ptr; w_ptr += c.dim;
        w.w1[l] = w_ptr; w_ptr += c.hidden_dim * c.dim;
        w.w2[l] = w_ptr; w_ptr += c.dim * c.hidden_dim;
        w.w3[l] = w_ptr; w_ptr += c.hidden_dim * c.dim;
    }
    
    w.rms_final_weight = w_ptr; w_ptr += c.dim;
    w.wcls = w_ptr; w_ptr += c.vocab_size * c.dim;

    load_vocab("vocab.bin");

    RunState s;
    CHECK_CUDA(cudaMalloc(&s.x, c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.xb, c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.xb2, c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.hb, c.hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.hb2, c.hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.q, c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.k, c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.v, c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.att, c.n_heads * c.max_seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.logits, c.vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.key_cache, c.n_layers * c.max_seq_len * c.dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.value_cache, c.n_layers * c.max_seq_len * c.dim * sizeof(float)));

    printf("Starting CUDA Generation:\n");
    int token = 1;
    printf("%s", vocab[token]);
    
    int pos = 0;
    int dim = c.dim;
    int hidden_dim = c.hidden_dim;
    int kv_dim = (c.dim * c.n_kv_heads) / c.n_heads;
    int head_size = dim / c.n_heads;
    int kv_mul = c.n_heads / c.n_kv_heads;

    float alpha = 1.0f;
    float beta = 0.0f;

    while (pos < c.max_seq_len - 1) {
        CHECK_CUDA(cudaMemcpy(s.x, w.token_embedding_table + token * dim, dim * sizeof(float), cudaMemcpyDeviceToDevice));

        for(int l = 0; l < c.n_layers; l++) {
            rmsnorm_kernel<<<1, dim>>>(s.xb, s.x, w.rms_att_weight[l], dim);
            
            // cuBLAS is col-major, so we transpose
            cublasSgemv(handle, CUBLAS_OP_T, dim, dim, &alpha, w.wq[l], dim, s.xb, 1, &beta, s.q, 1);
            cublasSgemv(handle, CUBLAS_OP_T, dim, kv_dim, &alpha, w.wk[l], dim, s.xb, 1, &beta, s.k, 1);
            cublasSgemv(handle, CUBLAS_OP_T, dim, kv_dim, &alpha, w.wv[l], dim, s.xb, 1, &beta, s.v, 1);

            rope_kernel<<<(dim+511)/512, 256>>>(s.q, s.k, dim, kv_dim, head_size, pos);

            int loff = l * c.max_seq_len * kv_dim;
            CHECK_CUDA(cudaMemcpy(s.key_cache + loff + pos * kv_dim, s.k, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(s.value_cache + loff + pos * kv_dim, s.v, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice));

            // Hybrid Attention (CPU) for brevity
            float* h_q = (float*)malloc(dim * sizeof(float));
            float* h_kcache = (float*)malloc(c.max_seq_len * kv_dim * sizeof(float));
            float* h_vcache = (float*)malloc(c.max_seq_len * kv_dim * sizeof(float));
            float* h_xb2 = (float*)malloc(dim * sizeof(float));
            
            CHECK_CUDA(cudaMemcpy(h_q, s.q, dim * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_kcache, s.key_cache + loff, (pos+1) * kv_dim * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_vcache, s.value_cache + loff, (pos+1) * kv_dim * sizeof(float), cudaMemcpyDeviceToHost));

            for (int h = 0; h < c.n_heads; h++) {
                float* q_head = h_q + h * head_size;
                float* att = (float*)malloc((pos+1) * sizeof(float));
                
                for (int t = 0; t <= pos; t++) {
                    float* k_head = h_kcache + t * kv_dim + (h / kv_mul) * head_size;
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) score += q_head[i] * k_head[i];
                    att[t] = score / sqrtf((float)head_size);
                }

                float max_val = att[0];
                for(int i=1; i<=pos; i++) if(att[i]>max_val) max_val=att[i];
                float sum=0;
                for(int i=0; i<=pos; i++) { att[i] = expf(att[i]-max_val); sum+=att[i]; }
                for(int i=0; i<=pos; i++) att[i]/=sum;

                float* xb_head = h_xb2 + h * head_size;
                memset(xb_head, 0, head_size * sizeof(float));
                for (int t = 0; t <= pos; t++) {
                    float* v_head = h_vcache + t * kv_dim + (h / kv_mul) * head_size;
                    for (int i = 0; i < head_size; i++) xb_head[i] += att[t] * v_head[i];
                }
                free(att);
            }
            CHECK_CUDA(cudaMemcpy(s.xb2, h_xb2, dim * sizeof(float), cudaMemcpyHostToDevice));
            free(h_q); free(h_kcache); free(h_vcache); free(h_xb2);

            cublasSgemv(handle, CUBLAS_OP_T, dim, dim, &alpha, w.wo[l], dim, s.xb2, 1, &beta, s.xb, 1);
            add_kernel<<<(dim+255)/256, 256>>>(s.x, s.xb, dim);

            rmsnorm_kernel<<<1, dim>>>(s.xb, s.x, w.rms_ffn_weight[l], dim);

            cublasSgemv(handle, CUBLAS_OP_T, dim, hidden_dim, &alpha, w.w1[l], dim, s.xb, 1, &beta, s.hb, 1);
            cublasSgemv(handle, CUBLAS_OP_T, dim, hidden_dim, &alpha, w.w3[l], dim, s.xb, 1, &beta, s.hb2, 1);
            
            swiglu_kernel<<<(hidden_dim+255)/256, 256>>>(s.hb, s.hb2, hidden_dim);

            cublasSgemv(handle, CUBLAS_OP_T, hidden_dim, dim, &alpha, w.w2[l], hidden_dim, s.hb, 1, &beta, s.xb, 1);
            add_kernel<<<(dim+255)/256, 256>>>(s.x, s.xb, dim);
        }

        rmsnorm_kernel<<<1, dim>>>(s.x, s.x, w.rms_final_weight, dim);
        cublasSgemv(handle, CUBLAS_OP_T, dim, c.vocab_size, &alpha, w.wcls, dim, s.x, 1, &beta, s.logits, 1);

        int next_token = argmax(s.logits, c.vocab_size);
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

    cublasDestroy(handle);
    return 0;
}
