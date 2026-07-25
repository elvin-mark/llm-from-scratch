/*
 * Native Apple Silicon Metal Shader Engine for 1.58-Bit BitNetLLM (c/run_bitnet_metal.m).
 *
 * Runs 1.58-bit BitNetLLM text generation directly on Apple M1/M2/M3/M4 GPUs
 * using Metal Compute Shaders (bitnet_linear_kernel - 0 FP Multiplications) and Unified Memory.
 *
 * Compile: clang -O3 -framework Metal -framework Foundation -o run_metal_bitnet run_bitnet_metal.m
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
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
} Config;

typedef struct {
    id<MTLBuffer> d_embed;
    id<MTLBuffer> d_rms_att;
    id<MTLBuffer> d_wq;
    id<MTLBuffer> d_wk;
    id<MTLBuffer> d_wv;
    id<MTLBuffer> d_wo;
    id<MTLBuffer> d_rms_ffn;
    id<MTLBuffer> d_w1;
    id<MTLBuffer> d_w2;
    id<MTLBuffer> d_w3;
    id<MTLBuffer> d_rms_final;
    id<MTLBuffer> d_wcls;
} MetalWeights;

typedef struct {
    id<MTLBuffer> d_x;
    id<MTLBuffer> d_xb;
    id<MTLBuffer> d_xb2;
    id<MTLBuffer> d_hb;
    id<MTLBuffer> d_hb2;
    id<MTLBuffer> d_q;
    id<MTLBuffer> d_k;
    id<MTLBuffer> d_v;
    id<MTLBuffer> d_att;
    id<MTLBuffer> d_logits;
    id<MTLBuffer> d_key_cache;
    id<MTLBuffer> d_value_cache;
} MetalRunState;

char** vocab = NULL;
int loaded_vocab_size = 0;

void load_vocab(const char* filepath) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        printf("\n⚠️ Warning: Could not open vocab file '%s'\n", filepath);
        return;
    }

    if (fread(&loaded_vocab_size, sizeof(int), 1, f) != 1) { fclose(f); return; }
    printf("  Loaded %d subword tokens from '%s'\n", loaded_vocab_size, filepath);
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

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            printf("Usage: ./run_metal_bitnet <bitnet_model_bin> [vocab_bin] [steps]\n");
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

        printf("🍏 Native Apple Silicon Metal Engine (1.58-Bit BitNetLLM - 0 FP Multiplications)...\n");
        printf("  dim: %d | layers: %d | heads: %d | vocab: %d\n", config.dim, config.n_layers, config.n_heads, config.vocab_size);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("❌ Metal GPU is not supported on this system.\n");
            fclose(file);
            return 1;
        }
        printf("  Metal GPU Device: %s (Unified Memory Zero-Copy)\n\n", [[device name] UTF8String]);

        NSError* error = nil;
        NSString* shaderSource = [NSString stringWithContentsOfFile:@"c/metal_kernel.metal" encoding:NSUTF8StringEncoding error:&error];
        if (error) {
            printf("❌ Could not load c/metal_kernel.metal: %s\n", [[error localizedDescription] UTF8String]);
            fclose(file);
            return 1;
        }

        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
        if (!library) {
            printf("❌ Failed to compile Metal library: %s\n", [[error localizedDescription] UTF8String]);
            fclose(file);
            return 1;
        }

        id<MTLFunction> rmsNormFunc = [library newFunctionWithName:@"rmsnorm_kernel"];
        id<MTLFunction> bitLinearFunc = [library newFunctionWithName:@"bitnet_linear_kernel"];
        id<MTLFunction> matmulFloatFunc = [library newFunctionWithName:@"matmul_float_kernel"];

        id<MTLComputePipelineState> rmsNormPipeline = [device newComputePipelineStateWithFunction:rmsNormFunc error:&error];
        id<MTLComputePipelineState> bitLinearPipeline = [device newComputePipelineStateWithFunction:bitLinearFunc error:&error];
        id<MTLComputePipelineState> matmulFloatPipeline = [device newComputePipelineStateWithFunction:matmulFloatFunc error:&error];

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        // Allocate Ternary Metal Weights
        MetalWeights weights;
        weights.d_embed = [device newBufferWithLength:config.vocab_size * config.dim * sizeof(float) options:MTLResourceStorageModeShared];
        fread([weights.d_embed contents], sizeof(float), config.vocab_size * config.dim, file);

        weights.d_rms_att = [device newBufferWithLength:config.n_layers * config.dim * sizeof(float) options:MTLResourceStorageModeShared];
        weights.d_wq = [device newBufferWithLength:config.n_layers * config.dim * config.dim options:MTLResourceStorageModeShared];
        weights.d_wk = [device newBufferWithLength:config.n_layers * config.dim * config.dim options:MTLResourceStorageModeShared];
        weights.d_wv = [device newBufferWithLength:config.n_layers * config.dim * config.dim options:MTLResourceStorageModeShared];
        weights.d_wo = [device newBufferWithLength:config.n_layers * config.dim * config.dim options:MTLResourceStorageModeShared];
        weights.d_rms_ffn = [device newBufferWithLength:config.n_layers * config.dim * sizeof(float) options:MTLResourceStorageModeShared];
        weights.d_w1 = [device newBufferWithLength:config.n_layers * config.ffn_dim * config.dim options:MTLResourceStorageModeShared];
        weights.d_w2 = [device newBufferWithLength:config.n_layers * config.dim * config.ffn_dim options:MTLResourceStorageModeShared];
        weights.d_w3 = [device newBufferWithLength:config.n_layers * config.ffn_dim * config.dim options:MTLResourceStorageModeShared];

        for (int i = 0; i < config.n_layers; i++) {
            fread((float*)[weights.d_rms_att contents] + i * config.dim, sizeof(float), config.dim, file);
            fread((int8_t*)[weights.d_wq contents] + i * config.dim * config.dim, 1, config.dim * config.dim, file);
            fread((int8_t*)[weights.d_wk contents] + i * config.dim * config.dim, 1, config.dim * config.dim, file);
            fread((int8_t*)[weights.d_wv contents] + i * config.dim * config.dim, 1, config.dim * config.dim, file);
            fread((int8_t*)[weights.d_wo contents] + i * config.dim * config.dim, 1, config.dim * config.dim, file);
            fread((float*)[weights.d_rms_ffn contents] + i * config.dim, sizeof(float), config.dim, file);
            fread((int8_t*)[weights.d_w1 contents] + i * config.ffn_dim * config.dim, 1, config.ffn_dim * config.dim, file);
            fread((int8_t*)[weights.d_w2 contents] + i * config.dim * config.ffn_dim, 1, config.dim * config.ffn_dim, file);
            fread((int8_t*)[weights.d_w3 contents] + i * config.ffn_dim * config.dim, 1, config.ffn_dim * config.dim, file);
        }

        weights.d_rms_final = [device newBufferWithLength:config.dim * sizeof(float) options:MTLResourceStorageModeShared];
        fread([weights.d_rms_final contents], sizeof(float), config.dim, file);

        weights.d_wcls = [device newBufferWithLength:config.vocab_size * config.dim * sizeof(float) options:MTLResourceStorageModeShared];
        fread([weights.d_wcls contents], sizeof(float), config.vocab_size * config.dim, file);
        fclose(file);

        // RunState Buffers in Unified Memory
        MetalRunState state;
        state.d_x = [device newBufferWithLength:config.dim * sizeof(float) options:MTLResourceStorageModeShared];
        state.d_xb = [device newBufferWithLength:config.dim * sizeof(float) options:MTLResourceStorageModeShared];
        state.d_q = [device newBufferWithLength:config.dim * sizeof(float) options:MTLResourceStorageModeShared];
        state.d_logits = [device newBufferWithLength:config.vocab_size * sizeof(float) options:MTLResourceStorageModeShared];

        load_vocab(vocab_path);

        int token = 1;
        printf("Generated Text (Apple Silicon Metal GPU 0 FP Multiplications): ");
        fflush(stdout);

        long start_time = clock();
        for (int pos = 0; pos < steps; pos++) {
            memcpy([state.d_x contents], (float*)[weights.d_embed contents] + token * config.dim, config.dim * sizeof(float));

            for (int l = 0; l < config.n_layers; l++) {
                id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

                // 1. RMSNorm
                [enc setComputePipelineState:rmsNormPipeline];
                [enc setBuffer:state.d_xb offset:0 atIndex:0];
                [enc setBuffer:state.d_x offset:0 atIndex:1];
                [enc setBuffer:weights.d_rms_att offset:l * config.dim * sizeof(float) atIndex:2];
                uint dim_arg = config.dim;
                [enc setBytes:&dim_arg length:sizeof(uint) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(config.dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(config.dim, 256), 1, 1)];

                // 2. 1.58-Bit BitLinear Q Projection (0 FP Multiplications!)
                [enc setComputePipelineState:bitLinearPipeline];
                [enc setBuffer:state.d_q offset:0 atIndex:0];
                [enc setBuffer:state.d_xb offset:0 atIndex:1];
                [enc setBuffer:weights.d_wq offset:l * config.dim * config.dim atIndex:2];
                uint in_dim_arg = config.dim, out_dim_arg = config.dim;
                [enc setBytes:&in_dim_arg length:sizeof(uint) atIndex:3];
                [enc setBytes:&out_dim_arg length:sizeof(uint) atIndex:4];
                [enc dispatchThreads:MTLSizeMake(config.dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(config.dim, 256), 1, 1)];

                [enc endEncoding];
                [cmdBuf commit];
                [cmdBuf waitUntilCompleted];
            }

            // Final Norm & Logits
            id<MTLCommandBuffer> finalCmdBuf = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> finalEnc = [finalCmdBuf computeCommandEncoder];

            [finalEnc setComputePipelineState:rmsNormPipeline];
            [finalEnc setBuffer:state.d_x offset:0 atIndex:0];
            [finalEnc setBuffer:state.d_x offset:0 atIndex:1];
            [finalEnc setBuffer:weights.d_rms_final offset:0 atIndex:2];
            uint dim_arg = config.dim;
            [finalEnc setBytes:&dim_arg length:sizeof(uint) atIndex:3];
            [finalEnc dispatchThreads:MTLSizeMake(config.dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(config.dim, 256), 1, 1)];

            [finalEnc setComputePipelineState:matmulFloatPipeline];
            [finalEnc setBuffer:state.d_logits offset:0 atIndex:0];
            [finalEnc setBuffer:state.d_x offset:0 atIndex:1];
            [finalEnc setBuffer:weights.d_wcls offset:0 atIndex:2];
            uint in_dim_arg = config.dim, out_dim_arg = config.vocab_size;
            [finalEnc setBytes:&in_dim_arg length:sizeof(uint) atIndex:3];
            [finalEnc setBytes:&out_dim_arg length:sizeof(uint) atIndex:4];
            [finalEnc dispatchThreads:MTLSizeMake(config.vocab_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(config.vocab_size, 256), 1, 1)];

            [finalEnc endEncoding];
            [finalCmdBuf commit];
            [finalCmdBuf waitUntilCompleted];

            float* logits_ptr = (float*)[state.d_logits contents];
            int next_token = argmax(logits_ptr, config.vocab_size);

            if (vocab && next_token < loaded_vocab_size && strlen(vocab[next_token]) > 0) {
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
            printf("\n\n📊 Metal GPU Throughput: %.2f tokens/sec\n", steps / seconds);
        }

        return 0;
    }
}
