#include <metal_stdlib>
using namespace metal;

// 1. RMSNorm Metal Compute Shader
kernel void rmsnorm_kernel(
    device float* out [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= size) return;

    float ss = 0.0f;
    for (uint j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss = 1.0f / sqrt((ss / (float)size) + 1e-5f);
    out[id] = x[id] * ss * weight[id];
}

// 2. 1.58-Bit Ternary Addition Metal Compute Shader (0 FP Multiplications!)
kernel void bitnet_linear_kernel(
    device float* out [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const char* w [[buffer(2)]],
    constant uint& in_dim [[buffer(3)]],
    constant uint& out_dim [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= out_dim) return;

    // Absmax activation scaling
    float max_val = 0.0f;
    for (uint j = 0; j < in_dim; j++) {
        float abs_val = abs(x[j]);
        if (abs_val > max_val) max_val = abs_val;
    }
    float gamma_x = (max_val < 1e-5f) ? 1e-5f : max_val;
    float scale_x = 127.0f / gamma_x;

    // Pure Addition/Subtraction Loop
    int sum = 0;
    device const char* w_row = w + row * in_dim;

    for (uint j = 0; j < in_dim; j++) {
        char weight = w_row[j];
        int x_q = (int)round(x[j] * scale_x);

        if (weight == 1) {
            sum += x_q;      // Pure Addition
        } else if (weight == -1) {
            sum -= x_q;      // Pure Subtraction
        }
    }

    out[row] = (float)sum * (gamma_x / 127.0f);
}

// 3. Float32 Matmul Metal Compute Shader
kernel void matmul_float_kernel(
    device float* out [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* w [[buffer(2)]],
    constant uint& in_dim [[buffer(3)]],
    constant uint& out_dim [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= out_dim) return;

    float val = 0.0f;
    for (uint j = 0; j < in_dim; j++) {
        val += w[row * in_dim + j] * x[j];
    }
    out[row] = val;
}

// 4. SwiGLU Activation Metal Compute Shader
kernel void swiglu_kernel(
    device float* hb [[buffer(0)]],
    device const float* hb2 [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < size) {
        float val = hb[id];
        val *= (1.0f / (1.0f + exp(-val))) * hb2[id];
        hb[id] = val;
    }
}
