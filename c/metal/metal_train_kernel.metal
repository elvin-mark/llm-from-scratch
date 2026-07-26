#include <metal_stdlib>
using namespace metal;

// 1. Cross Entropy Loss & Gradient Metal Compute Shader
kernel void cross_entropy_loss_kernel(
    device float* loss_out [[buffer(0)]],
    device float* d_logits [[buffer(1)]],
    device const float* logits [[buffer(2)]],
    device const int* targets [[buffer(3)]],
    constant uint& vocab_size [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= 1) return;

    int target_tok = targets[0];
    if (target_tok < 0 || target_tok >= (int)vocab_size) return;

    // Softmax Max Trick
    float max_val = logits[0];
    for (uint i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    float sum_exp = 0.0f;
    for (uint i = 0; i < vocab_size; i++) {
        sum_exp += exp(logits[i] - max_val);
    }

    // Calculate Loss: -log(softmax[target])
    float target_logit = logits[target_tok];
    float prob_target = exp(target_logit - max_val) / sum_exp;
    loss_out[0] = -log(prob_target + 1e-7f);

    // Calculate Logit Gradients (d_logits = prob - 1_target)
    for (uint i = 0; i < vocab_size; i++) {
        float prob = exp(logits[i] - max_val) / sum_exp;
        d_logits[i] = (i == (uint)target_tok) ? (prob - 1.0f) : prob;
    }
}

// 2. AdamW Weight Update Metal Compute Shader
kernel void adamw_update_kernel(
    device float* weight [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* m [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    constant uint& size [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= size) return;

    float g = grad[id];
    float w = weight[id];

    // Weight decay step
    w -= lr * weight_decay * w;

    // Moment estimations
    m[id] = beta1 * m[id] + (1.0f - beta1) * g;
    v[id] = beta2 * v[id] + (1.0f - beta2) * (g * g);

    float m_hat = m[id] / (1.0f - beta1);
    float v_hat = v[id] / (1.0f - beta2);

    weight[id] = w - (lr * m_hat / (sqrt(v_hat) + eps));
}
