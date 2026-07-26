# Mathematical Foundations of TinyLLM

This document provides the theoretical mathematical formulas defining the entire forward pass of the `TinyLLM` model. The architecture is a causal, decoder-only transformer heavily inspired by Llama 2/3.

## Consolidated Model Equations

The complete inference step of the network, from token IDs to output logits, can be summarized mathematically by the following consolidated equations:

$$
\begin{aligned}
H_0 &= \text{Embed}(X) \\
H_l &= F_l(H_{l-1}) \quad \text{for } l \in \{1, 2, \dots, N\} \\
\text{Logits}(X) &= \text{RMSNorm}_{\text{final}}(H_N) W_{\text{head}}^T \\
F_l(H) &= \tilde{H} + \left[ \text{SiLU}\left( \text{RMSNorm}_{\text{ffn}}(\tilde{H}) W_1^T \right) \odot \left( \text{RMSNorm}_{\text{ffn}}(\tilde{H}) W_3^T \right) \right] W_2^T \\
\text{where} \quad \tilde{H} &= H + A_l\big(\text{RMSNorm}_{\text{attn}}(H)\big) \\
A_l(x) &= \text{softmax}\left(\frac{\text{RoPE}(x W_q^T)\text{RoPE}(x W_k^T)^T}{\sqrt{d_{\text{head}}}} + M\right)(x W_v^T) W_o^T
\end{aligned}
$$

### 1. Token Embedding ($H_0$)
The input sequence of integer token IDs $X$ is mapped to dense continuous vectors via the learned embedding matrix.

### 2. The Transformer Block ($F_l$)
Each transformer layer $F_l$ processes the hidden states iteratively. The layer consists of an Attention block and a Feed-Forward Network (FFN), both utilizing the **Pre-Norm** architecture with residual connections.

The intermediate state after the Attention block is defined as $\tilde{H}$:

$$
\tilde{H} = H + A_l\big(\text{RMSNorm}_{\text{attn}}(H)\big)
$$

The final output of the layer applies the SwiGLU Feed-Forward Network:

$$
F_l(H) = \tilde{H} + \left[ \text{SiLU}\left( \text{RMSNorm}_{\text{ffn}}(\tilde{H}) W_1^T \right) \odot \left( \text{RMSNorm}_{\text{ffn}}(\tilde{H}) W_3^T \right) \right] W_2^T
$$

*Note: $\odot$ denotes the element-wise Hadamard product (the gating mechanism that makes SwiGLU powerful).*

### 3. Masked Multi-Head Attention ($A_l$)
The attention mechanism allows tokens to route information between each other. It applies **Rotary Positional Embeddings (RoPE)** to the Queries and Keys to inject relative positional data without requiring absolute position embeddings.

$$
A_l(x) = \text{softmax}\left(\frac{\text{RoPE}(x W_q^T)\text{RoPE}(x W_k^T)^T}{\sqrt{d_{\text{head}}}} + M\right)(x W_v^T) W_o^T
$$

Where:
- $W_q, W_k, W_v$: Query, Key, and Value projection weight matrices.
- $W_o$: Output projection weight matrix.
- $M$: The causal mask (a lower triangular matrix where upper triangle values are $-\infty$, preventing tokens from attending to future tokens).
- $\sqrt{d_{\text{head}}}$: The scaling factor to prevent softmax saturation.

### 4. Root Mean Square Normalization (RMSNorm)
Instead of standard LayerNorm, the model uses RMSNorm for computational efficiency. It removes the mean-centering operation and only scales by the root mean square of the activations, multiplied by a learned scale parameter $\gamma$:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma
$$
