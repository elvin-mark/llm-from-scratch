import torch
import torch.nn as nn
import torch.nn.functional as F


class STETernaryQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for Ternary Weight Quantization {-1, 0, +1} (BitNet b1.58).

    Forward Pass:
        gamma = mean(|W|)
        W_tilde = Clip(Round(W / gamma), -1, +1) * gamma

    Backward Pass:
        dL/dW = dL/dW_tilde (Gradients pass straight through to FP32 master weights)
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor) -> torch.Tensor:
        gamma = weight.abs().mean().clamp(min=1e-5)
        weight_scaled = weight / gamma
        quantized = torch.round(weight_scaled).clamp(-1.0, 1.0)
        return quantized * gamma

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class BitLinear(nn.Module):
    """
    1.58-bit BitLinear Layer (BitNet b1.58 by Microsoft Research).
    Ternarizes weights to {-1, 0, +1} during forward pass while keeping FP32 master weights for training.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize activations (absmax scaling)
        gamma_x = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_quant = (x / gamma_x) * 127.0
        x_quant = x_quant.clamp(-128.0, 127.0)

        # Quantize weights to ternary {-1, 0, +1} via STE
        w_quant = STETernaryQuantize.apply(self.weight)

        # Matmul using ternary weights
        out = F.linear(x_quant, w_quant, self.bias)

        # De-scale output back to continuous range
        return out * (gamma_x / 127.0)
