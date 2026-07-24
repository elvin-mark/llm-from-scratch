import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear layer to add trainable low-rank A and B matrices (Hu et al.).

    Formula:
        h = W_0 * x + (alpha / r) * (x @ A^T) @ B^T
    """

    def __init__(self, base_linear: nn.Linear, r: int = 4, lora_alpha: float = 1.0):
        super().__init__()
        self.base_linear = base_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Freeze base linear layer weights and bias
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Low-rank matrices A and B
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))

        # Initialize A with Kaiming uniform and B with zeros (so step 0 is identical to base model)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward pass (frozen)
        result = self.base_linear(x)
        # Add LoRA delta: (x @ A.T) @ B.T * scaling
        lora_delta = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return result + lora_delta

    def get_merged_linear(self) -> nn.Linear:
        """
        Merges low-rank weights (W_0 + (alpha/r) * B @ A) into a single nn.Linear.
        """
        merged_weight = (
            self.base_linear.weight.data
            + (self.lora_B.data @ self.lora_A.data) * self.scaling
        )
        merged_linear = nn.Linear(
            self.base_linear.in_features,
            self.base_linear.out_features,
            bias=self.base_linear.bias is not None,
            device=self.base_linear.weight.device,
            dtype=self.base_linear.weight.dtype,
        )
        merged_linear.weight.data.copy_(merged_weight)
        if self.base_linear.bias is not None:
            merged_linear.bias.data.copy_(self.base_linear.bias.data)
        return merged_linear


def inject_lora(
    model: nn.Module, r: int = 4, lora_alpha: float = 1.0, target_modules=("wq", "wv")
) -> nn.Module:
    """
    Traverses the model and replaces target linear projections with LoRALinear wrappers.
    Freezes all non-LoRA parameters.
    """
    # Freeze all parameters in the model by default
    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        for attr in target_modules:
            if hasattr(module, attr):
                target_layer = getattr(module, attr)
                if isinstance(target_layer, nn.Linear) and not isinstance(
                    target_layer, LoRALinear
                ):
                    setattr(
                        module,
                        attr,
                        LoRALinear(target_layer, r=r, lora_alpha=lora_alpha),
                    )
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """
    Traverses the model and converts all LoRALinear layers back to standard nn.Linear layers
    with merged weights (W = W_0 + delta_W) for zero-latency inference export.
    """
    for module in model.modules():
        for attr_name in dir(module):
            try:
                attr = getattr(module, attr_name)
                if isinstance(attr, LoRALinear):
                    setattr(module, attr_name, attr.get_merged_linear())
            except Exception:
                continue
    return model
