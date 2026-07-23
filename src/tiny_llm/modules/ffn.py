import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Standard SwiGLU Feed-Forward Network (Dense MLP).
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoERouter(nn.Module):
    """
    Top-K Router for gating tokens to expert feed-forward networks.
    """

    def __init__(self, dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x: torch.Tensor):
        logits = self.gate(x)  # [B, T, num_experts]
        probs = F.softmax(logits, dim=-1)
        routing_weights, selected_experts = torch.topk(probs, self.top_k, dim=-1)
        # Normalize routing weights across selected top-k experts
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        return routing_weights, selected_experts


class MoEFeedForward(nn.Module):
    """
    Mixture-of-Experts (MoE) Feed-Forward Network with SwiGLU Experts.
    """

    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = MoERouter(dim, num_experts, top_k)
        self.experts = nn.ModuleList(
            [FeedForward(dim, hidden_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        x_flat = x.view(-1, dim)  # [B*T, dim]

        routing_weights, selected_experts = self.router(x)  # [B, T, top_k]
        routing_weights = routing_weights.view(-1, self.top_k)  # [B*T, top_k]
        selected_experts = selected_experts.view(-1, self.top_k)  # [B*T, top_k]

        final_output = torch.zeros_like(x_flat)

        # Dispatch tokens to each expert
        for expert_idx, expert in enumerate(self.experts):
            token_indices, top_k_indices = torch.where(selected_experts == expert_idx)
            if token_indices.numel() == 0:
                continue

            expert_tokens = x_flat[token_indices]
            expert_out = expert(expert_tokens)

            weights = routing_weights[token_indices, top_k_indices].unsqueeze(-1)
            final_output.index_add_(0, token_indices, expert_out * weights)

        return final_output.view(bsz, seqlen, dim)
