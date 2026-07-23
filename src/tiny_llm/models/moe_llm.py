import torch
import torch.nn as nn
from tiny_llm.configs import MoELLMConfig
from tiny_llm.modules import (
    RMSNorm,
    GroupedQueryAttention,
    MoEFeedForward,
    precompute_freqs_cis,
)


class MoETransformerBlock(nn.Module):
    """
    A single Transformer layer combining GQA Attention and MoE FeedForward blocks.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        self.attention = GroupedQueryAttention(dim, n_heads, n_kv_heads)
        self.feed_forward = MoEFeedForward(
            dim, ffn_dim, num_experts=num_experts, top_k=top_k
        )
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class MoELLM(nn.Module):
    """
    Advanced Causal Language Model incorporating Grouped Query Attention (GQA) and Mixture-of-Experts (MoE).
    """

    def __init__(
        self,
        vocab_size: int = None,
        dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        n_kv_heads: int = 2,
        ffn_dim: int = 512,
        num_experts: int = 8,
        top_k: int = 2,
        max_seq_len: int = 128,
        config: MoELLMConfig = None,
    ):
        super().__init__()
        if config is not None:
            vocab_size = config.vocab_size
            dim = config.dim
            n_layers = config.n_layers
            n_heads = config.n_heads
            n_kv_heads = config.n_kv_heads
            ffn_dim = config.ffn_dim
            num_experts = config.num_experts
            top_k = config.num_experts_per_tok
            max_seq_len = config.max_seq_len

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                MoETransformerBlock(
                    dim, n_heads, n_kv_heads, ffn_dim, num_experts, top_k
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len * 2)

    def forward(self, tokens):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:seqlen].to(tokens.device)

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        h = self.norm(h)
        return self.output(h)
