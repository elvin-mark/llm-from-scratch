import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tiny_llm.configs import TinyLLMConfig
from tiny_llm.modules import RMSNorm, apply_rotary_emb, precompute_freqs_cis
from tiny_llm.modules.bitlinear import BitLinear


class BitNetAttention(nn.Module):
    """
    Multi-Head Attention using 1.58-bit BitLinear projections.
    """

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = BitLinear(dim, n_heads * self.head_dim, bias=False)
        self.wk = BitLinear(dim, n_heads * self.head_dim, bias=False)
        self.wv = BitLinear(dim, n_heads * self.head_dim, bias=False)
        self.wo = BitLinear(n_heads * self.head_dim, dim, bias=False)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class BitNetFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network using 1.58-bit BitLinear projections.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = BitLinear(dim, hidden_dim, bias=False)
        self.w2 = BitLinear(hidden_dim, dim, bias=False)
        self.w3 = BitLinear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class BitNetBlock(nn.Module):
    """
    Single Transformer Block using 1.58-bit BitLinear Attention and SwiGLU FFN.
    """

    def __init__(self, dim: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.attention = BitNetAttention(dim, n_heads)
        self.feed_forward = BitNetFeedForward(dim, ffn_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class BitNetLLM(nn.Module):
    """
    1.58-bit Causal Language Model (BitNet b1.58 by Microsoft Research).
    All weight matrices in Attention and FFN blocks are quantized to ternary {-1, 0, +1} values.
    """

    def __init__(
        self,
        vocab_size: int = None,
        dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int = 512,
        max_seq_len: int = 128,
        config: TinyLLMConfig = None,
    ):
        super().__init__()
        if config is not None:
            vocab_size = config.vocab_size
            dim = config.dim
            n_layers = config.n_layers
            n_heads = config.n_heads
            ffn_dim = config.ffn_dim
            max_seq_len = config.max_seq_len

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([BitNetBlock(dim, n_heads, ffn_dim) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len * 2)

    def forward(self, tokens):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:seqlen].to(tokens.device)

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        h = self.norm(h)
        return self.output(h)
