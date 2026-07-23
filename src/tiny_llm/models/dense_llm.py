import torch
import torch.nn as nn
from tiny_llm.configs import TinyLLMConfig
from tiny_llm.modules import RMSNorm, Attention, FeedForward, precompute_freqs_cis


class TransformerBlock(nn.Module):
    """
    A single Transformer layer combining Attention and FeedForward blocks with pre-RMSNorm.
    """

    def __init__(self, dim: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(dim, ffn_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class TinyLLM(nn.Module):
    """
    The main Dense Causal Language Model built using a Llama-like architecture.
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
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, n_heads, ffn_dim) for _ in range(n_layers)]
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
