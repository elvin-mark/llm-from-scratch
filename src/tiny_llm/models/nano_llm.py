import torch
import torch.nn as nn

from tiny_llm.models.dense_llm import TransformerBlock
from tiny_llm.modules.norm import RMSNorm
from tiny_llm.modules.rope import precompute_freqs_cis


class NanoLLM(nn.Module):
    """
    NanoLLM: Ultra-compact, weight-tied Llama-style Transformer model.

    Ties embedding and output projection weights (self.output.weight = self.tok_embeddings.weight),
    eliminating 50% of parameter overhead from vocabulary matrices and enabling sub-1.2MB model footprints.
    """

    def __init__(
        self,
        vocab_size: int = 4000,
        dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int = 512,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len * 2)

        self.layers = nn.ModuleList(
            [TransformerBlock(dim=dim, n_heads=n_heads, ffn_dim=ffn_dim) for _ in range(n_layers)]
        )

        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # Weight Tying: Share memory between tok_embeddings and output head
        self.output.weight = self.tok_embeddings.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:seqlen].to(tokens.device)

        # Causal Attention Mask
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        h = self.norm(h)
        logits = self.output(h)
        return logits
