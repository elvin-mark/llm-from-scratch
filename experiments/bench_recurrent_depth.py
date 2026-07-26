import time

import torch
import torch.nn as nn

from tiny_llm import RMSNorm, TransformerBlock, precompute_freqs_cis


class RecurrentLoopLLM(nn.Module):
    """
    Recurrent Weight-Sharing Transformer (ALBERT / Universal Transformer style).
    Squeezes multi-layer depth out of a 1-layer parameter budget by looping through
    a single shared TransformerBlock sequentially multiple times.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_heads: int,
        ffn_dim: int,
        max_seq_len: int,
        num_loops: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_loops = num_loops
        self.max_seq_len = max_seq_len

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len * 2)

        # Single shared TransformerBlock repeated sequentially
        self.shared_layer = TransformerBlock(dim=dim, n_heads=n_heads, ffn_dim=ffn_dim)

        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:seqlen].to(tokens.device)

        # Sequential Recurrent Loop through the SAME shared block
        for _ in range(self.num_loops):
            h = self.shared_layer(h, freqs_cis, mask=None)

        h = self.norm(h)
        return self.output(h)


def run_recurrent_depth_benchmark():
    print("=" * 95)
    print("📊 BENCHMARK: Standard 4-Layer Transformer vs. 1-Layer Recurrent Loop Transformer")
    print("=" * 95)

    vocab_size = 4000
    dim = 128
    n_heads = 4
    ffn_dim = 512
    max_seq_len = 64
    batch_size = 2
    seqlen = 32

    # 1. Standard 4-Layer TinyLLM
    from tiny_llm import TinyLLM

    standard_model = TinyLLM(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=4,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
    )
    standard_model.eval()

    # 2. 1-Layer Recurrent Loop Model (Looped 4 times)
    recurrent_model = RecurrentLoopLLM(
        vocab_size=vocab_size,
        dim=dim,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
        num_loops=4,
    )
    recurrent_model.eval()

    std_params = sum(p.numel() for p in standard_model.parameters())
    rec_params = sum(p.numel() for p in recurrent_model.parameters())

    std_mem_mb = (std_params * 4.0) / (1024.0 * 1024.0)
    rec_mem_mb = (rec_params * 4.0) / (1024.0 * 1024.0)
    param_savings = (1.0 - (rec_params / std_params)) * 100.0

    dummy_input = torch.randint(0, vocab_size, (batch_size, seqlen))
    target_labels = torch.randint(0, vocab_size, (batch_size, seqlen))
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        # Standard Timing
        for _ in range(5):
            _ = standard_model(dummy_input)
        t0 = time.perf_counter()
        for _ in range(20):
            std_logits = standard_model(dummy_input)
        std_time_ms = ((time.perf_counter() - t0) / 20.0) * 1000.0
        std_loss = criterion(std_logits.view(-1, vocab_size), target_labels.view(-1)).item()

        # Recurrent Timing
        for _ in range(5):
            _ = recurrent_model(dummy_input)
        t0 = time.perf_counter()
        for _ in range(20):
            rec_logits = recurrent_model(dummy_input)
        rec_time_ms = ((time.perf_counter() - t0) / 20.0) * 1000.0
        rec_loss = criterion(rec_logits.view(-1, vocab_size), target_labels.view(-1)).item()

    print(
        f"  Architecture                 | {'Total Params':<13} | {'Memory (MB)':<12} | {'Effective Layers':<17} | {'Forward (ms)':<12} | {'Loss':<6}"
    )
    print("-" * 95)
    print(
        f"  Standard 4-Layer TinyLLM    | {std_params:<13,} | {std_mem_mb:<12.2f} | {4:<17} | {std_time_ms:<12.2f} | {std_loss:<6.3f}"
    )
    print(
        f"  Recurrent 1-Layer (4 Loops) | {rec_params:<13,} | {rec_mem_mb:<12.2f} | {4:<17} | {rec_time_ms:<12.2f} | {rec_loss:<6.3f}"
    )
    print("-" * 95)
    print(
        f"💡 Key Takeaway: Recurrent Weight-Sharing saves {param_savings:.1f}% of total model parameters ({std_params:,} -> {rec_params:,})"
    )
    print(
        "   by reusing a single Transformer block 4 times sequentially, providing 4-layer depth with 1-layer memory footprint!"
    )
    print("=" * 95 + "\n")


if __name__ == "__main__":
    run_recurrent_depth_benchmark()
