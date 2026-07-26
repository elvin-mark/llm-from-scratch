import time

import torch

from tiny_llm import Attention, GroupedQueryAttention, precompute_freqs_cis


def run_attention_benchmark(
    dim: int = 128, n_heads: int = 8, n_kv_heads: int = 2, batch_size: int = 2
):
    print("=" * 75)
    print("📊 BENCHMARK 1: Multi-Head Attention (MHA) vs. Grouped Query Attention (GQA)")
    print("=" * 75)
    print(
        f"  Configuration: dim={dim}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, batch_size={batch_size}"
    )
    print("-" * 75)
    print(
        f"{'Seq Len':<10} | {'MHA KV-Cache (KB)':<18} | {'GQA KV-Cache (KB)':<18} | {'MHA Time (ms)':<14} | {'GQA Time (ms)':<14}"
    )
    print("-" * 75)

    mha = Attention(dim=dim, n_heads=n_heads)
    gqa = GroupedQueryAttention(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads)
    mha.eval()
    gqa.eval()

    seq_lengths = [64, 256, 1024, 4096]
    head_dim = dim // n_heads

    for seqlen in seq_lengths:
        # KV Cache Size in Bytes = 2 (K and V) * batch * n_heads * seqlen * head_dim * 4 bytes (FP32)
        mha_kv_kb = (2 * batch_size * n_heads * seqlen * head_dim * 4) / 1024.0
        gqa_kv_kb = (2 * batch_size * n_kv_heads * seqlen * head_dim * 4) / 1024.0

        x = torch.randn(batch_size, seqlen, dim)
        freqs_cis = precompute_freqs_cis(head_dim, seqlen)

        # Warmup & MHA Benchmark
        with torch.no_grad():
            for _ in range(5):
                _ = mha(x, freqs_cis)

            t0 = time.perf_counter()
            for _ in range(20):
                _ = mha(x, freqs_cis)
            mha_time_ms = ((time.perf_counter() - t0) / 20.0) * 1000.0

        # Warmup & GQA Benchmark
        with torch.no_grad():
            for _ in range(5):
                _ = gqa(x, freqs_cis)

            t0 = time.perf_counter()
            for _ in range(20):
                _ = gqa(x, freqs_cis)
            gqa_time_ms = ((time.perf_counter() - t0) / 20.0) * 1000.0

        print(
            f"{seqlen:<10} | {mha_kv_kb:<18.2f} | {gqa_kv_kb:<18.2f} | {mha_time_ms:<14.2f} | {gqa_time_ms:<14.2f}"
        )

    print("-" * 75)
    print("💡 Key Takeaway: GQA reduces KV-Cache memory footprint by 4x (n_heads / n_kv_heads = 4)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_attention_benchmark()
