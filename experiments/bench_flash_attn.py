import time

import torch

from tiny_llm import Attention, EducationalFlashAttention, precompute_freqs_cis


def run_flash_attn_benchmark(
    dim: int = 128, n_heads: int = 4, batch_size: int = 2, block_size: int = 16
):
    print("=" * 80)
    print("📊 BENCHMARK 2: Standard Attention O(N^2) vs. Educational FlashAttention O(N)")
    print("=" * 80)
    print(
        f"  Configuration: dim={dim}, n_heads={n_heads}, batch_size={batch_size}, tile_block_size={block_size}"
    )
    print("-" * 80)
    print(
        f"{'Seq Len':<10} | {'Standard Alloc (KB)':<20} | {'Flash Tile Alloc (KB)':<22} | {'Speed Ratio':<12}"
    )
    print("-" * 80)

    standard_attn = Attention(dim=dim, n_heads=n_heads)
    flash_attn = EducationalFlashAttention(dim=dim, n_heads=n_heads, block_size=block_size)
    standard_attn.eval()
    flash_attn.eval()

    seq_lengths = [64, 128, 256, 512, 1024]
    head_dim = dim // n_heads

    for seqlen in seq_lengths:
        # Standard attention stores [batch, n_heads, seqlen, seqlen] scores matrix
        std_matrix_kb = (batch_size * n_heads * seqlen * seqlen * 4) / 1024.0

        # FlashAttention stores only local block tile [batch, n_heads, block_size, block_size]
        flash_tile_kb = (batch_size * n_heads * block_size * block_size * 4) / 1024.0

        x = torch.randn(batch_size, seqlen, dim)
        freqs_cis = precompute_freqs_cis(head_dim, seqlen)

        # Benchmark Standard Attention
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(10):
                _ = standard_attn(x, freqs_cis)
            std_time = (time.perf_counter() - t0) / 10.0

        # Benchmark FlashAttention
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(10):
                _ = flash_attn(x, freqs_cis)
            flash_time = (time.perf_counter() - t0) / 10.0

        speed_ratio = f"{std_time / flash_time:.2f}x" if flash_time > 0 else "N/A"
        print(f"{seqlen:<10} | {std_matrix_kb:<20.2f} | {flash_tile_kb:<22.2f} | {speed_ratio:<12}")

    print("-" * 80)
    print("💡 Key Takeaway: Standard Attention matrix allocation scales quadratically O(N^2),")
    print("   while FlashAttention memory allocation stays CONSTANT at tile size O(B_r^2).")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_flash_attn_benchmark()
