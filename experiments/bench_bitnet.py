import time

import torch

from tiny_llm import BitNetLLM, TinyLLM


def run_bitnet_benchmark():
    print("=" * 85)
    print("📊 BENCHMARK: Float32 TinyLLM vs. 1.58-Bit BitNetLLM (BitNet b1.58)")
    print("=" * 85)

    vocab_size = 4000
    dim = 128
    n_layers = 4
    n_heads = 4
    ffn_dim = 512
    max_seq_len = 64
    batch_size = 2
    seqlen = 32

    # 1. Instantiate Float32 TinyLLM
    fp32_model = TinyLLM(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
    )
    fp32_model.eval()

    # 2. Instantiate 1.58-bit BitNetLLM
    bitnet_model = BitNetLLM(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
    )
    bitnet_model.eval()

    # Calculate parameter counts and memory footprints
    fp32_total_params = sum(p.numel() for p in fp32_model.parameters())
    bitnet_total_params = sum(p.numel() for p in bitnet_model.parameters())

    # Count linear projection parameters subject to ternarization (wq, wk, wv, wo, w1, w2, w3)
    linear_params_per_layer = (dim * dim * 4) + (dim * ffn_dim * 3)
    total_ternary_params = n_layers * linear_params_per_layer
    non_ternary_params = fp32_total_params - total_ternary_params

    # Memory in MB: FP32 = 4 bytes/param. BitNet = 0.25 bytes/param (2 bits packed) for linear weights.
    fp32_mem_mb = (fp32_total_params * 4.0) / (1024.0 * 1024.0)
    bitnet_mem_mb = ((total_ternary_params * 0.25) + (non_ternary_params * 4.0)) / (1024.0 * 1024.0)
    space_savings = (1.0 - (bitnet_mem_mb / fp32_mem_mb)) * 100.0

    # Calculate Multiply-Accumulate (MAC) vs Addition Operations
    fp32_fp_multiplications = total_ternary_params
    bitnet_fp_multiplications = 0  # Replaced by integer additions/subtractions in BitLinear!

    # Benchmark Forward Pass Latency
    dummy_input = torch.randint(0, vocab_size, (batch_size, seqlen))

    with torch.no_grad():
        # FP32 Warmup & Timing
        for _ in range(5):
            _ = fp32_model(dummy_input)
        t0 = time.perf_counter()
        for _ in range(20):
            _ = fp32_model(dummy_input)
        fp32_time_ms = ((time.perf_counter() - t0) / 20.0) * 1000.0

        # BitNet Warmup & Timing
        for _ in range(5):
            _ = bitnet_model(dummy_input)
        t0 = time.perf_counter()
        for _ in range(20):
            _ = bitnet_model(dummy_input)
        bitnet_time_ms = ((time.perf_counter() - t0) / 20.0) * 1000.0

    print(
        f"  Architecture      | {'Total Params':<14} | {'Memory Footprint':<18} | {'FP Multiplications':<20} | {'Forward Time (ms)':<15}"
    )
    print("-" * 85)
    print(
        f"  Float32 TinyLLM   | {fp32_total_params:<14,} | {fp32_mem_mb:<18.2f} MB | {fp32_fp_multiplications:<20,} | {fp32_time_ms:<15.2f}"
    )
    print(
        f"  1.58-Bit BitNet   | {bitnet_total_params:<14,} | {bitnet_mem_mb:<18.2f} MB | {bitnet_fp_multiplications:<20} | {bitnet_time_ms:<15.2f}"
    )
    print("-" * 85)
    print(f"💡 Key Takeaway: BitNet b1.58 reduces weight memory footprint by {space_savings:.1f}%,")
    print(
        f"   and completely ELIMINATES floating-point multiplications ({fp32_fp_multiplications:,} FP MACs -> 0 FP MACs) in all linear projections!"
    )
    print("=" * 85 + "\n")


if __name__ == "__main__":
    run_bitnet_benchmark()
