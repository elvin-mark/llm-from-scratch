import time

import torch

from tiny_llm import MoELLM, MoELLMConfig, TinyLLM


def run_moe_benchmark():
    print("=" * 80)
    print("📊 BENCHMARK 3: Dense (TinyLLM) vs. Sparse Mixture-of-Experts (MoELLM)")
    print("=" * 80)

    vocab_size = 4000
    dim = 128
    n_layers = 4
    n_heads = 4
    ffn_dim = 512
    num_experts = 8
    top_k = 2

    # 1. Instantiate Dense TinyLLM
    dense_model = TinyLLM(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
    )
    dense_model.eval()

    # 2. Instantiate Sparse MoELLM (8 experts, top-2 active)
    moe_config = MoELLMConfig(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=2,
        ffn_dim=ffn_dim,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
    )
    moe_model = MoELLM(config=moe_config)
    moe_model.eval()

    # Count parameters
    dense_total_params = sum(p.numel() for p in dense_model.parameters())
    moe_total_params = sum(p.numel() for p in moe_model.parameters())

    # Active parameters calculation (Dense uses 1 MLP, MoE uses 2 of 8 MLPs per token)
    single_mlp_params = dim * ffn_dim * 3  # w1, w2, w3
    moe_active_params = moe_total_params - (n_layers * (num_experts - top_k) * single_mlp_params)

    # Measure inference execution time
    dummy_input = torch.randint(0, vocab_size, (2, 32))

    with torch.no_grad():
        # Dense warmup & test
        for _ in range(5):
            _ = dense_model(dummy_input)
        t0 = time.perf_counter()
        for _ in range(20):
            _ = dense_model(dummy_input)
        dense_time = (time.perf_counter() - t0) / 20.0

        # MoE warmup & test
        for _ in range(5):
            _ = moe_model(dummy_input)
        t0 = time.perf_counter()
        for _ in range(20):
            _ = moe_model(dummy_input)
        moe_time = (time.perf_counter() - t0) / 20.0

    print(
        f"  Model Architecture    | {'Total Params':<16} | {'Active Params/Tok':<18} | {'Latency (ms)':<12}"
    )
    print("-" * 80)
    print(
        f"  Dense (TinyLLM)       | {dense_total_params:<16,} | {dense_total_params:<18,} | {dense_time * 1000:<12.2f}"
    )
    print(
        f"  Sparse (MoELLM, 8x2)  | {moe_total_params:<16,} | {moe_active_params:<18,} | {moe_time * 1000:<12.2f}"
    )
    print("-" * 80)
    print("💡 Key Takeaway: MoELLM increases total model capacity by ~3.5x while active compute")
    print(f"   per token stays lean (only {top_k} of {num_experts} experts active per token).")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_moe_benchmark()
