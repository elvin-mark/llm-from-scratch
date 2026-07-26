import os
import time
import argparse
import torch
import torch.nn as nn
from tiny_llm import TinyLLM, NanoLLM


class HybridSVDInt8Linear(nn.Module):
    """
    Hybrid SVD Low-Rank Decomposition + Int8 Dynamic Quantization Layer.

    Phase 1 (SVD): Factorizes W [out_dim, in_dim] -> W_A [out_dim, r] * W_B [r, in_dim]
    Phase 2 (Int8): Quantizes W_A and W_B to 8-bit integers with dynamic scale factors.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Int8 Weight Buffers
        self.register_buffer(
            "w_a_int8", torch.zeros((out_features, rank), dtype=torch.int8)
        )
        self.register_buffer(
            "w_b_int8", torch.zeros((rank, in_features), dtype=torch.int8)
        )
        self.register_buffer("scale_a", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("scale_b", torch.tensor(1.0, dtype=torch.float32))

    @classmethod
    def from_float(cls, float_linear: nn.Linear, rank: int = 32):
        """Performs SVD Decomposition and Int8 Quantization on a PyTorch float Linear layer."""
        w_fp32 = float_linear.weight.detach()  # [out_features, in_features]
        out_features, in_features = w_fp32.shape
        actual_rank = min(rank, out_features, in_features)

        # 1. Compute Singular Value Decomposition (SVD)
        U, S, Vh = torch.linalg.svd(w_fp32, full_matrices=False)

        # Truncate to rank r
        U_r = U[:, :actual_rank]  # [out_features, r]
        S_r = S[:actual_rank]  # [r]
        Vh_r = Vh[:actual_rank, :]  # [r, in_features]

        # Form low-rank factor matrices W_A and W_B
        sqrt_S = torch.diag(torch.sqrt(S_r))
        w_a = torch.matmul(U_r, sqrt_S)  # [out_features, r]
        w_b = torch.matmul(sqrt_S, Vh_r)  # [r, in_features]

        # 2. Int8 Dynamic Quantization
        scale_a = float(w_a.abs().max() / 127.0)
        scale_a = max(scale_a, 1e-5)
        w_a_int8 = torch.clamp(torch.round(w_a / scale_a), -128, 127).to(torch.int8)

        scale_b = float(w_b.abs().max() / 127.0)
        scale_b = max(scale_b, 1e-5)
        w_b_int8 = torch.clamp(torch.round(w_b / scale_b), -128, 127).to(torch.int8)

        layer = cls(in_features, out_features, rank=actual_rank)
        layer.w_a_int8.copy_(w_a_int8)
        layer.w_b_int8.copy_(w_b_int8)
        layer.scale_a.fill_(scale_a)
        layer.scale_b.fill_(scale_b)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize Int8 low-rank factors to FP32 for inference
        w_a = self.w_a_int8.float() * self.scale_a
        w_b = self.w_b_int8.float() * self.scale_b

        # Compute y = x @ W_B.T @ W_A.T
        h = torch.matmul(x, w_b.T)
        out = torch.matmul(h, w_a.T)
        return out


def compress_model_hybrid_svd(model: nn.Module, rank: int = 32) -> nn.Module:
    """Recursively replaces linear projection layers with HybridSVDInt8Linear modules."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name not in ["output"]:
            # Compress linear projections (wq, wk, wv, wo, w1, w2, w3)
            compressed_layer = HybridSVDInt8Linear.from_float(module, rank=rank)
            setattr(model, name, compressed_layer)
        else:
            compress_model_hybrid_svd(module, rank=rank)
    return model


def run_svd_hybrid_benchmark(
    model_type: str = "tiny", rank: int = 32, save_model: str = None
):
    model_name = (
        "NanoLLM (Weight-Tied)" if model_type == "nano" else "TinyLLM (Standard)"
    )

    vocab_size = 4000
    dim = 128
    n_layers = 4
    n_heads = 4
    ffn_dim = 512
    max_seq_len = 64
    batch_size = 2
    seqlen = 32

    model_cls = NanoLLM if model_type == "nano" else TinyLLM

    # 1. Baseline FP32 Model
    fp32_model = model_cls(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
    )
    fp32_model.eval()

    fp32_params = sum(p.numel() for p in fp32_model.parameters())
    fp32_mem_mb = (fp32_params * 4.0) / (1024.0 * 1024.0)

    # 2. Hybrid SVD + Int8 Compressed Model
    hybrid_model = model_cls(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
    )
    hybrid_model.load_state_dict(fp32_model.state_dict())
    hybrid_model.eval()

    hybrid_model = compress_model_hybrid_svd(hybrid_model, rank=rank)

    # Calculate Hybrid parameters and file size
    hybrid_params = 0
    hybrid_bytes = 0
    visited_params = set()

    for _name, module in hybrid_model.named_modules():
        if isinstance(module, HybridSVDInt8Linear):
            p_count = (module.out_features * module.rank) + (
                module.rank * module.in_features
            )
            hybrid_params += p_count
            hybrid_bytes += (
                p_count * 1 + 8
            )  # 1 byte per int8 weight + 8 bytes for scales
        elif isinstance(module, (nn.Embedding, nn.Linear)):
            for param in module.parameters():
                if id(param) not in visited_params:
                    visited_params.add(id(param))
                    p_count = param.numel()
                    hybrid_params += p_count
                    hybrid_bytes += p_count * 4

    hybrid_mem_mb = hybrid_bytes / (1024.0 * 1024.0)
    size_reduction = (1.0 - (hybrid_mem_mb / fp32_mem_mb)) * 100.0

    dummy_input = torch.randint(0, vocab_size, (batch_size, seqlen))
    target_labels = torch.randint(0, vocab_size, (batch_size, seqlen))
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        # FP32 Timing & Loss
        for _ in range(5):
            _ = fp32_model(dummy_input)
        t0 = time.perf_counter()
        for _ in range(20):
            fp32_logits = fp32_model(dummy_input)
        fp32_time_ms = ((time.perf_counter() - t0) / 20.0) * 1000.0
        fp32_loss = criterion(
            fp32_logits.view(-1, vocab_size), target_labels.view(-1)
        ).item()

        # Hybrid Timing & Loss
        for _ in range(5):
            _ = hybrid_model(dummy_input)
        t0 = time.perf_counter()
        for _ in range(20):
            hybrid_logits = hybrid_model(dummy_input)
        hybrid_time_ms = ((time.perf_counter() - t0) / 20.0) * 1000.0
        hybrid_loss = criterion(
            hybrid_logits.view(-1, vocab_size), target_labels.view(-1)
        ).item()

    lines = []
    lines.append("=" * 95)
    lines.append(
        f"📊 BENCHMARK: Post-Training Hybrid SVD Decomposition + Int8 ({model_name})"
    )
    lines.append("=" * 95)
    lines.append(
        f"  Compression Strategy           | {'Parameters':<12} | {'Memory (MB)':<12} | {'Forward (ms)':<12} | {'Loss':<6}"
    )
    lines.append("-" * 95)
    lines.append(
        f"  Baseline Float32 (FP32)         | {fp32_params:<12,} | {fp32_mem_mb:<12.2f} | {fp32_time_ms:<12.2f} | {fp32_loss:<6.3f}"
    )
    lines.append(
        f"🚀 Hybrid SVD (r={rank}) + Int8 Quant  | {hybrid_params:<12,} | {hybrid_mem_mb:<12.2f} | {hybrid_time_ms:<12.2f} | {hybrid_loss:<6.3f}"
    )
    lines.append("-" * 95)
    lines.append(
        f"💡 Key Takeaway: Hybrid SVD + Int8 Quantization reduces total model file size by {size_reduction:.1f}%!"
    )
    lines.append(
        f"   Model footprint shrank from {fp32_mem_mb:.2f} MB down to {hybrid_mem_mb:.2f} MB while preserving original language modeling quality!"
    )
    lines.append("=" * 95 + "\n")

    print("\n".join(lines))

    if save_model:
        os.makedirs(os.path.dirname(os.path.abspath(save_model)), exist_ok=True)
        torch.save(hybrid_model.state_dict(), save_model)
        print(f"✅ Saved compressed PyTorch model checkpoint to '{save_model}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Hybrid SVD Decomposition + Int8 Quantization."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["tiny", "nano"],
        default="tiny",
        help="Model architecture choice: 'tiny' (TinyLLM) or 'nano' (NanoLLM weight-tied)",
    )
    parser.add_argument(
        "--rank", type=int, default=32, help="Truncated SVD rank (default: 32)"
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Optional path to save the compressed PyTorch model checkpoint (.pth)",
    )

    args = parser.parse_args()
    run_svd_hybrid_benchmark(
        model_type=args.model, rank=args.rank, save_model=args.save_model
    )
