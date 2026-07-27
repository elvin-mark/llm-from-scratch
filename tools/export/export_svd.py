import argparse
import os
import struct

import torch

from tiny_llm import NanoLLM, TinyLLM


def quantize_svd_matrix(w_fp32: torch.Tensor, rank: int):
    """
    Performs SVD Factorization W -> W_A * W_B and Int8 Quantization on factor matrices.
    W_A shape: [out_features, rank]
    W_B shape: [rank, in_features]
    """
    out_features, in_features = w_fp32.shape
    actual_rank = min(rank, out_features, in_features)

    # 1. Singular Value Decomposition
    U, S, Vh = torch.linalg.svd(w_fp32, full_matrices=False)
    U_r = U[:, :actual_rank]
    S_r = S[:actual_rank]
    Vh_r = Vh[:actual_rank, :]

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

    return w_a_int8, scale_a, w_b_int8, scale_b, actual_rank


def export_svd_model(
    checkpoint_path: str,
    output_path: str,
    tokenizer_path: str = "checkpoints/tokenizer.json",
    rank: int = 32,
    model_type: str = "tiny",
):
    print("⚡ Starting SVD Low-Rank + Int8 Quantization Exporter...")
    print(f"  Model Type:        {model_type}")
    print(f"  Truncated Rank:    {rank}")
    print(f"  Checkpoint Path:   {checkpoint_path}")
    print(f"  Output Binary:     {output_path}")

    vocab_size = 4000
    dim = 128
    n_layers = 4
    n_heads = 4
    ffn_dim = 512
    max_seq_len = 64

    model_cls = NanoLLM if model_type == "nano" else TinyLLM
    model = model_cls(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
    )

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"  Loaded trained PyTorch weights from '{checkpoint_path}'.")
    else:
        print(
            f"  ⚠️ Warning: Checkpoint '{checkpoint_path}' not found. Using randomly initialized weights for export demonstration."
        )

    model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "wb") as f:
        # Write 256-byte header
        # 8 ints: dim, ffn_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len, rank
        header = struct.pack(
            "iiiiiiii",
            dim,
            ffn_dim,
            n_layers,
            n_heads,
            n_heads,
            vocab_size,
            max_seq_len,
            rank,
        )
        f.write(header)
        f.write(b"\0" * (256 - len(header)))

        # 1. Token Embeddings (FP32)
        tok_emb = model.tok_embeddings.weight.detach().cpu().numpy().astype("float32")
        f.write(tok_emb.tobytes())

        # Helper to export an SVD Int8 Linear Projection
        def write_svd_layer(linear_module):
            w_fp32 = linear_module.weight.detach().cpu()
            w_a_int8, scale_a, w_b_int8, scale_b, _actual_r = quantize_svd_matrix(w_fp32, rank)

            f.write(w_a_int8.numpy().tobytes())
            f.write(struct.pack("f", scale_a))
            f.write(w_b_int8.numpy().tobytes())
            f.write(struct.pack("f", scale_b))

        # 2. Transformer Layers
        for layer in model.layers:
            # Attention RMSNorm
            f.write(layer.attention_norm.weight.detach().cpu().numpy().astype("float32").tobytes())

            # Attention Projections: wq, wk, wv, wo
            write_svd_layer(layer.attention.wq)
            write_svd_layer(layer.attention.wk)
            write_svd_layer(layer.attention.wv)
            write_svd_layer(layer.attention.wo)

            # FFN RMSNorm
            f.write(layer.ffn_norm.weight.detach().cpu().numpy().astype("float32").tobytes())

            # SwiGLU Projections: w1, w2, w3
            write_svd_layer(layer.feed_forward.w1)
            write_svd_layer(layer.feed_forward.w2)
            write_svd_layer(layer.feed_forward.w3)

        # 3. Final RMSNorm
        f.write(model.norm.weight.detach().cpu().numpy().astype("float32").tobytes())

        # 4. Output Projection Head (FP32)
        out_w = model.output.weight.detach().cpu().numpy().astype("float32")
        f.write(out_w.tobytes())

    # Export vocabulary file to c/vocab.bin as well
    export_vocab_bin("c/vocab.bin", tokenizer_path, vocab_size)

    file_size_mb = os.path.getsize(output_path) / (1024.0 * 1024.0)
    print(
        f"✅ Export Complete! Saved Hybrid SVD+Int8 model binary to '{output_path}' ({file_size_mb:.2f} MB)."
    )


def export_vocab_bin(vocab_out_path: str, tokenizer_path: str, vocab_size: int):
    import json

    os.makedirs(os.path.dirname(os.path.abspath(vocab_out_path)), exist_ok=True)
    tokens = [""] * vocab_size

    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            vocab = data.get("model", {}).get("vocab", {})
            for token, idx in vocab.items():
                if idx < vocab_size:
                    tokens[idx] = token

    with open(vocab_out_path, "wb") as f:
        for i in range(vocab_size):
            t_bytes = tokens[i].encode("utf-8") if tokens[i] else f"[{i}]".encode("utf-8")
            t_bytes = t_bytes[:31]
            f.write(bytes([len(t_bytes)]))
            f.write(t_bytes)
            f.write(b"\0" * (31 - len(t_bytes)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export PyTorch model to SVD+Int8 Quantized Binary format."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/tiny_llm.pth",
        help="Input PyTorch checkpoint path",
    )
    parser.add_argument(
        "--output", type=str, default="c/model_svd.bin", help="Output SVD model binary path"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="checkpoints/tokenizer.json", help="Path to tokenizer.json"
    )
    parser.add_argument("--rank", type=int, default=32, help="Truncated SVD rank")
    parser.add_argument(
        "--model", type=str, choices=["tiny", "nano"], default="tiny", help="Model architecture"
    )

    args = parser.parse_args()
    export_svd_model(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        tokenizer_path=args.tokenizer,
        rank=args.rank,
        model_type=args.model,
    )
