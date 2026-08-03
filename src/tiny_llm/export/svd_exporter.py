import json
import os
import struct

import torch

from tiny_llm.models.factory import create_model


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


def export_svd(
    checkpoint_path: str = None,
    output_path: str = None,
    tokenizer_path: str = "checkpoints/tokenizer.json",
    rank: int = 32,
    arch: str = None,
    max_seq_len: int = 512,
):
    """Export model to Hybrid SVD Low-Rank + Int8 Quantized binary format for C engine."""
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        for p in [
            "checkpoints/tiny_llm.pth",
            "../checkpoints/tiny_llm.pth",
            "tiny_llm.pth",
            "../tiny_llm.pth",
            "checkpoints/nano_llm.pth",
            "checkpoints/moe_llm.pth",
        ]:
            if os.path.exists(p):
                checkpoint_path = p
                break
    if output_path is None:
        output_path = "model_svd.bin" if os.path.basename(os.getcwd()) == "c" else "c/model_svd.bin"

    print("⚡ Starting SVD Low-Rank + Int8 Quantization Exporter...")
    print(f"  Truncated Rank:    {rank}")
    print(f"  Checkpoint Path:   {checkpoint_path}")
    print(f"  Output Binary:     {output_path}")

    state_dict = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    else:
        print(
            f"  ⚠️ Warning: Checkpoint '{checkpoint_path}' not found. Using initialized model weights."
        )

    ckpt_dir = os.path.dirname(checkpoint_path) if checkpoint_path else "."
    base_name = os.path.splitext(checkpoint_path)[0] if checkpoint_path else "model"
    config_path = base_name + ".json"
    if not os.path.exists(config_path):
        config_path = os.path.join(ckpt_dir, "config.json")

    cfg = {}
    if os.path.exists(config_path) and state_dict is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded_cfg = json.load(f)
        if (
            "dim" not in loaded_cfg
            or loaded_cfg["dim"] == state_dict["tok_embeddings.weight"].shape[1]
        ):
            cfg = loaded_cfg

    if arch is None:
        arch = cfg.get("arch", None)
    if arch is None:
        if checkpoint_path and "nano" in checkpoint_path.lower():
            arch = "nano"
        elif checkpoint_path and (
            "moe" in checkpoint_path.lower()
            or (state_dict and any("experts" in k for k in state_dict.keys()))
        ):
            arch = "moe"
        elif checkpoint_path and (
            "bitnet" in checkpoint_path.lower()
            or (state_dict and any("weight_scale" in k or "gamma" in k for k in state_dict.keys()))
        ):
            arch = "bitnet"
        else:
            arch = "dense"

    print(f"  Auto-detected Model Architecture: {arch.upper()}")

    dim = (
        cfg.get("dim", state_dict["tok_embeddings.weight"].shape[1])
        if state_dict
        else cfg.get("dim", 128)
    )
    vocab_size = (
        cfg.get("vocab_size", state_dict["tok_embeddings.weight"].shape[0])
        if state_dict
        else cfg.get("vocab_size", 4000)
    )
    n_layers = (
        cfg.get(
            "n_layers",
            len([k for k in state_dict.keys() if k.endswith(".attention_norm.weight")]),
        )
        if state_dict
        else cfg.get("n_layers", 4)
    )
    n_heads = cfg.get("n_heads", 4)
    n_kv_heads = cfg.get("n_kv_heads", n_heads)

    if state_dict and "layers.0.feed_forward.w1.weight" in state_dict:
        ffn_dim = cfg.get("ffn_dim", state_dict["layers.0.feed_forward.w1.weight"].shape[0])
    else:
        ffn_dim = cfg.get("ffn_dim", 512)

    max_seq_len = cfg.get("max_seq_len", max_seq_len)

    model, _ = create_model(
        arch=arch,
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
    )
    if state_dict:
        model.load_state_dict(state_dict)
    model.eval()

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        # Write 256-byte header
        header = struct.pack(
            "iiiiiiii",
            dim,
            ffn_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            max_seq_len,
            rank,
        )
        f.write(header)
        f.write(b"\0" * (256 - len(header)))

        tok_emb = model.tok_embeddings.weight.detach().cpu().numpy().astype("float32")
        f.write(tok_emb.tobytes())

        def write_svd_layer(linear_module):
            w_fp32 = linear_module.weight.detach().cpu()
            w_a_int8, scale_a, w_b_int8, scale_b, _actual_r = quantize_svd_matrix(w_fp32, rank)

            f.write(w_a_int8.numpy().tobytes())
            f.write(struct.pack("f", scale_a))
            f.write(w_b_int8.numpy().tobytes())
            f.write(struct.pack("f", scale_b))

        for layer in model.layers:
            f.write(layer.attention_norm.weight.detach().cpu().numpy().astype("float32").tobytes())

            write_svd_layer(layer.attention.wq)
            write_svd_layer(layer.attention.wk)
            write_svd_layer(layer.attention.wv)
            write_svd_layer(layer.attention.wo)

            f.write(layer.ffn_norm.weight.detach().cpu().numpy().astype("float32").tobytes())

            if hasattr(layer.feed_forward, "experts"):
                write_svd_layer(layer.feed_forward.router.gate)
                for expert in layer.feed_forward.experts:
                    write_svd_layer(expert.w1)
                    write_svd_layer(expert.w2)
                    write_svd_layer(expert.w3)
            else:
                write_svd_layer(layer.feed_forward.w1)
                write_svd_layer(layer.feed_forward.w2)
                write_svd_layer(layer.feed_forward.w3)

        f.write(model.norm.weight.detach().cpu().numpy().astype("float32").tobytes())
        out_w = model.output.weight.detach().cpu().numpy().astype("float32")
        f.write(out_w.tobytes())

    export_vocab_bin("c/vocab.bin", tokenizer_path, vocab_size)
    file_size_mb = os.path.getsize(output_path) / (1024.0 * 1024.0)
    print(
        f"✅ Export Complete! Saved Hybrid SVD+Int8 model binary to '{output_path}' ({file_size_mb:.2f} MB)."
    )


def export_vocab_bin(vocab_out_path: str, tokenizer_path: str, vocab_size: int):
    if os.path.dirname(vocab_out_path):
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
