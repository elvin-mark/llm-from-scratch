import json
import os
import struct

import numpy as np
import torch
from tokenizers import Tokenizer

from tiny_llm.models.factory import create_model


def export_q8(
    model_path=None,
    tokenizer_path=None,
    output_path=None,
    arch=None,
):
    """Export model to Int8 row-wise dynamic quantized binary format for any architecture."""
    if output_path is None:
        output_path = "model_q8.bin" if os.path.basename(os.getcwd()) == "c" else "c/model_q8.bin"

    if model_path is None:
        for p in [
            "checkpoints/tiny_llm.pth",
            "../checkpoints/tiny_llm.pth",
            "tiny_llm.pth",
            "../tiny_llm.pth",
            "checkpoints/nano_llm.pth",
            "checkpoints/moe_llm.pth",
            "checkpoints/bitnet_model.pth",
        ]:
            if os.path.exists(p):
                model_path = p
                break
    if tokenizer_path is None:
        for p in [
            "checkpoints/tokenizer.json",
            "../checkpoints/tokenizer.json",
            "tokenizer.json",
            "../tokenizer.json",
        ]:
            if os.path.exists(p):
                tokenizer_path = p
                break

    print(f"Loading model ({model_path}) and tokenizer ({tokenizer_path})...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    ckpt_dir = os.path.dirname(model_path) if model_path else "."
    base_name = os.path.splitext(model_path)[0] if model_path else "model"
    config_path = base_name + ".json"
    if not os.path.exists(config_path):
        config_path = os.path.join(ckpt_dir, "config.json")

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            loaded_cfg = json.load(f)
        if (
            "dim" not in loaded_cfg
            or loaded_cfg["dim"] == state_dict["tok_embeddings.weight"].shape[1]
        ):
            cfg = loaded_cfg

    # Auto-detect architecture
    if arch is None:
        arch = cfg.get("arch", None)
    if arch is None:
        if "nano" in model_path.lower():
            arch = "nano"
        elif "moe" in model_path.lower() or any("experts" in k for k in state_dict.keys()):
            arch = "moe"
        elif "bitnet" in model_path.lower() or any(
            "weight_scale" in k or "gamma" in k for k in state_dict.keys()
        ):
            arch = "bitnet"
        else:
            arch = "dense"

    print(f"  Architecture auto-detected: {arch.upper()}")

    dim = cfg.get("dim", state_dict["tok_embeddings.weight"].shape[1])
    vocab_size = cfg.get("vocab_size", state_dict["tok_embeddings.weight"].shape[0])
    n_layers = cfg.get(
        "n_layers",
        len([k for k in state_dict.keys() if k.endswith(".attention_norm.weight")]),
    )
    n_heads = cfg.get("n_heads", 4)
    n_kv_heads = cfg.get("n_kv_heads", n_heads)

    if "layers.0.feed_forward.w1.weight" in state_dict:
        ffn_dim = cfg.get("ffn_dim", state_dict["layers.0.feed_forward.w1.weight"].shape[0])
    elif "layers.0.feed_forward.experts.0.w1.weight" in state_dict:
        ffn_dim = cfg.get(
            "ffn_dim", state_dict["layers.0.feed_forward.experts.0.w1.weight"].shape[0]
        )
    else:
        ffn_dim = cfg.get("ffn_dim", 512)

    max_seq_len = cfg.get("max_seq_len", 64)

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
    model.load_state_dict(state_dict)
    model.eval()

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exporting Int8 Quantized model ({arch.upper()}) to {output_path}...")
    with open(output_path, "wb") as f:
        header = struct.pack(
            "iiiiiii", dim, ffn_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len
        )
        header += b"\x00" * (256 - len(header))
        f.write(header)

        def write_tensor_fp32(t):
            d = t.detach().cpu().to(torch.float32).numpy()
            f.write(d.tobytes())

        def write_quantized_matrix(t):
            d = t.detach().cpu().to(torch.float32).numpy()
            max_vals = np.max(np.abs(d), axis=1)
            scales = max_vals / 127.0
            scales[scales == 0] = 1.0

            q_weights = np.round(d / scales[:, None]).astype(np.int8)

            f.write(q_weights.tobytes())
            f.write(scales.astype(np.float32).tobytes())

        write_tensor_fp32(model.tok_embeddings.weight)
        for layer in model.layers:
            write_tensor_fp32(layer.attention_norm.weight)
            write_quantized_matrix(layer.attention.wq.weight)
            write_quantized_matrix(layer.attention.wk.weight)
            write_quantized_matrix(layer.attention.wv.weight)
            write_quantized_matrix(layer.attention.wo.weight)
            write_tensor_fp32(layer.ffn_norm.weight)

            if hasattr(layer.feed_forward, "experts"):
                write_quantized_matrix(layer.feed_forward.router.gate.weight)
                for expert in layer.feed_forward.experts:
                    write_quantized_matrix(expert.w1.weight)
                    write_quantized_matrix(expert.w2.weight)
                    write_quantized_matrix(expert.w3.weight)
            else:
                write_quantized_matrix(layer.feed_forward.w1.weight)
                write_quantized_matrix(layer.feed_forward.w2.weight)
                write_quantized_matrix(layer.feed_forward.w3.weight)

        write_tensor_fp32(model.norm.weight)
        write_tensor_fp32(model.output.weight)

    print(f"Done! Saved {output_path}.")
