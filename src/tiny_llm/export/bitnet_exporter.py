import json
import os
import struct

import torch

from tiny_llm.models.factory import create_model
from tiny_llm.modules import STETernaryQuantize
from tiny_llm.tokenizer import ScratchTokenizer


def export_bitnet(
    model_path: str = None,
    tokenizer_path: str = None,
    output_path: str = None,
    arch: str = "bitnet",
):
    """Export BitNet 1.58-bit model checkpoint into packed ternary bitstream binary format."""
    if model_path is None:
        for p in [
            "checkpoints/bitnet_model.pth",
            "../checkpoints/bitnet_model.pth",
            "bitnet_model.pth",
            "../bitnet_model.pth",
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
    if output_path is None:
        output_path = (
            "model_bitnet.bin" if os.path.basename(os.getcwd()) == "c" else "c/model_bitnet.bin"
        )

    tokenizer = ScratchTokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    state_dict = None
    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    else:
        print(f"  ⚠️ Warning: Checkpoint '{model_path}' not found. Using initialized model weights.")

    ckpt_dir = os.path.dirname(model_path) if model_path else "."
    base_name = os.path.splitext(model_path)[0] if model_path else "model"
    config_path = base_name + ".json"
    if not os.path.exists(config_path):
        config_path = os.path.join(ckpt_dir, "config.json")

    cfg = {}
    if os.path.exists(config_path) and state_dict is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    arch = cfg.get("arch", arch)
    dim = (
        cfg.get("dim", state_dict["tok_embeddings.weight"].shape[1])
        if state_dict
        else cfg.get("dim", 128)
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

    max_seq_len = cfg.get("max_seq_len", 128)

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

    print(f"Exporting 1.58-bit BitNet model ({arch.upper()}) to {output_path}...")
    with open(output_path, "wb") as f:
        header = struct.pack(
            "iiiiiii", dim, ffn_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len
        )
        header += b"\x00" * (256 - len(header))
        f.write(header)

        def write_fp32(t):
            d = t.detach().cpu().to(torch.float32).numpy()
            f.write(d.tobytes())

        def write_ternary_packed(weight_tensor):
            gamma = weight_tensor.abs().mean().item()
            w_quant = STETernaryQuantize.apply(weight_tensor)
            w_int = w_quant.detach().cpu().to(torch.int8).numpy().flatten()

            f.write(struct.pack("f", gamma))

            packed_bytes = bytearray()
            for i in range(0, len(w_int), 4):
                chunk = w_int[i : i + 4]
                packed_byte = 0
                for j, val in enumerate(chunk):
                    code = 0 if val == 0 else (1 if val > 0 else 2)
                    packed_byte |= code << (j * 2)
                packed_bytes.append(packed_byte)

            f.write(bytes(packed_bytes))

        write_fp32(model.tok_embeddings.weight)

        for layer in model.layers:
            write_fp32(layer.attention_norm.weight)
            write_ternary_packed(layer.attention.wq.weight)
            write_ternary_packed(layer.attention.wk.weight)
            write_ternary_packed(layer.attention.wv.weight)
            write_ternary_packed(layer.attention.wo.weight)
            write_fp32(layer.ffn_norm.weight)
            write_ternary_packed(layer.feed_forward.w1.weight)
            write_ternary_packed(layer.feed_forward.w2.weight)
            write_ternary_packed(layer.feed_forward.w3.weight)

        write_fp32(model.norm.weight)
        write_fp32(model.output.weight)

    print(f"Done! Saved BitNet binary to {output_path}")
