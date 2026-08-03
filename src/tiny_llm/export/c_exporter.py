import json
import os
import struct

import torch
from tokenizers import Tokenizer

from tiny_llm.models.factory import create_model


def export_c(
    model_path=None,
    tokenizer_path=None,
    output_path=None,
    vocab_path=None,
    arch=None,
):
    """Export trained model checkpoint and tokenizer to bare-metal C binary format for any architecture."""
    if output_path is None:
        output_path = "model.bin" if os.path.basename(os.getcwd()) == "c" else "c/model.bin"
    if vocab_path is None:
        vocab_path = "vocab.bin" if os.path.basename(os.getcwd()) == "c" else "c/vocab.bin"

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
    if os.path.dirname(vocab_path):
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)

    print(f"Exporting model ({arch.upper()}) to {output_path}...")
    with open(output_path, "wb") as f:
        header = struct.pack(
            "iiiiiii", dim, ffn_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len
        )
        header += b"\x00" * (256 - len(header))
        f.write(header)

        def write_tensor(t):
            d = t.detach().cpu().to(torch.float32).numpy()
            f.write(d.tobytes())

        write_tensor(model.tok_embeddings.weight)
        for layer in model.layers:
            write_tensor(layer.attention_norm.weight)
            write_tensor(layer.attention.wq.weight)
            write_tensor(layer.attention.wk.weight)
            write_tensor(layer.attention.wv.weight)
            write_tensor(layer.attention.wo.weight)
            write_tensor(layer.ffn_norm.weight)

            if hasattr(layer.feed_forward, "experts"):
                # MoE FeedForward layer
                write_tensor(layer.feed_forward.router.gate.weight)
                for expert in layer.feed_forward.experts:
                    write_tensor(expert.w1.weight)
                    write_tensor(expert.w2.weight)
                    write_tensor(expert.w3.weight)
            else:
                # Standard SwiGLU FeedForward layer
                write_tensor(layer.feed_forward.w1.weight)
                write_tensor(layer.feed_forward.w2.weight)
                write_tensor(layer.feed_forward.w3.weight)

        write_tensor(model.norm.weight)
        write_tensor(model.output.weight)

    print(f"Exporting tokenizer to {vocab_path}...")
    vocab = tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    with open(vocab_path, "wb") as f:
        f.write(struct.pack("i", vocab_size))
        for i in range(vocab_size):
            token_str = inv_vocab.get(i, "").encode("utf-8")
            f.write(struct.pack("i", len(token_str)))
            f.write(token_str)

    print(f"Done! Saved {output_path} and {vocab_path}.")
