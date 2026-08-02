import os
import struct
import sys

import torch

# Add parent directory to path to import TinyLLM
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers import Tokenizer

from tiny_llm.model import TinyLLM


def export_model_q8(model_path=None, tokenizer_path=None, output_path=None):
    if output_path is None:
        output_path = "model_q8.bin" if os.path.basename(os.getcwd()) == "c" else "c/model_q8.bin"

    if model_path is None:
        for p in [
            "checkpoints/tiny_llm.pth",
            "../checkpoints/tiny_llm.pth",
            "tiny_llm.pth",
            "../tiny_llm.pth",
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

    ckpt_dir = os.path.dirname(model_path)
    base_name = os.path.splitext(model_path)[0]
    config_path = base_name + ".json"
    if not os.path.exists(config_path):
        config_path = os.path.join(ckpt_dir, "config.json")

    state_dict = torch.load(model_path, map_location="cpu")

    if os.path.exists(config_path):
        import json

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        dim = cfg.get("dim", state_dict["tok_embeddings.weight"].shape[1])
        n_layers = cfg.get(
            "n_layers",
            len([k for k in state_dict.keys() if k.endswith(".attention_norm.weight")]),
        )
        n_heads = cfg.get("n_heads", 4)
        ffn_dim = cfg.get(
            "ffn_dim", state_dict["layers.0.feed_forward.w1.weight"].shape[0]
        )
        max_seq_len = cfg.get("max_seq_len", 64)
    else:
        dim = state_dict["tok_embeddings.weight"].shape[1]
        n_layers = len(
            [k for k in state_dict.keys() if k.endswith(".attention_norm.weight")]
        )
        n_heads = 4
        ffn_dim = state_dict["layers.0.feed_forward.w1.weight"].shape[0]
        max_seq_len = 64

    model = TinyLLM(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
    )
    model.load_state_dict(state_dict)
    model.eval()

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exporting model to {output_path}...")
    with open(output_path, "wb") as f:
        # Write header (256 bytes)
        # struct format: 8 ints (dim, ffn_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len, is_quantized)
        header = struct.pack(
            "iiiiiiii",
            dim,
            ffn_dim,
            n_layers,
            n_heads,
            n_heads,
            vocab_size,
            max_seq_len,
            1,
        )
        # Pad with zeros to 256 bytes
        header += b"\x00" * (256 - len(header))
        f.write(header)

        # Helper to write FP32 tensor
        def write_tensor_fp32(t):
            d = t.detach().cpu().to(torch.float32).numpy()
            f.write(d.tobytes())

        # Helper to write Int8 tensor (Q8_0 style symmetric quantization)
        def write_tensor_q8(t):
            d = t.detach().cpu().to(torch.float32)
            # Find the absolute max for each row
            amax = d.abs().max(dim=1, keepdim=True).values
            scales = amax / 127.0
            scales[scales == 0] = 1.0  # Prevent division by zero
            q = torch.round(d / scales).to(torch.int8)

            # Write scales (FP32) followed by the quantized weights (INT8)
            f.write(scales.squeeze(1).numpy().tobytes())
            f.write(q.numpy().tobytes())

        # Write weights
        # We keep 1D tensors (embeddings and norms) as FP32, and quantize the large 2D matrix multiplications
        write_tensor_fp32(model.tok_embeddings.weight)
        for layer in model.layers:
            write_tensor_fp32(layer.attention_norm.weight)
            write_tensor_q8(layer.attention.wq.weight)
            write_tensor_q8(layer.attention.wk.weight)
            write_tensor_q8(layer.attention.wv.weight)
            write_tensor_q8(layer.attention.wo.weight)

            write_tensor_fp32(layer.ffn_norm.weight)
            write_tensor_q8(layer.feed_forward.w1.weight)
            write_tensor_q8(layer.feed_forward.w2.weight)
            write_tensor_q8(layer.feed_forward.w3.weight)

        write_tensor_fp32(model.norm.weight)
        write_tensor_q8(model.output.weight)

    print(
        f"Done! Saved {output_path}. You can now run `make runq` and execute the quantized C code."
    )


if __name__ == "__main__":
    export_model_q8()
