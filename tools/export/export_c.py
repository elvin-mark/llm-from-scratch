import struct
import torch
import os
import sys

# Add parent directory to path to import TinyLLM
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers import Tokenizer
from tiny_llm.model import TinyLLM


def export_model(
    model_path=None, tokenizer_path=None, output_path=None, vocab_path=None
):
    if output_path is None:
        output_path = (
            "model.bin" if os.path.basename(os.getcwd()) == "c" else "c/model.bin"
        )
    if vocab_path is None:
        vocab_path = (
            "vocab.bin" if os.path.basename(os.getcwd()) == "c" else "c/vocab.bin"
        )

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

    # Model configuration (from train.py)
    dim = 128
    n_layers = 4
    n_heads = 4
    ffn_dim = 512
    max_seq_len = 64

    model = TinyLLM(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.dirname(vocab_path):
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)

    print(f"Exporting model to {output_path}...")
    with open(output_path, "wb") as f:
        # Write header (256 bytes to leave room for future expansion)
        # struct format: 7 ints (dim, ffn_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len)
        header = struct.pack(
            "iiiiiii", dim, ffn_dim, n_layers, n_heads, n_heads, vocab_size, max_seq_len
        )
        # Pad with zeros to 256 bytes
        header += b"\x00" * (256 - len(header))
        f.write(header)

        # Helper to write tensor
        def write_tensor(t):
            d = t.detach().cpu().to(torch.float32).numpy()
            f.write(d.tobytes())

        # Write weights
        write_tensor(model.tok_embeddings.weight)
        for layer in model.layers:
            write_tensor(layer.attention_norm.weight)
            write_tensor(layer.attention.wq.weight)
            write_tensor(layer.attention.wk.weight)
            write_tensor(layer.attention.wv.weight)
            write_tensor(layer.attention.wo.weight)
            write_tensor(layer.ffn_norm.weight)
            write_tensor(layer.feed_forward.w1.weight)
            write_tensor(layer.feed_forward.w2.weight)
            write_tensor(layer.feed_forward.w3.weight)
        write_tensor(model.norm.weight)
        write_tensor(model.output.weight)

    print(f"Exporting tokenizer to {vocab_path}...")
    vocab = tokenizer.get_vocab()
    # Invert vocab map
    inv_vocab = {v: k for k, v in vocab.items()}
    with open(vocab_path, "wb") as f:
        f.write(struct.pack("i", vocab_size))
        for i in range(vocab_size):
            token_str = inv_vocab.get(i, "").encode("utf-8")
            f.write(struct.pack("i", len(token_str)))
            f.write(token_str)

    print(
        f"Done! Saved {output_path} and {vocab_path}. You can now execute the C code."
    )


if __name__ == "__main__":
    export_model()
