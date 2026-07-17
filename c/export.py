import struct
import torch
import json
import os
import sys

# Add parent directory to path to import TinyLLM
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers import Tokenizer
from model import TinyLLM

def export_model():
    print("Loading model and tokenizer...")
    tokenizer = Tokenizer.from_file("../tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()

    # Model configuration (from train.py)
    dim = 128
    n_layers = 4
    n_heads = 4
    ffn_dim = 512
    max_seq_len = 64

    model = TinyLLM(vocab_size=vocab_size, dim=dim, n_layers=n_layers, n_heads=n_heads, ffn_dim=ffn_dim, max_seq_len=max_seq_len)
    model.load_state_dict(torch.load("../tiny_llm.pth", map_location="cpu", weights_only=True))
    model.eval()

    print("Exporting model to model.bin...")
    with open("model.bin", "wb") as f:
        # Write header (256 bytes to leave room for future expansion)
        # struct format: 7 ints (dim, ffn_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len)
        header = struct.pack("iiiiiii", dim, ffn_dim, n_layers, n_heads, n_heads, vocab_size, max_seq_len)
        # Pad with zeros to 256 bytes
        header += b'\x00' * (256 - len(header))
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
    
    print("Exporting tokenizer to vocab.bin...")
    vocab = tokenizer.get_vocab()
    # Invert vocab map
    inv_vocab = {v: k for k, v in vocab.items()}
    with open("vocab.bin", "wb") as f:
        f.write(struct.pack("i", vocab_size))
        for i in range(vocab_size):
            token_str = inv_vocab.get(i, "").encode('utf-8')
            f.write(struct.pack("i", len(token_str)))
            f.write(token_str)

    print("Done! You can now run `make` and execute the C code.")

if __name__ == "__main__":
    export_model()
