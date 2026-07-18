import struct
import torch
import json
import os
import sys

# Add parent directory to path to import TinyLLM
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers import Tokenizer
from model import TinyLLM

def export_model_q8():
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

    print("Exporting model to model_q8.bin...")
    with open("model_q8.bin", "wb") as f:
        # Write header (256 bytes)
        # struct format: 8 ints (dim, ffn_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len, is_quantized)
        header = struct.pack("iiiiiiii", dim, ffn_dim, n_layers, n_heads, n_heads, vocab_size, max_seq_len, 1)
        # Pad with zeros to 256 bytes
        header += b'\x00' * (256 - len(header))
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
            scales[scales == 0] = 1.0 # Prevent division by zero
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
    
    print("Done! You can now run `make runq` and execute the quantized C code.")

if __name__ == "__main__":
    export_model_q8()
