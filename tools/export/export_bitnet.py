import struct
import argparse
import torch

from tiny_llm import BitNetLLM, ScratchTokenizer, STETernaryQuantize


def export_bitnet(model_path: str, tokenizer_path: str, output_path: str):
    print("📦 Exporting 1.58-Bit BitNet Model...")
    print(f"  Model Input:     {model_path}")
    print(f"  Tokenizer Input: {tokenizer_path}")
    print(f"  Binary Output:   {output_path}")

    # Load tokenizer
    tokenizer = ScratchTokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    # Load PyTorch checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    dim = checkpoint["tok_embeddings.weight"].shape[1]
    n_layers = len(
        [k for k in checkpoint.keys() if k.endswith(".attention_norm.weight")]
    )
    n_heads = 4
    ffn_dim = checkpoint["layers.0.feed_forward.w1.weight"].shape[0]
    max_seq_len = 128

    model = BitNetLLM(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
    )
    model.load_state_dict(checkpoint)
    model.eval()

    with open(output_path, "wb") as f:
        # Write 256-byte header struct
        # Format: dim, ffn_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len
        header = struct.pack(
            "iiiiiii",
            dim,
            ffn_dim,
            n_layers,
            n_heads,
            n_heads,
            vocab_size,
            max_seq_len,
        )
        header = header + b"\x00" * (256 - len(header))
        f.write(header)

        # 1. Write Token Embeddings (FP32)
        emb_weights = model.tok_embeddings.weight.detach().numpy().astype("float32")
        f.write(emb_weights.tobytes())

        # 2. Write Transformer Layers
        for i in range(n_layers):
            layer = model.layers[i]

            # Attention Norm (FP32)
            f.write(
                layer.attention_norm.weight.detach().numpy().astype("float32").tobytes()
            )

            # Ternarize Attention Weights to int8 {-1, 0, +1}
            for proj in [
                layer.attention.wq,
                layer.attention.wk,
                layer.attention.wv,
                layer.attention.wo,
            ]:
                w_fp32 = proj.weight.detach()
                w_ternary = STETernaryQuantize.apply(w_fp32).numpy()
                gamma = float(w_fp32.abs().mean().clamp(min=1e-5))
                w_int8 = (w_ternary / gamma).round().clip(-1, 1).astype("int8")
                f.write(w_int8.tobytes())

            # FFN Norm (FP32)
            f.write(layer.ffn_norm.weight.detach().numpy().astype("float32").tobytes())

            # Ternarize FFN Weights to int8 {-1, 0, +1}
            for proj in [
                layer.feed_forward.w1,
                layer.feed_forward.w2,
                layer.feed_forward.w3,
            ]:
                w_fp32 = proj.weight.detach()
                w_ternary = STETernaryQuantize.apply(w_fp32).numpy()
                gamma = float(w_fp32.abs().mean().clamp(min=1e-5))
                w_int8 = (w_ternary / gamma).round().clip(-1, 1).astype("int8")
                f.write(w_int8.tobytes())

        # 3. Final Norm (FP32) & Output Head (FP32)
        f.write(model.norm.weight.detach().numpy().astype("float32").tobytes())
        f.write(model.output.weight.detach().numpy().astype("float32").tobytes())

    # Export Tokenizer Vocabulary to vocab.bin
    vocab_path = output_path.replace("bitnet_model.bin", "vocab.bin")
    if not vocab_path.endswith("vocab.bin"):
        vocab_path = "checkpoints/vocab.bin"

    print(f"Exporting tokenizer vocabulary to {vocab_path}...")
    vocab = tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    with open(vocab_path, "wb") as f:
        f.write(struct.pack("i", vocab_size))
        for i in range(vocab_size):
            token_str = inv_vocab.get(i, "").encode("utf-8")
            f.write(struct.pack("i", len(token_str)))
            f.write(token_str)

    print(f"✅ Successfully exported 1.58-bit binary to '{output_path}' and vocabulary to '{vocab_path}'!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export BitNetLLM model to ternary int8 binary."
    )
    parser.add_argument("--model-path", default="checkpoints/bitnet_llm.pth")
    parser.add_argument("--tokenizer-path", default="checkpoints/tokenizer.json")
    parser.add_argument("--output-path", default="checkpoints/bitnet_model.bin")
    args = parser.parse_args()

    export_bitnet(args.model_path, args.tokenizer_path, args.output_path)
