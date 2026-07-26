import os
import tempfile

import torch

from tiny_llm import ScratchTokenizer, TinyLLM
from tools.export.export_c import export_model
from tools.export.export_q8 import export_model_q8


def run_quantization_benchmark():
    print("=" * 80)
    print("📊 BENCHMARK 5: FP32 vs. Int8 Dynamic Quantization Model Size & Footprint")
    print("=" * 80)

    vocab_size = 4000
    dim = 128
    n_layers = 4
    n_heads = 4
    ffn_dim = 512
    max_seq_len = 64

    with (
        tempfile.NamedTemporaryFile("wb", suffix=".pth", delete=False) as model_f,
        tempfile.NamedTemporaryFile("w+", suffix=".json", encoding="utf-8", delete=False) as tok_f,
        tempfile.NamedTemporaryFile("wb", suffix=".bin", delete=False) as fp32_bin_f,
        tempfile.NamedTemporaryFile("wb", suffix=".bin", delete=False) as q8_bin_f,
        tempfile.NamedTemporaryFile("wb", suffix=".bin", delete=False) as vocab_bin_f,
    ):
        tokenizer_data = ScratchTokenizer.train(
            "dummy corpus for quantization test with extra text to generate tokens",
            vocab_size=vocab_size,
        )
        actual_vocab_size = len(tokenizer_data["model"]["vocab"])

        model = TinyLLM(
            vocab_size=actual_vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            max_seq_len=max_seq_len,
        )
        torch.save(model.state_dict(), model_f.name)

        import json

        json.dump(tokenizer_data, tok_f, ensure_ascii=False)
        tok_f.flush()

        # Export FP32 binary
        export_model(
            model_path=model_f.name,
            tokenizer_path=tok_f.name,
            output_path=fp32_bin_f.name,
            vocab_path=vocab_bin_f.name,
        )

        # Export Int8 binary
        export_model_q8(
            model_path=model_f.name,
            tokenizer_path=tok_f.name,
            output_path=q8_bin_f.name,
        )

        fp32_size_mb = os.path.getsize(fp32_bin_f.name) / (1024.0 * 1024.0)
        q8_size_mb = os.path.getsize(q8_bin_f.name) / (1024.0 * 1024.0)

    compression_ratio = (1.0 - (q8_size_mb / fp32_size_mb)) * 100.0

    print(
        f"  Precision / Format  | {'Binary File Size (MB)':<22} | {'Relative Footprint':<20} | {'Space Savings':<14}"
    )
    print("-" * 80)
    print(f"  FP32 (Standard C)   | {fp32_size_mb:<22.2f} | 100.0%               | Baseline")
    print(
        f"  Int8 (Quantized C)  | {q8_size_mb:<22.2f} | {(q8_size_mb / fp32_size_mb) * 100:<19.2f}% | {compression_ratio:.1f}% Smaller"
    )
    print("-" * 80)
    print(
        "💡 Key Takeaway: Int8 dynamic quantization compresses 2D matrix weights from 4 bytes to 1 byte,"
    )
    print(f"   reducing overall binary footprint by {compression_ratio:.1f}%.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_quantization_benchmark()
