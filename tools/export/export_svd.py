import argparse

from tiny_llm.export.svd_exporter import export_svd


def export_model(
    checkpoint_path=None,
    output_path=None,
    tokenizer_path="checkpoints/tokenizer.json",
    rank=32,
    arch=None,
    max_seq_len=512,
):
    return export_svd(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        tokenizer_path=tokenizer_path,
        rank=rank,
        arch=arch,
        max_seq_len=max_seq_len,
    )


export_svd_model = export_svd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export PyTorch model to SVD+Int8 Quantized Binary format."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/tiny_llm.pth",
        help="Input PyTorch checkpoint path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="c/model_svd.bin",
        help="Output SVD model binary path",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="checkpoints/tokenizer.json",
        help="Path to tokenizer.json",
    )
    parser.add_argument("--rank", type=int, default=32, help="Truncated SVD rank")
    parser.add_argument(
        "--arch",
        type=str,
        choices=["dense", "nano", "moe", "bitnet"],
        default=None,
        help="Model architecture (auto-detected if omitted)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Max sequence length for RoPE KV-cache buffers",
    )

    args = parser.parse_args()
    export_svd(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        tokenizer_path=args.tokenizer,
        rank=args.rank,
        arch=args.arch,
        max_seq_len=args.max_seq_len,
    )
