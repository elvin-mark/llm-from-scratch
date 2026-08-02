import argparse
import os

import torch
from tokenizers import Tokenizer

from tiny_llm.model import TinyLLM


def export_to_onnx(
    model_path=None, tokenizer_path=None, output_path="tiny_llm.onnx", quantize=False
):
    if model_path is None:
        model_path = (
            "checkpoints/tiny_llm.pth"
            if os.path.exists("checkpoints/tiny_llm.pth")
            else "tiny_llm.pth"
        )
    if tokenizer_path is None:
        tokenizer_path = (
            "checkpoints/tokenizer.json"
            if os.path.exists("checkpoints/tokenizer.json")
            else "tokenizer.json"
        )
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist.")
        return
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer path '{tokenizer_path}' does not exist.")
        return

    print("Loading tokenizer...")
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

    print(f"Loading model with vocab_size={vocab_size}, dim={dim}, n_layers={n_layers}...")
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

    # Create dummy input: Batch size 1, Sequence Length 4
    # The actual values don't matter much; ONNX just needs them to trace the execution graph.
    dummy_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,  # Opset 14 is highly recommended for modern NLP ops
        do_constant_folding=True,  # Let ONNX optimize static branches
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={  # Crucial for autoregressive generation
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
    )
    print("✅ Export successful!")

    # Optional Quantization
    if quantize:
        import importlib.util

        quantized_path = output_path.replace(".onnx", "_quantized.onnx")
        if importlib.util.find_spec("onnx") is not None:
            try:
                from onnxruntime.quantization import QuantType, quantize_dynamic

                print(f"Quantizing model to {quantized_path} (Int8)...")
                quantize_dynamic(
                    output_path,
                    quantized_path,
                    weight_type=QuantType.QUInt8,
                )
                print("✅ Quantization successful!")
            except Exception as e:
                print(f"\n⚠️ Note: Could not quantize the model: {e}")
        else:
            print(
                "\n⚠️ Note: Could not quantize the model because 'onnx' and 'onnxruntime' are not installed."
            )
            print(
                "Run `pip install onnx onnxruntime` and run this script again with --quantize to create an Int8 version for the web."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export TinyLLM to ONNX format for Web deployment."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to input PyTorch model (.pth)",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to input tokenizer (.json) file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tiny_llm.onnx",
        help="Output path for the ONNX file",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Also generate an 8-bit quantized version of the ONNX model",
    )

    args = parser.parse_args()

    export_to_onnx(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        output_path=args.output,
        quantize=args.quantize,
    )
