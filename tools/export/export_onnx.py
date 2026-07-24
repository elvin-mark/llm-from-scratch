import torch
import os
import argparse
from tiny_llm.model import TinyLLM
from tokenizers import Tokenizer


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

    print(f"Loading model with vocab_size={vocab_size}...")
    # These match the default hyperparams used during training
    model = TinyLLM(
        vocab_size=vocab_size,
        dim=128,
        n_layers=4,
        n_heads=4,
        ffn_dim=512,
        max_seq_len=64,
    )

    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
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
                from onnxruntime.quantization import quantize_dynamic, QuantType

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
