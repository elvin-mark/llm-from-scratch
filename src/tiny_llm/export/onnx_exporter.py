import json
import os

import torch
from tokenizers import Tokenizer

from tiny_llm.models import TinyLLM


def export_onnx(model_path=None, tokenizer_path=None, output_path="tiny_llm.onnx", quantize=False):
    """Export model to ONNX computational graph format for WebGL / WebAssembly execution."""
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

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    ckpt_dir = os.path.dirname(model_path)
    base_name = os.path.splitext(model_path)[0]
    config_path = base_name + ".json"
    if not os.path.exists(config_path):
        config_path = os.path.join(ckpt_dir, "config.json")

    state_dict = torch.load(model_path, map_location="cpu")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        dim = cfg.get("dim", state_dict["tok_embeddings.weight"].shape[1])
        n_layers = cfg.get(
            "n_layers",
            len([k for k in state_dict.keys() if k.endswith(".attention_norm.weight")]),
        )
        n_heads = cfg.get("n_heads", 4)
        ffn_dim = cfg.get("ffn_dim", state_dict["layers.0.feed_forward.w1.weight"].shape[0])
        max_seq_len = cfg.get("max_seq_len", 64)
    else:
        dim = state_dict["tok_embeddings.weight"].shape[1]
        n_layers = len([k for k in state_dict.keys() if k.endswith(".attention_norm.weight")])
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

    dummy_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
    )

    if quantize:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quant_output = output_path.replace(".onnx", "_quant.onnx")
            quantize_dynamic(output_path, quant_output, weight_type=QuantType.QUInt8)
            output_path = quant_output
        except Exception as e:
            print(f"ONNX Quantization Warning: {e}")

    print(f"Done! Model exported to {output_path}")
