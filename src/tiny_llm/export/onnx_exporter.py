import json
import os

import torch
from tokenizers import Tokenizer

from tiny_llm.models.factory import create_model


def export_onnx(
    model_path=None,
    tokenizer_path=None,
    output_path="tiny_llm.onnx",
    quantize=False,
    arch=None,
):
    """Export model to ONNX computational graph format for WebGL / WebAssembly execution for any architecture."""
    if model_path is None:
        for p in [
            "checkpoints/tiny_llm.pth",
            "../checkpoints/tiny_llm.pth",
            "tiny_llm.pth",
            "../tiny_llm.pth",
            "checkpoints/nano_llm.pth",
            "checkpoints/moe_llm.pth",
            "checkpoints/bitnet_model.pth",
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

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    ckpt_dir = os.path.dirname(model_path) if model_path else "."
    base_name = os.path.splitext(model_path)[0] if model_path else "model"
    config_path = base_name + ".json"
    if not os.path.exists(config_path):
        config_path = os.path.join(ckpt_dir, "config.json")

    state_dict = None
    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    else:
        print(f"  ⚠️ Warning: Checkpoint '{model_path}' not found. Using initialized model weights.")

    cfg = {}
    if os.path.exists(config_path) and state_dict is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded_cfg = json.load(f)
        if (
            "dim" not in loaded_cfg
            or loaded_cfg["dim"] == state_dict["tok_embeddings.weight"].shape[1]
        ):
            cfg = loaded_cfg

    # Auto-detect architecture
    if arch is None:
        arch = cfg.get("arch", None)
    if arch is None:
        if model_path and "nano" in model_path.lower():
            arch = "nano"
        elif model_path and (
            "moe" in model_path.lower()
            or (state_dict and any("experts" in k for k in state_dict.keys()))
        ):
            arch = "moe"
        elif model_path and (
            "bitnet" in model_path.lower()
            or (state_dict and any("weight_scale" in k or "gamma" in k for k in state_dict.keys()))
        ):
            arch = "bitnet"
        else:
            arch = "dense"

    print(f"  Architecture auto-detected: {arch.upper()}")

    dim = (
        cfg.get("dim", state_dict["tok_embeddings.weight"].shape[1])
        if state_dict
        else cfg.get("dim", 128)
    )
    vocab_size = (
        cfg.get("vocab_size", state_dict["tok_embeddings.weight"].shape[0])
        if state_dict
        else cfg.get("vocab_size", vocab_size)
    )
    n_layers = (
        cfg.get(
            "n_layers",
            len([k for k in state_dict.keys() if k.endswith(".attention_norm.weight")]),
        )
        if state_dict
        else cfg.get("n_layers", 4)
    )
    n_heads = cfg.get("n_heads", 4)
    n_kv_heads = cfg.get("n_kv_heads", n_heads)

    if state_dict and "layers.0.feed_forward.w1.weight" in state_dict:
        ffn_dim = cfg.get("ffn_dim", state_dict["layers.0.feed_forward.w1.weight"].shape[0])
    else:
        ffn_dim = cfg.get("ffn_dim", 512)

    max_seq_len = cfg.get("max_seq_len", 64)

    model, _ = create_model(
        arch=arch,
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
    )
    if state_dict:
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

    print(f"Done! Model ({arch.upper()}) exported to {output_path}")
