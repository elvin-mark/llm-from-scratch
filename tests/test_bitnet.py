import os

import torch

from tiny_llm import BitLinear, BitNetLLM, STETernaryQuantize


def test_ste_ternary_quantization_values():
    """Verify that STETernaryQuantize maps weights to scaled ternary set {-1, 0, +1} * gamma."""
    w = torch.tensor([[-2.5, -0.8, 0.1, 0.9, 3.2]], requires_grad=True)
    quantized = STETernaryQuantize.apply(w)

    gamma = w.abs().mean()
    scaled = (quantized / gamma).round()

    # Verify ternary set values
    unique_vals = set(scaled.detach().numpy().flatten().tolist())
    assert unique_vals.issubset({-1.0, 0.0, 1.0})


def test_bitlinear_forward_backward():
    """Verify that BitLinear computes forward pass and accumulates gradients on FP32 master weights."""
    bit_layer = BitLinear(32, 64)
    x = torch.randn(2, 10, 32, requires_grad=True)

    out = bit_layer(x)
    assert out.shape == (2, 10, 64)

    loss = out.sum()
    loss.backward()

    assert bit_layer.weight.grad is not None
    assert bit_layer.weight.grad.shape == bit_layer.weight.shape
    assert not torch.isnan(bit_layer.weight.grad).any()


def test_bitnet_llm_forward():
    """Verify full BitNetLLM forward pass."""
    model = BitNetLLM(vocab_size=100, dim=64, n_layers=2, n_heads=4, ffn_dim=128)
    model.eval()

    tokens = torch.randint(0, 100, (2, 12))
    with torch.no_grad():
        logits = model(tokens)

    assert logits.shape == (2, 12, 100)


def test_export_bitnet_binary():
    """Verify exporting BitNetLLM to ternary binary format."""
    import json
    import subprocess
    import tempfile

    from tiny_llm import ScratchTokenizer
    from tiny_llm.export import export_bitnet

    vocab_size = 50
    tokenizer_data = ScratchTokenizer.train(
        "dummy text for bitnet test corpus with extra sentences",
        vocab_size=vocab_size,
    )
    actual_vocab_size = len(tokenizer_data["model"]["vocab"])
    model = BitNetLLM(vocab_size=actual_vocab_size, dim=32, n_layers=2, n_heads=2, ffn_dim=64)

    with (
        tempfile.NamedTemporaryFile("wb", suffix=".pth", delete=False) as model_f,
        tempfile.NamedTemporaryFile("w+", suffix=".json", encoding="utf-8", delete=False) as tok_f,
        tempfile.NamedTemporaryFile("wb", suffix=".bin", delete=False) as out_bin_f,
    ):
        torch.save(model.state_dict(), model_f.name)
        json.dump(tokenizer_data, tok_f, ensure_ascii=False)
        tok_f.flush()

        export_bitnet(model_f.name, tok_f.name, out_bin_f.name)

        # Run compiled ./c/run_bitnet executable if present
        if os.path.exists("./c/run_bitnet"):
            res = subprocess.run(["./c/run_bitnet", out_bin_f.name], capture_output=True, text=True)
            assert res.returncode == 0
            assert "0 FP Multiplications" in res.stdout
