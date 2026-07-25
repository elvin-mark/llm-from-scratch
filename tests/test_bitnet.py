import torch

from tiny_llm import BitNetLLM, BitLinear, STETernaryQuantize


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
