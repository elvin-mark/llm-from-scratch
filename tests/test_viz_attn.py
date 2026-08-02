import torch

from tiny_llm.models import create_model


def test_return_attn_weights_dense():
    """Verify return_attn_weights returns list of 4D attention matrices for TinyLLM."""
    model, config = create_model("dense", vocab_size=100, dim=64, n_layers=2, n_heads=2)
    tokens = torch.randint(0, 100, (1, 6))

    logits, weights = model(tokens, return_attn_weights=True)

    assert logits.shape == (1, 6, 100)
    assert len(weights) == 2  # 2 layers
    assert weights[0].shape == (1, 2, 6, 6)  # [bsz, n_heads, seqlen, seqlen]
    assert weights[1].shape == (1, 2, 6, 6)


def test_return_attn_weights_moe():
    """Verify return_attn_weights returns list of 4D attention matrices for MoELLM (GQA)."""
    model, config = create_model("moe", vocab_size=100, dim=64, n_layers=2, n_heads=4, n_kv_heads=2)
    tokens = torch.randint(0, 100, (1, 6))

    logits, weights = model(tokens, return_attn_weights=True)

    assert logits.shape == (1, 6, 100)
    assert len(weights) == 2
    assert weights[0].shape == (1, 4, 6, 6)  # [bsz, n_heads, seqlen, seqlen]


def test_return_attn_weights_nano():
    """Verify return_attn_weights returns list of 4D attention matrices for NanoLLM."""
    model, config = create_model("nano", vocab_size=100, dim=64, n_layers=2, n_heads=2)
    tokens = torch.randint(0, 100, (1, 6))

    logits, weights = model(tokens, return_attn_weights=True)

    assert logits.shape == (1, 6, 100)
    assert len(weights) == 2
    assert weights[0].shape == (1, 2, 6, 6)
