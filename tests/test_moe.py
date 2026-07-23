import torch
import pytest
from tiny_llm import GroupedQueryAttention, MoERouter, MoEFeedForward, MoELLM, MoELLMConfig, precompute_freqs_cis


def test_gqa_shapes():
    """Verify Grouped Query Attention output shape with key/value repetition."""
    dim = 64
    n_heads = 4
    n_kv_heads = 2
    seqlen = 16
    batch = 2

    gqa = GroupedQueryAttention(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads)
    gqa.eval()

    x = torch.randn(batch, seqlen, dim)
    freqs_cis = precompute_freqs_cis(dim // n_heads, seqlen)

    with torch.no_grad():
        out = gqa(x, freqs_cis)

    assert out.shape == (batch, seqlen, dim)


def test_moe_router_routing():
    """Verify Top-K Router outputs valid softmax weights and expert indices."""
    dim = 64
    num_experts = 8
    top_k = 2
    batch, seqlen = 2, 10

    router = MoERouter(dim=dim, num_experts=num_experts, top_k=top_k)
    x = torch.randn(batch, seqlen, dim)

    weights, selected = router(x)

    assert weights.shape == (batch, seqlen, top_k)
    assert selected.shape == (batch, seqlen, top_k)
    # Check that weights sum to 1.0 per token across top-k
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch, seqlen), atol=1e-5)


def test_moe_feedforward():
    """Verify Mixture-of-Experts SwiGLU MLP output shape."""
    dim = 64
    hidden_dim = 128
    num_experts = 4
    top_k = 2
    batch, seqlen = 2, 16

    moe_ffn = MoEFeedForward(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k)
    moe_ffn.eval()

    x = torch.randn(batch, seqlen, dim)
    with torch.no_grad():
        out = moe_ffn(x)

    assert out.shape == (batch, seqlen, dim)


def test_moe_llm_forward():
    """Verify complete MoELLM model forward pass (GQA + MoE)."""
    config = MoELLMConfig(
        vocab_size=100,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=128,
        num_experts=4,
        num_experts_per_tok=2,
        max_seq_len=32,
    )

    model = MoELLM(config=config)
    model.eval()

    batch_size = 2
    seq_len = 10
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
