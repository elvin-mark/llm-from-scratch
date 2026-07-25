import torch
from tiny_llm import MultiHeadLatentAttention, precompute_freqs_cis


def test_mla_output_shape():
    """Verify that MultiHeadLatentAttention outputs expected tensor shapes."""
    bsz, seqlen, dim = 2, 8, 64
    n_heads = 4
    kv_lora_rank = 16
    rope_dim = 8

    mla = MultiHeadLatentAttention(
        dim=dim,
        n_heads=n_heads,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=32,
        rope_dim=rope_dim,
    )
    mla.eval()

    x = torch.randn(bsz, seqlen, dim)
    freqs_cis = precompute_freqs_cis(rope_dim, seqlen * 2)[:seqlen]

    with torch.no_grad():
        out = mla(x, freqs_cis)

    assert out.shape == (bsz, seqlen, dim)


def test_mla_gradient_flow():
    """Verify gradient propagation through low-rank latent projections in MLA."""
    bsz, seqlen, dim = 1, 4, 32
    mla = MultiHeadLatentAttention(
        dim=dim, n_heads=2, kv_lora_rank=8, q_lora_rank=16, rope_dim=8
    )

    x = torch.randn(bsz, seqlen, dim, requires_grad=True)
    freqs_cis = precompute_freqs_cis(8, 10)[:seqlen]

    out = mla(x, freqs_cis)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert mla.w_dkv.weight.grad is not None
    assert mla.w_uk.weight.grad is not None
    assert mla.w_uv.weight.grad is not None
