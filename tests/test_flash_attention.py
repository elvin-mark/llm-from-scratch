import torch
from tiny_llm import Attention, EducationalFlashAttention, precompute_freqs_cis


def test_flash_attention_exact_match():
    """
    Verifies that EducationalFlashAttention matches standard Attention output
    down to 1e-5 precision across batch, seqlen, and head dimensions.
    """
    dim = 64
    n_heads = 4
    seqlen = 32
    batch = 2
    block_size = 8

    standard_attn = Attention(dim=dim, n_heads=n_heads)
    flash_attn = EducationalFlashAttention(
        dim=dim, n_heads=n_heads, block_size=block_size
    )

    # Copy identical weights from standard to flash attention
    flash_attn.wq.weight.data.copy_(standard_attn.wq.weight.data)
    flash_attn.wk.weight.data.copy_(standard_attn.wk.weight.data)
    flash_attn.wv.weight.data.copy_(standard_attn.wv.weight.data)
    flash_attn.wo.weight.data.copy_(standard_attn.wo.weight.data)

    standard_attn.eval()
    flash_attn.eval()

    torch.manual_seed(42)
    x = torch.randn(batch, seqlen, dim)
    freqs_cis = precompute_freqs_cis(dim // n_heads, seqlen)

    # Test without mask
    with torch.no_grad():
        out_standard = standard_attn(x, freqs_cis)
        out_flash = flash_attn(x, freqs_cis)

    assert torch.allclose(out_standard, out_flash, atol=1e-5), (
        "FlashAttention output does not match standard Attention (no mask)"
    )


def test_flash_attention_causal_mask_match():
    """
    Verifies that EducationalFlashAttention matches standard Attention under Causal Masking.
    """
    dim = 64
    n_heads = 4
    seqlen = 24
    batch = 2
    block_size = 8

    standard_attn = Attention(dim=dim, n_heads=n_heads)
    flash_attn = EducationalFlashAttention(
        dim=dim, n_heads=n_heads, block_size=block_size
    )

    # Copy weights
    flash_attn.wq.weight.data.copy_(standard_attn.wq.weight.data)
    flash_attn.wk.weight.data.copy_(standard_attn.wk.weight.data)
    flash_attn.wv.weight.data.copy_(standard_attn.wv.weight.data)
    flash_attn.wo.weight.data.copy_(standard_attn.wo.weight.data)

    standard_attn.eval()
    flash_attn.eval()

    x = torch.randn(batch, seqlen, dim)
    freqs_cis = precompute_freqs_cis(dim // n_heads, seqlen)
    mask = torch.full((1, 1, seqlen, seqlen), float("-inf"))
    mask = torch.triu(mask, diagonal=1)

    with torch.no_grad():
        out_standard = standard_attn(x, freqs_cis, mask)
        out_flash = flash_attn(x, freqs_cis, mask)

    assert torch.allclose(out_standard, out_flash, atol=1e-5), (
        "FlashAttention output does not match standard Attention (with causal mask)"
    )
