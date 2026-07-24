import torch
import math
from tiny_llm.model import TinyLLM, Attention, apply_rotary_emb, precompute_freqs_cis


def test_rope_shapes():
    """Verify that applying rotary embeddings does not change shape."""
    batch, seqlen, n_heads, head_dim = 2, 16, 4, 32
    xq = torch.randn(batch, seqlen, n_heads, head_dim)
    xk = torch.randn(batch, seqlen, n_heads, head_dim)

    freqs_cis = precompute_freqs_cis(head_dim, seqlen)

    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

    assert xq_out.shape == xq.shape
    assert xk_out.shape == xk.shape


def test_rope_relative_property():
    """Verify that RoPE dot product depends only on relative positions."""
    head_dim = 16
    seqlen = 10

    # Precompute frequencies
    freqs_cis = precompute_freqs_cis(head_dim, seqlen)

    # Create single query/key vectors at different positions
    q = torch.randn(1, 1, 1, head_dim)  # batch=1, seqlen=1, heads=1
    k = torch.randn(1, 1, 1, head_dim)

    # Rotate q at position 2 and k at position 5 (diff = 3)
    q_pos2, _ = apply_rotary_emb(q, q, freqs_cis[2:3])
    _, k_pos5 = apply_rotary_emb(k, k, freqs_cis[5:6])
    dot_product_2_5 = torch.sum(q_pos2 * k_pos5).item()

    # Rotate q at position 4 and k at position 7 (diff = 3)
    q_pos4, _ = apply_rotary_emb(q, q, freqs_cis[4:5])
    _, k_pos7 = apply_rotary_emb(k, k, freqs_cis[7:8])
    dot_product_4_7 = torch.sum(q_pos4 * k_pos7).item()

    # Dot products should be equal because the relative distance is identical (3 positions apart)
    assert math.isclose(dot_product_2_5, dot_product_4_7, rel_tol=1e-4)


def test_attention_causality():
    """
    Verify that the causal mask prevents future tokens from affecting past token representations.
    If we change sequence tokens at position > t, the attention output at position <= t must remain unchanged.
    """
    dim = 64
    n_heads = 2
    seq_len = 8

    attn = Attention(dim=dim, n_heads=n_heads)
    attn.eval()

    # Create two sequences A and B that are identical up to index 4 (0, 1, 2, 3, 4)
    # but differ at index 5, 6, 7
    seq_A = torch.randn(1, seq_len, dim)
    seq_B = seq_A.clone()
    seq_B[0, 5:] = torch.randn(1, 3, dim)  # mutate future tokens

    # Causal Mask (upper triangular matrix initialized to -inf above diagonal)
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)

    freqs_cis = precompute_freqs_cis(dim // n_heads, seq_len)

    with torch.no_grad():
        out_A = attn(seq_A, freqs_cis, mask)
        out_B = attn(seq_B, freqs_cis, mask)

    # Outputs at index 0, 1, 2, 3, 4 should be mathematically identical
    # we allow minor float precision limits
    assert torch.allclose(out_A[0, :5, :], out_B[0, :5, :], atol=1e-5)

    # Outputs at index 5, 6, 7 should differ
    assert not torch.allclose(out_A[0, 5:, :], out_B[0, 5:, :], atol=1e-5)


def test_model_output_shape():
    """Verify that input tokens correctly flow through the TinyLLM model and output expected logits shape."""
    vocab_size = 100
    dim = 32
    n_layers = 2
    n_heads = 2
    ffn_dim = 64
    max_seq_len = 16

    model = TinyLLM(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
    )
    model.eval()

    batch_size = 3
    seq_len = 10
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (batch_size, seq_len, vocab_size)
