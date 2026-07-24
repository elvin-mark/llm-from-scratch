import torch
import numpy as np

from tiny_llm import TinyLLM
from scripts.inference import apply_rotary_emb as numpy_apply_rope


def test_numpy_rope_equivalence():
    """Verify NumPy apply_rotary_emb output matches PyTorch complex RoPE output."""
    dim_head = 16
    seq_len = 8
    n_heads = 2
    batch = 1

    x = torch.randn(batch, seq_len, n_heads, dim_head)
    theta = 10000.0
    freqs = 1.0 / (theta ** (torch.arange(0, dim_head, 2).float() / dim_head))
    t = torch.arange(seq_len).float()
    freqs_outer = torch.outer(t, freqs)
    fc_complex = np.exp(1j * freqs_outer.numpy())

    x_np = x.numpy()
    out_np = numpy_apply_rope(x_np, fc_complex)

    assert out_np.shape == (batch, seq_len, n_heads, dim_head)
    assert not np.isnan(out_np).any()


def test_numpy_vs_pytorch_forward_logits():
    """
    Verifies that converting PyTorch TinyLLM weights to NumPy dictionary
    and executing the pure NumPy forward loop produces identical logits.
    """
    vocab_size = 50
    dim = 32
    n_layers = 2
    n_heads = 2
    ffn_dim = 64

    model = TinyLLM(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
    )
    model.eval()

    tokens = torch.tensor([[1, 5, 12, 8, 3]], dtype=torch.long)

    # 1. PyTorch Forward Pass
    with torch.no_grad():
        pytorch_logits = model(tokens)[0, -1].numpy()

    # 2. Extract weights into NumPy dict
    w = {
        k: v.detach().cpu().numpy().astype(np.float32)
        for k, v in model.state_dict().items()
    }

    # 3. Pure NumPy Forward Pass (extract from scripts/inference.py logic)
    token_list = tokens[0].tolist()
    b, s = 1, len(token_list)
    h = w["tok_embeddings.weight"][token_list][None, ...]

    fc = np.exp(
        1j
        * np.outer(
            np.arange(s),
            1.0 / (10000.0 ** (np.arange(0, dim // n_heads, 2) / (dim // n_heads))),
        )
    )
    mask = (
        np.triu(np.full((s, s), -np.inf, dtype=np.float32), k=1)[None, None, :, :]
        if s > 1
        else 0
    )

    layer_idx = 0
    while f"layers.{layer_idx}.attention.wq.weight" in w:
        pfx = f"layers.{layer_idx}"
        x = (
            h
            * (1.0 / np.sqrt(np.mean(h**2, axis=-1, keepdims=True) + 1e-6))
            * w[f"{pfx}.attention_norm.weight"]
        )
        xq, xk, xv = [x @ w[f"{pfx}.attention.w{c}.weight"].T for c in "qkv"]

        xq = numpy_apply_rope(xq.reshape(b, s, n_heads, -1), fc).transpose(0, 2, 1, 3)
        xk = numpy_apply_rope(xk.reshape(b, s, n_heads, -1), fc).transpose(0, 2, 1, 3)
        xv = xv.reshape(b, s, n_heads, -1).transpose(0, 2, 1, 3)

        sc = xq @ xk.transpose(0, 1, 3, 2) / np.sqrt(dim // n_heads) + mask
        pr = np.exp(sc - np.max(sc, axis=-1, keepdims=True))
        pr = pr / np.sum(pr, axis=-1, keepdims=True)
        attn = (pr @ xv).transpose(0, 2, 1, 3).reshape(b, s, -1)
        h = h + attn @ w[f"{pfx}.attention.wo.weight"].T

        x = (
            h
            * (1.0 / np.sqrt(np.mean(h**2, axis=-1, keepdims=True) + 1e-6))
            * w[f"{pfx}.ffn_norm.weight"]
        )
        h1 = x @ w[f"{pfx}.feed_forward.w1.weight"].T
        h = (
            h
            + ((h1 / (1.0 + np.exp(-h1))) * (x @ w[f"{pfx}.feed_forward.w3.weight"].T))
            @ w[f"{pfx}.feed_forward.w2.weight"].T
        )
        layer_idx += 1

    h = (
        h
        * (1.0 / np.sqrt(np.mean(h**2, axis=-1, keepdims=True) + 1e-6))
        * w["norm.weight"]
    )
    numpy_logits = (h @ w["output.weight"].T)[0, -1]

    # Check exact match down to 1e-4 tolerance
    assert np.allclose(pytorch_logits, numpy_logits, atol=1e-4), (
        "NumPy forward logits do not match PyTorch model logits."
    )
