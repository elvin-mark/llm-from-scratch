import torch
from tiny_llm import NanoLLM, TinyLLM


def test_nano_llm_weight_tying():
    """Verify that NanoLLM output head and embedding weights share the exact same memory."""
    model = NanoLLM(vocab_size=1000, dim=64, n_layers=2, n_heads=2, ffn_dim=256)
    assert model.output.weight is model.tok_embeddings.weight


def test_nano_llm_parameter_savings():
    """Verify that NanoLLM saves parameter count compared to untied TinyLLM."""
    vocab_size = 4000
    dim = 128
    nano_model = NanoLLM(
        vocab_size=vocab_size, dim=dim, n_layers=4, n_heads=4, ffn_dim=512
    )
    tiny_model = TinyLLM(
        vocab_size=vocab_size, dim=dim, n_layers=4, n_heads=4, ffn_dim=512
    )

    nano_params = sum(p.numel() for p in nano_model.parameters())
    tiny_params = sum(p.numel() for p in tiny_model.parameters())

    expected_savings = vocab_size * dim
    assert tiny_params - nano_params == expected_savings


def test_nano_llm_forward():
    """Verify NanoLLM forward pass output shapes."""
    bsz, seqlen = 2, 8
    model = NanoLLM(vocab_size=500, dim=32, n_layers=2, n_heads=2, ffn_dim=128)
    model.eval()

    tokens = torch.randint(0, 500, (bsz, seqlen))
    with torch.no_grad():
        logits = model(tokens)

    assert logits.shape == (bsz, seqlen, 500)
