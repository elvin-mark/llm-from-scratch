import struct
import tempfile
import torch

from tiny_llm import RMSNorm
from tools.export.export_c import export_model


def test_rmsnorm_scale_invariance():
    """
    RMSNorm property: RMSNorm(c * x) == RMSNorm(x) for positive scalar c.
    """
    dim = 64
    norm = RMSNorm(dim)
    norm.eval()

    x = torch.randn(2, 10, dim)
    scale = 3.5

    with torch.no_grad():
        out1 = norm(x)
        out2 = norm(x * scale)

    assert torch.allclose(out1, out2, atol=1e-5), "RMSNorm must be scale invariant."


def test_export_c_binary_header_format():
    """Verify that export_c writes the 256-byte header with correct struct packing."""
    dim = 128
    n_layers = 4
    n_heads = 4
    ffn_dim = 512
    max_seq_len = 64

    # Create dummy model checkpoint and tokenizer file
    with (
        tempfile.NamedTemporaryFile("wb", suffix=".pth", delete=False) as model_f,
        tempfile.NamedTemporaryFile(
            "w+", suffix=".json", encoding="utf-8", delete=False
        ) as tok_f,
        tempfile.NamedTemporaryFile("wb", suffix=".bin", delete=False) as out_model_f,
        tempfile.NamedTemporaryFile("wb", suffix=".bin", delete=False) as out_vocab_f,
    ):
        from tiny_llm import TinyLLM, ScratchTokenizer

        tokenizer_data = ScratchTokenizer.train(
            "dummy text for export test", vocab_size=50
        )
        actual_vocab_size = len(tokenizer_data["model"]["vocab"])

        model = TinyLLM(
            vocab_size=actual_vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            max_seq_len=max_seq_len,
        )
        torch.save(model.state_dict(), model_f.name)
        import json

        json.dump(tokenizer_data, tok_f, ensure_ascii=False)
        tok_f.flush()

        # Run export_model
        export_model(
            model_path=model_f.name,
            tokenizer_path=tok_f.name,
            output_path=out_model_f.name,
            vocab_path=out_vocab_f.name,
        )

        # Read 256-byte header from exported binary
        with open(out_model_f.name, "rb") as f:
            header_bytes = f.read(256)

        assert len(header_bytes) == 256
        header_ints = struct.unpack("iiiiiii", header_bytes[:28])

        # Verify struct fields: dim, ffn_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len
        assert header_ints[0] == dim
        assert header_ints[1] == ffn_dim
        assert header_ints[2] == n_layers
        assert header_ints[3] == n_heads
        assert header_ints[4] == n_heads  # n_kv_heads == n_heads for dense model
        assert header_ints[5] == len(tokenizer_data["model"]["vocab"])
        assert header_ints[6] == max_seq_len
