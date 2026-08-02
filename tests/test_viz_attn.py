import json

import torch
from click.testing import CliRunner

from tiny_llm.cli.viz_attn import viz_attn_cmd
from tiny_llm.models import create_model
from tiny_llm.tokenizer import ScratchTokenizer


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


def test_viz_attn_cmd_cli(tmp_path):
    """Verify tiny-llm viz-attn CLI command invocation in isolated environment."""
    prompt = "The cat sat on the mat"
    tok_data = ScratchTokenizer.train(prompt, vocab_size=50)
    tok_path = tmp_path / "tokenizer.json"
    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump(tok_data, f, ensure_ascii=False, indent=2)

    vocab_size = len(tok_data["model"]["vocab"])
    model, config = create_model("dense", vocab_size=vocab_size, dim=32, n_layers=2, n_heads=2)
    ckpt_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), ckpt_path)
    config.save_json(tmp_path / "config.json")

    runner = CliRunner()
    result = runner.invoke(
        viz_attn_cmd,
        [
            "--checkpoint",
            str(ckpt_path),
            "--prompt",
            prompt,
            "--tokenizer-path",
            str(tok_path),
            "--use-scratch-tokenizer",
            "--layer",
            "0",
            "--head",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert "Attention Heatmap Visualizer" in result.output
