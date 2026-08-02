import json

import torch
from click.testing import CliRunner

from tiny_llm.cli.token_entropy import token_entropy_cmd
from tiny_llm.models import create_model
from tiny_llm.tokenizer import ScratchTokenizer


def test_token_entropy_cli(tmp_path):
    """Verify tiny-llm token-entropy CLI command execution."""
    prompt = "Once upon a time in a tiny land"
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
        token_entropy_cmd,
        [
            "--checkpoint",
            str(ckpt_path),
            "--prompt",
            prompt,
            "--tokenizer-path",
            str(tok_path),
            "--use-scratch-tokenizer",
        ],
    )

    assert result.exit_code == 0
    assert "Token Entropy & Surprisal Heatmap Visualizer" in result.output
    assert "Detailed Token Information-Theoretic Breakdown" in result.output
