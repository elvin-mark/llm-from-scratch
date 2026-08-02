import json

import torch
from click.testing import CliRunner

from tiny_llm.cli.eval import eval_cmd
from tiny_llm.eval import evaluate_perplexity
from tiny_llm.models import create_model
from tiny_llm.tokenizer import ScratchTokenizer


def test_evaluate_perplexity_math():
    """Verify evaluate_perplexity function computes valid perplexity and accuracy."""
    model, config = create_model("dense", vocab_size=100, dim=64, n_layers=2, n_heads=2)

    from torch.utils.data import DataLoader, TensorDataset

    x = torch.randint(0, 100, (4, 16))
    y = torch.randint(0, 100, (4, 16))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    metrics = evaluate_perplexity(model, loader)

    assert metrics["perplexity"] >= 1.0
    assert 0.0 <= metrics["token_accuracy"] <= 100.0
    assert metrics["total_tokens"] == 64


def test_eval_cmd_cli(tmp_path):
    """Verify tiny-llm eval CLI command invocation in isolated environment."""
    corpus_text = "The cat sat on the mat.\nHello world!\n"
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(corpus_text, encoding="utf-8")

    tok_data = ScratchTokenizer.train(corpus_text, vocab_size=50)
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
        eval_cmd,
        [
            "--checkpoint",
            str(ckpt_path),
            "--dataset",
            str(corpus_path),
            "--tokenizer-path",
            str(tok_path),
            "--use-scratch-tokenizer",
        ],
    )

    assert result.exit_code == 0
    assert "Cross-Entropy Loss" in result.output
    assert "Perplexity (PPL)" in result.output
