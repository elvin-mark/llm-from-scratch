from click.testing import CliRunner

from tiny_llm.cli.eval import eval_cmd
from tiny_llm.eval import evaluate_perplexity
from tiny_llm.models import create_model


def test_evaluate_perplexity_math():
    """Verify evaluate_perplexity function computes valid perplexity and accuracy."""
    model, config = create_model("dense", vocab_size=100, dim=64, n_layers=2, n_heads=2)

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    x = torch.randint(0, 100, (4, 16))
    y = torch.randint(0, 100, (4, 16))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    metrics = evaluate_perplexity(model, loader)

    assert metrics["perplexity"] >= 1.0
    assert 0.0 <= metrics["token_accuracy"] <= 100.0
    assert metrics["total_tokens"] == 64


def test_eval_cmd_cli(tmp_path):
    """Verify tiny-llm eval CLI command invocation."""
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("The cat sat on the mat.\nHello world!\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(eval_cmd, ["--dataset", str(corpus_path)])

    assert result.exit_code == 0
    assert "Cross-Entropy Loss" in result.output
    assert "Perplexity (PPL)" in result.output
