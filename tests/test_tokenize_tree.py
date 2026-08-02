import json

from click.testing import CliRunner

from tiny_llm.cli.tokenize_tree import tokenize_tree_cmd
from tiny_llm.tokenizer import ScratchTokenizer


def test_tokenize_tree_cli(tmp_path):
    """Verify tiny-llm tokenize-tree CLI command execution."""
    text = "Unbelievable processing of subwords!"
    tok_data = ScratchTokenizer.train(text, vocab_size=50)
    tok_path = tmp_path / "tokenizer.json"
    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump(tok_data, f, ensure_ascii=False, indent=2)

    runner = CliRunner()
    result = runner.invoke(
        tokenize_tree_cmd,
        [
            "--text",
            text,
            "--tokenizer-path",
            str(tok_path),
            "--use-scratch-tokenizer",
        ],
    )

    assert result.exit_code == 0
    assert "BPE Subword Tokenizer Tree Visualizer" in result.output
    assert "Sequential Token Decomposition Table" in result.output
