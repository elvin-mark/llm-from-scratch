from click.testing import CliRunner

from tiny_llm.cli.info import info_cmd
from tiny_llm.models import create_model


def test_info_cmd_dense(tmp_path):
    """Verify tiny-llm info runs cleanly on dense model checkpoint."""
    model, config = create_model("dense", vocab_size=200, dim=64, n_layers=2, n_heads=2)
    ckpt_path = tmp_path / "test_dense.pth"
    config_path = tmp_path / "config.json"

    import torch

    torch.save(model.state_dict(), ckpt_path)
    config.save_json(config_path)

    runner = CliRunner()
    result = runner.invoke(info_cmd, ["--checkpoint", str(ckpt_path)])

    assert result.exit_code == 0
    assert "DENSE" in result.output
    assert "FP32" in result.output
    assert "BitNet 1.58-Bit" in result.output


def test_info_cmd_moe(tmp_path):
    """Verify tiny-llm info runs cleanly on MoE model checkpoint."""
    model, config = create_model("moe", vocab_size=200, dim=64, n_layers=2, n_heads=4, n_kv_heads=2)
    ckpt_path = tmp_path / "test_moe.pth"
    config_path = tmp_path / "config.json"

    import torch

    torch.save(model.state_dict(), ckpt_path)
    config.save_json(config_path)

    runner = CliRunner()
    result = runner.invoke(info_cmd, ["--checkpoint", str(ckpt_path)])

    assert result.exit_code == 0
    assert "MOE" in result.output
    assert "GQA ratio" in result.output
