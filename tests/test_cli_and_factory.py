import os
import tempfile

import torch

from tiny_llm.configs import MoELLMConfig
from tiny_llm.models import (
    MoELLM,
    create_model,
    load_model_from_checkpoint,
)


def test_create_model_all_architectures():
    """Verify factory can create dense, moe, nano, and bitnet models."""
    for arch in ["dense", "moe", "nano", "bitnet"]:
        model, config = create_model(arch=arch, vocab_size=200, dim=64, n_layers=2, n_heads=2)
        assert model is not None
        assert config.arch == arch
        assert config.vocab_size == 200

        # Run dummy forward pass
        dummy_input = torch.randint(0, 200, (1, 8))
        logits = model(dummy_input)
        assert logits.shape == (1, 8, 200)


def test_config_json_serialization():
    """Verify config JSON saving and loading."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = os.path.join(tmp_dir, "config.json")
        cfg = MoELLMConfig(arch="moe", num_experts=4, num_experts_per_tok=2)
        cfg.save_json(config_path)

        assert os.path.exists(config_path)
        loaded_cfg = MoELLMConfig.from_json(config_path)
        assert loaded_cfg.arch == "moe"
        assert loaded_cfg.num_experts == 4


def test_checkpoint_auto_detection():
    """Verify load_model_from_checkpoint auto-detects architecture from config.json."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "moe_model.pth")
        config_path = os.path.join(tmp_dir, "config.json")

        # Create & save MoE model
        model, config = create_model("moe", vocab_size=100, dim=32, n_layers=2, n_heads=2)
        torch.save(model.state_dict(), ckpt_path)
        config.save_json(config_path)

        # Auto-load from checkpoint
        loaded_model, loaded_config = load_model_from_checkpoint(ckpt_path)
        assert loaded_config.arch == "moe"
        assert isinstance(loaded_model, MoELLM)
