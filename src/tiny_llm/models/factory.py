import json
import os
import torch

from tiny_llm.configs import (
    BitNetConfig,
    MoELLMConfig,
    NanoLLMConfig,
    TinyLLMConfig,
)
from tiny_llm.models.bitnet_llm import BitNetLLM
from tiny_llm.models.dense_llm import TinyLLM
from tiny_llm.models.moe_llm import MoELLM
from tiny_llm.models.nano_llm import NanoLLM

MODEL_REGISTRY = {
    "dense": (TinyLLMConfig, TinyLLM),
    "moe": (MoELLMConfig, MoELLM),
    "nano": (NanoLLMConfig, NanoLLM),
    "bitnet": (BitNetConfig, BitNetLLM),
}


def create_model(arch: str = None, **kwargs):
    """
    Factory method to create a model and its config for any supported architecture.

    Supported architectures: 'dense', 'moe', 'nano', 'bitnet'.
    """
    if arch is None:
        arch = kwargs.pop("arch", "dense")
    elif "arch" in kwargs:
        kwargs.pop("arch")

    arch = arch.lower()
    if arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch}'. Supported: {list(MODEL_REGISTRY.keys())}"
        )

    config_cls, model_cls = MODEL_REGISTRY[arch]

    # Create config object
    cfg_args = {"arch": arch, **kwargs}
    config = config_cls.from_dict(cfg_args)

    # Instantiate model using config=config or kwargs
    if arch in ("dense", "moe", "bitnet"):
        model = model_cls(config=config)
    elif arch == "nano":
        model = model_cls(
            vocab_size=config.vocab_size,
            dim=config.dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            ffn_dim=config.ffn_dim,
            max_seq_len=config.max_seq_len,
        )

    return model, config


def load_model_from_checkpoint(checkpoint_path: str, device=None):
    """
    Loads a model and auto-detects its architecture config from checkpoints.
    Looks for config.json or <checkpoint_name>.json in the checkpoint directory.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = os.path.dirname(checkpoint_path)
    base_name = os.path.splitext(checkpoint_path)[0]

    # Look for matching json config
    config_path = base_name + ".json"
    if not os.path.exists(config_path):
        config_path = os.path.join(ckpt_dir, "config.json")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        arch = cfg_dict.pop("arch", "dense")
        model, config = create_model(arch=arch, **cfg_dict)
    else:
        # Default fallback to dense model
        config = TinyLLMConfig(arch="dense")
        model = TinyLLM(config=config)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config
