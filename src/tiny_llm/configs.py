import json
import os
from dataclasses import asdict, dataclass


@dataclass
class TinyLLMConfig:
    arch: str = "dense"
    vocab_size: int = 4000
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ffn_dim: int = 512
    max_seq_len: int = 128

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict):
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)


@dataclass
class MoELLMConfig(TinyLLMConfig):
    arch: str = "moe"
    n_kv_heads: int = 2  # GQA: Key/Value heads count
    num_experts: int = 8  # MoE: Total expert networks
    num_experts_per_tok: int = 2  # MoE: Top-k experts selected per token


@dataclass
class NanoLLMConfig(TinyLLMConfig):
    arch: str = "nano"


@dataclass
class BitNetConfig(TinyLLMConfig):
    arch: str = "bitnet"
