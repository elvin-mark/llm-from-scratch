from dataclasses import dataclass


@dataclass
class TinyLLMConfig:
    vocab_size: int = 4000
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ffn_dim: int = 512
    max_seq_len: int = 128


@dataclass
class MoELLMConfig(TinyLLMConfig):
    n_kv_heads: int = 2  # GQA: Key/Value heads count (expansion ratio = n_heads // n_kv_heads)
    num_experts: int = 8  # MoE: Total expert networks
    num_experts_per_tok: int = 2  # MoE: Top-k experts selected per token
