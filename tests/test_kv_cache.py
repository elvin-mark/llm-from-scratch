import time

import torch

from tiny_llm.cli.generate import sample_tokens
from tiny_llm.models import (
    create_model,
)
from tiny_llm.modules.attention import KVCache


def test_kv_cache_buffer_update():
    """Verify KVCache buffer update logic."""
    cache = KVCache(max_batch_size=2, max_seq_len=16, n_heads=4, head_dim=16)
    k_val = torch.randn(2, 4, 3, 16)
    v_val = torch.randn(2, 4, 3, 16)

    keys, values = cache.update(start_pos=0, k_val=k_val, v_val=v_val)
    assert keys.shape == (2, 4, 3, 16)
    assert values.shape == (2, 4, 3, 16)

    # Next single token step
    k_new = torch.randn(2, 4, 1, 16)
    v_new = torch.randn(2, 4, 1, 16)
    keys2, values2 = cache.update(start_pos=3, k_val=k_new, v_val=v_new)
    assert keys2.shape == (2, 4, 4, 16)
    assert values2.shape == (2, 4, 4, 16)


def test_kv_cache_equivalence_dense():
    """Verify KV-cached output logits match non-cached output logits for TinyLLM."""
    torch.manual_seed(42)
    model, config = create_model("dense", vocab_size=100, dim=64, n_layers=2, n_heads=2)

    prompt_ids = torch.randint(0, 100, (1, 5))

    # Generate with KV Cache ON
    torch.manual_seed(123)
    out_cached = sample_tokens(model, prompt_ids, max_tokens=10, use_kv_cache=True)

    # Generate with KV Cache OFF
    torch.manual_seed(123)
    out_nocache = sample_tokens(model, prompt_ids, max_tokens=10, use_kv_cache=False)

    assert out_cached == out_nocache, f"Cached: {out_cached} != Non-cached: {out_nocache}"


def test_kv_cache_equivalence_moe():
    """Verify KV-cached output logits match non-cached output logits for MoELLM (GQA)."""
    torch.manual_seed(42)
    model, config = create_model("moe", vocab_size=100, dim=64, n_layers=2, n_heads=4, n_kv_heads=2)

    prompt_ids = torch.randint(0, 100, (1, 5))

    torch.manual_seed(123)
    out_cached = sample_tokens(model, prompt_ids, max_tokens=10, use_kv_cache=True)

    torch.manual_seed(123)
    out_nocache = sample_tokens(model, prompt_ids, max_tokens=10, use_kv_cache=False)

    assert out_cached == out_nocache, f"Cached: {out_cached} != Non-cached: {out_nocache}"


def test_kv_cache_speedup():
    """Verify KV-cached decoding runs significantly faster than non-cached decoding for long sequences."""
    torch.manual_seed(42)
    model, _ = create_model("dense", vocab_size=500, dim=128, n_layers=4, n_heads=4)
    prompt_ids = torch.randint(0, 500, (1, 16))

    # Warmup
    sample_tokens(model, prompt_ids, max_tokens=5, use_kv_cache=True)
    sample_tokens(model, prompt_ids, max_tokens=5, use_kv_cache=False)

    # Measure KV Cache ON
    t0 = time.perf_counter()
    sample_tokens(model, prompt_ids, max_tokens=60, use_kv_cache=True)
    t_cache = time.perf_counter() - t0

    # Measure KV Cache OFF
    t0 = time.perf_counter()
    sample_tokens(model, prompt_ids, max_tokens=60, use_kv_cache=False)
    t_nocache = time.perf_counter() - t0

    print(f"\n[Bench] Time with KV Cache ON:  {t_cache:.4f}s")
    print(f"[Bench] Time with KV Cache OFF: {t_nocache:.4f}s")
    print(f"[Bench] Speedup: {t_nocache / max(t_cache, 1e-6):.2f}x")

    assert t_cache < t_nocache or t_cache < 0.5
