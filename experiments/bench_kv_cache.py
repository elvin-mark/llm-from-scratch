import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tiny_llm import TinyLLM, apply_rotary_emb


class KVCacheTinyLLM(nn.Module):
    """
    TinyLLM model equipped with an explicit stateful Key-Value (KV) Cache for O(1) per-token generation.
    """

    def __init__(
        self,
        vocab_size: int = 4000,
        dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int = 512,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.model = TinyLLM(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            max_seq_len=max_seq_len,
        )
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_seq_len = max_seq_len

    def forward_step_with_cache(
        self,
        token: torch.Tensor,
        pos: int,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        """
        Processes a SINGLE token at step `pos` using stateful pre-allocated Key and Value caches.
        """
        bsz = token.shape[0]
        h = self.model.tok_embeddings(token)
        freqs_cis = self.model.freqs_cis[pos : pos + 1].to(token.device)

        for l_idx, layer in enumerate(self.model.layers):
            # Attention RMSNorm
            norm_x = layer.attention_norm(h)

            # Compute Q, K, V for current token ONLY
            xq = layer.attention.wq(norm_x).view(bsz, 1, self.n_heads, self.head_dim)
            xk = layer.attention.wk(norm_x).view(bsz, 1, self.n_heads, self.head_dim)
            xv = layer.attention.wv(norm_x).view(bsz, 1, self.n_heads, self.head_dim)

            # Apply RoPE for current position
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

            xq = xq.transpose(1, 2)  # [bsz, n_heads, 1, head_dim]
            xk = xk.transpose(1, 2)  # [bsz, n_heads, 1, head_dim]
            xv = xv.transpose(1, 2)  # [bsz, n_heads, 1, head_dim]

            # Store current K, V into stateful cache buffers
            k_cache[l_idx, :, :, pos : pos + 1, :] = xk
            v_cache[l_idx, :, :, pos : pos + 1, :] = xv

            # Retrieve active cache keys and values up to pos + 1
            keys = k_cache[l_idx, :, :, : pos + 1, :]
            values = v_cache[l_idx, :, :, : pos + 1, :]

            # Attention scores over cached history [1..pos+1]
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            out_att = torch.matmul(scores, values)

            out_att = out_att.transpose(1, 2).contiguous().view(bsz, 1, -1)
            h = h + layer.attention.wo(out_att)

            # FFN Block
            h = h + layer.feed_forward(layer.ffn_norm(h))

        h = self.model.norm(h)
        logits = self.model.output(h)
        return logits


def run_kv_cache_benchmark():
    print("=" * 95)
    print("📊 BENCHMARK: Stateful KV-Cache vs. Non-Cached Quadratic Slowdown")
    print("=" * 95)

    vocab_size = 4000
    dim = 128
    n_layers = 4
    n_heads = 4
    ffn_dim = 512
    max_seq_len = 128
    generate_steps = 40
    batch_size = 1

    model = KVCacheTinyLLM(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
    )
    model.eval()

    start_token = torch.tensor([[1]])

    # --- 1. Non-Cached Generation Benchmark (Re-evaluates full sequence at each step) ---
    non_cached_step_times = []
    tokens_history = start_token.clone()

    t0_total = time.perf_counter()
    with torch.no_grad():
        for step in range(generate_steps):
            t0 = time.perf_counter()
            logits = model.model(tokens_history)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tokens_history = torch.cat([tokens_history, next_token], dim=1)
            t1 = time.perf_counter()
            non_cached_step_times.append((t1 - t0) * 1000.0)
    non_cached_total_ms = (time.perf_counter() - t0_total) * 1000.0

    # --- 2. Stateful KV-Cache Generation Benchmark (O(1) per token) ---
    cached_step_times = []
    head_dim = dim // n_heads
    k_cache = torch.zeros((n_layers, batch_size, n_heads, max_seq_len, head_dim))
    v_cache = torch.zeros((n_layers, batch_size, n_heads, max_seq_len, head_dim))

    current_token = start_token.clone()

    t0_total = time.perf_counter()
    with torch.no_grad():
        for step in range(generate_steps):
            t0 = time.perf_counter()
            logits = model.forward_step_with_cache(
                current_token, step, k_cache, v_cache
            )
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            current_token = next_token
            t1 = time.perf_counter()
            cached_step_times.append((t1 - t0) * 1000.0)
    cached_total_ms = (time.perf_counter() - t0_total) * 1000.0

    speedup = non_cached_total_ms / cached_total_ms

    print(
        f"  Generated Tokens | {'Non-Cached Step (ms)':<22} | {'Stateful KV-Cache Step (ms)':<28} | {'Latency Shift':<15}"
    )
    print("-" * 95)
    checkpoints_to_print = [1, 10, 20, 30, 40]
    for step_idx in checkpoints_to_print:
        nc_t = non_cached_step_times[step_idx - 1]
        c_t = cached_step_times[step_idx - 1]
        ratio = nc_t / max(c_t, 1e-5)
        print(
            f"  Step {step_idx:<11} | {nc_t:<22.3f} | {c_t:<28.3f} | {ratio:<5.1f}x slower non-cached"
        )

    print("-" * 95)
    print(
        f"⏱️ Total Latency for {generate_steps} Tokens: Non-Cached = {non_cached_total_ms:.2f} ms | KV-Cached = {cached_total_ms:.2f} ms"
    )
    print(
        f"💡 Key Takeaway: Stateful KV-Cache provides a {speedup:.2f}x OVERALL GENERATION SPEEDUP!"
    )
    print(
        "   Non-cached generation grows quadratically O(N^2) as sequence length extends,"
    )
    print(
        "   whereas KV-cache maintains flat O(1) constant step latency for all generated tokens!"
    )
    print("=" * 95 + "\n")


if __name__ == "__main__":
    run_kv_cache_benchmark()
