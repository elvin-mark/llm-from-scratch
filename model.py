import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Model Architecture ---
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    A variant of LayerNorm that removes the mean-centering operation, improving
    training speed while maintaining similar performance. Used in Llama architectures.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim, end, theta=10000.0):
    """
    Precomputes complex frequencies for Rotary Position Embeddings (RoPE).

    Args:
        dim (int): The dimension of the attention head.
        end (int): The maximum sequence length to precompute for.
        theta (float): The base for the frequency scaling.

    Returns:
        torch.Tensor: Complex exponential tensor containing frequencies.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis, x):
    """
    Reshapes the precomputed frequencies tensor to match the target tensor for broadcasting.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Applies Rotary Position Embeddings (RoPE) to query and key tensors.

    This rotates the features of the queries and keys to inject absolute
    positional information into the attention mechanism relative to their positions.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """
    Standard Multi-Head Attention mechanism enriched with Rotary Embeddings.
    """

    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Projection matrices for Query, Key, Value without bias (Llama standard)
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, freqs_cis, mask=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        # Inject positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Transpose for attention
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # Apply causal mask

        # Calculate attention weights and apply to values
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Instead of the traditional two-layer MLP with ReLU/GELU, this uses three linear
    projections and a Swish (SiLU) activation gate, as seen in Llama.
    """

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU activation formula: (SiLU(x * W1) * (x * W3)) * W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))  # SwiGLU


class TransformerBlock(nn.Module):
    """
    A single Transformer layer combining Attention and FeedForward blocks with pre-RMSNorm.
    """

    def __init__(self, dim, n_heads, ffn_dim):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(dim, ffn_dim)

        # Pre-normalization (norm applied before the residual connection)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x, freqs_cis, mask):
        # Attention sub-layer
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        # Feed-forward sub-layer
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class TinyLLM(nn.Module):
    """
    The main Causal Language Model built using a Llama-like architecture.
    """

    def __init__(
        self, vocab_size, dim=128, n_layers=4, n_heads=4, ffn_dim=512, max_seq_len=128
    ):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, n_heads, ffn_dim) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # Precompute rotational frequencies for the max possible sequence length
        self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len * 2)

    def forward(self, tokens):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:seqlen].to(tokens.device)

        # Create causal mask for autoregressive generation
        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=1)

        # Pass through all transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h)
        return output
