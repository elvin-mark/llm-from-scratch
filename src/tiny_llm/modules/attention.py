import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rotary_emb


class Attention(nn.Module):
    """
    Standard Multi-Head Attention (MHA) mechanism with Rotary Embeddings.
    """

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) mechanism.
    Reduces KV cache size by sharing Key/Value heads across Query head groups.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeats Key/Value heads to match Query heads count."""
        if self.n_rep == 1:
            return x
        bsz, n_kv, seqlen, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(bsz, n_kv, self.n_rep, seqlen, head_dim)
            .reshape(bsz, n_kv * self.n_rep, seqlen, head_dim)
        )

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Transpose to [bsz, n_heads/n_kv_heads, seqlen, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Repeat KV heads for GQA broadcast
        keys = self.repeat_kv(xk)
        values = self.repeat_kv(xv)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class EducationalFlashAttention(nn.Module):
    """
    Pure PyTorch educational implementation of FlashAttention (Dao et al.).
    Computes exact attention using block-tiled online softmax with ZERO N x N matrix allocations.
    """

    def __init__(self, dim: int, n_heads: int, block_size: int = 16):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.block_size = block_size

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Transpose to [bsz, n_heads, seqlen, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        output = torch.zeros_like(xq)

        # Tiled online softmax loop over Query blocks (i) and Key/Value blocks (j)
        for i_start in range(0, seqlen, self.block_size):
            i_end = min(i_start + self.block_size, seqlen)
            q_block = xq[:, :, i_start:i_end, :]  # [bsz, n_heads, q_len, head_dim]
            q_len = i_end - i_start

            # Initialize running max (m), running sum (l), and output accumulator (acc)
            m_i = torch.full(
                (bsz, self.n_heads, q_len, 1), float("-inf"), device=x.device
            )
            l_i = torch.zeros((bsz, self.n_heads, q_len, 1), device=x.device)
            acc_i = torch.zeros(
                (bsz, self.n_heads, q_len, self.head_dim), device=x.device
            )

            # Process each Key/Value block j
            for j_start in range(0, seqlen, self.block_size):
                j_end = min(j_start + self.block_size, seqlen)

                # Skip blocks strictly after current query block under causal mask
                if mask is not None and j_start >= i_end:
                    continue

                k_block = xk[:, :, j_start:j_end, :]  # [bsz, n_heads, kv_len, head_dim]
                v_block = xv[:, :, j_start:j_end, :]  # [bsz, n_heads, kv_len, head_dim]

                # Tile logits: S_ij = Q_i @ K_j.T * scale
                scores_ij = torch.matmul(q_block, k_block.transpose(2, 3)) * scale

                if mask is not None:
                    mask_ij = mask[:, :, i_start:i_end, j_start:j_end]
                    scores_ij = scores_ij + mask_ij

                # Block row max
                m_ij = torch.max(scores_ij, dim=-1, keepdim=True).values

                # New running max across blocks
                m_new = torch.maximum(m_i, m_ij)

                # Rescale factors for online softmax update
                alpha = torch.exp(m_i - m_new)
                p_ij = torch.exp(scores_ij - m_new)

                # Update running sum of exponentials
                l_new = alpha * l_i + p_ij.sum(dim=-1, keepdim=True)

                # Update output accumulator: acc_new = alpha * acc_i + p_ij @ V_j
                acc_i = alpha * acc_i + torch.matmul(p_ij, v_block)

                # Update running statistics for next block iteration
                m_i = m_new
                l_i = l_new

            # Final normalization for Query block i: O_i = acc_i / l_i
            output[:, :, i_start:i_end, :] = acc_i / l_i

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

