import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rotary_emb


class KVCache(nn.Module):
    """
    Pre-allocated Key-Value Cache buffer for autoregressive token decoding.
    Reduces single-token inference cost from O(N^2) to O(1).
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.register_buffer(
            "k",
            torch.zeros(
                (max_batch_size, n_heads, max_seq_len, head_dim),
                device=device,
                dtype=dtype,
            ),
            persistent=False,
        )
        self.register_buffer(
            "v",
            torch.zeros(
                (max_batch_size, n_heads, max_seq_len, head_dim),
                device=device,
                dtype=dtype,
            ),
            persistent=False,
        )

    def update(self, start_pos: int, k_val: torch.Tensor, v_val: torch.Tensor):
        """
        Updates cache at start_pos and returns accumulated keys and values up to (start_pos + seqlen).
        """
        bsz, n_heads, seqlen, head_dim = k_val.shape
        self.k[:bsz, :, start_pos : start_pos + seqlen, :] = k_val
        self.v[:bsz, :, start_pos : start_pos + seqlen, :] = v_val

        keys = self.k[:bsz, :, : start_pos + seqlen, :]
        values = self.v[:bsz, :, : start_pos + seqlen, :]
        return keys, values


class Attention(nn.Module):
    """
    Standard Multi-Head Attention (MHA) mechanism with Rotary Embeddings & KV Cache support.
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
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor = None,
        start_pos: int = 0,
        kv_cache: KVCache = None,
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

        if kv_cache is not None:
            keys, values = kv_cache.update(start_pos, xk, xv)
        else:
            keys, values = xk, xv

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)

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
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor = None,
        start_pos: int = 0,
        kv_cache: KVCache = None,
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

        if kv_cache is not None:
            xk, xv = kv_cache.update(start_pos, xk, xv)

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
            m_i = torch.full((bsz, self.n_heads, q_len, 1), float("-inf"), device=x.device)
            l_i = torch.zeros((bsz, self.n_heads, q_len, 1), device=x.device)
            acc_i = torch.zeros((bsz, self.n_heads, q_len, self.head_dim), device=x.device)

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


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) mechanism (DeepSeek-V3 / DeepSeek-R1).
    Compresses Key and Value projections into a shared low-rank latent vector (c^KV),
    reducing KV-cache memory footprint by up to 90% while maintaining full MHA quality.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        kv_lora_rank: int = 32,
        q_lora_rank: int = 64,
        rope_dim: int = 16,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.rope_dim = rope_dim

        # 1. KV Low-Rank Compression & Up-Projections
        self.w_dkv = nn.Linear(dim, kv_lora_rank, bias=False)
        self.w_uk = nn.Linear(kv_lora_rank, n_heads * self.head_dim, bias=False)
        self.w_uv = nn.Linear(kv_lora_rank, n_heads * self.head_dim, bias=False)

        # 2. Decoupled RoPE Key Projection
        self.w_kr = nn.Linear(dim, rope_dim, bias=False)

        # 3. Query Compression & Up-Projections
        self.w_dq = nn.Linear(dim, q_lora_rank, bias=False)
        self.w_uq = nn.Linear(q_lora_rank, n_heads * self.head_dim, bias=False)
        self.w_qr = nn.Linear(q_lora_rank, n_heads * rope_dim, bias=False)

        # 4. Output Projection
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        # 1. Compress KV to low-rank latent vector c_kv: [bsz, seqlen, kv_lora_rank]
        c_kv = self.w_dkv(x)

        # 2. Up-project c_kv to Content Keys and Values
        xk_c = self.w_uk(c_kv).view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = self.w_uv(c_kv).view(bsz, seqlen, self.n_heads, self.head_dim)

        # 3. Compute Decoupled Positional Key (K^R): [bsz, seqlen, 1, rope_dim] -> broadcast across heads
        xk_r = self.w_kr(x).view(bsz, seqlen, 1, self.rope_dim)

        # 4. Compress Query to c_q, up-project to Content Query and Positional Query
        c_q = self.w_dq(x)
        xq_c = self.w_uq(c_q).view(bsz, seqlen, self.n_heads, self.head_dim)
        xq_r = self.w_qr(c_q).view(bsz, seqlen, self.n_heads, self.rope_dim)

        # 5. Apply RoPE to Positional Queries (Q^R) and Positional Keys (K^R)
        xk_r_exp = xk_r.expand(bsz, seqlen, self.n_heads, self.rope_dim)
        xq_r, xk_r_exp = apply_rotary_emb(xq_r, xk_r_exp, freqs_cis)

        # 6. Transpose to [bsz, n_heads, seqlen, head_dim / rope_dim]
        xq_c = xq_c.transpose(1, 2)
        xk_c = xk_c.transpose(1, 2)
        xv = xv.transpose(1, 2)

        xq_r = xq_r.transpose(1, 2)
        xk_r_exp = xk_r_exp.transpose(1, 2)

        # 7. Compute Combined Attention Scores: (Q^C * K^C + Q^R * K^R) / sqrt(head_dim + rope_dim)
        scores_c = torch.matmul(xq_c, xk_c.transpose(2, 3))
        scores_r = torch.matmul(xq_r, xk_r_exp.transpose(2, 3))

        scale = 1.0 / math.sqrt(self.head_dim + self.rope_dim)
        scores = (scores_c + scores_r) * scale

        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq_c)
        output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
