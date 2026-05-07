"""Linear-attention SASRec — same architecture as SASRec but the attention
block is replaced with causal linear attention via the elu+1 feature map.

Reference: Katharopoulos et al., ICML 2020, arXiv:2006.16236
("Transformers are RNNs: Fast Autoregressive Transformers with Linear
Attention").

Math: replace softmax(QKᵀ)V with φ(Q)·(φ(K)ᵀ V). For causal models the inner
sum becomes a prefix sum (cumsum) over keys/values — O(L·d²) memory in the
naive PyTorch form, but linear in L for compute. Theoretically constant-time
per-token at inference (RNN-style hidden state of size d_φ × d_v).

For ML-20M with L=200 this is **slower** than FlashAttention but useful as a
pedagogical comparison and as a stepping stone for synthetic-L experiments
where SASRec's quadratic attention starts to dominate (L>=2000).

The block plugs into the same `LiGRBlock`-style residual scaffolding as the
main SASRec, sharing item embeddings, RoPE positional bias, SwiGLU FFN, and
tied output projection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sasrec import RotaryEmbedding, SwiGLU, SASRecConfig


@dataclass
class LinearAttnConfig(SASRecConfig):
    """Reuses SASRecConfig defaults; presence of this class is for explicit
    typing in the model registry."""
    pass


class CausalLinearAttention(nn.Module):
    """elu+1 feature map, causal via cumulative sums. Per-batch [B, L, d]."""

    def __init__(self, d: int, n_heads: int, max_len: int, dropout: float, use_rope: bool):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"d ({d}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.resid_drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_len) if use_rope else None
        self.eps = 1e-6

    @staticmethod
    def phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d];  key_padding_mask: [B, L] True at PAD.
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                              # each [B, L, H, hd]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        q = self.phi(q)                                          # [B, H, L, hd]
        k = self.phi(k)

        # Zero out PAD positions in keys/values so they don't contribute to the
        # prefix sums.
        keep = (~key_padding_mask).to(x.dtype).view(B, 1, L, 1)
        k = k * keep
        v = v * keep

        # KV outer product per token, then cumsum across L.
        kv = torch.einsum("bhld,bhle->bhlde", k, v)              # [B, H, L, hd, hd]
        kv_cum = kv.cumsum(dim=2)
        k_cum = k.cumsum(dim=2)                                  # [B, H, L, hd]
        num = torch.einsum("bhld,bhlde->bhle", q, kv_cum)        # [B, H, L, hd]
        den = torch.einsum("bhld,bhld->bhl", q, k_cum).unsqueeze(-1).clamp_min(self.eps)
        y = (num / den).transpose(1, 2).contiguous().view(B, L, D)
        return self.resid_drop(self.out(y))


class LinearLiGRBlock(nn.Module):
    """Same scaffolding as `LiGRBlock` but with linear attention inside."""

    def __init__(self, cfg: SASRecConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d)
        self.attn = CausalLinearAttention(cfg.d, cfg.n_heads, cfg.max_len, cfg.dropout, cfg.use_rope)
        self.ln2 = nn.LayerNorm(cfg.d)
        self.ffn = SwiGLU(cfg.d, mult=cfg.ffn_mult, dropout=cfg.dropout)
        if cfg.use_ligr_gates:
            self.g_attn = nn.Parameter(torch.zeros(cfg.d))
            self.g_ffn = nn.Parameter(torch.zeros(cfg.d))
        else:
            self.register_parameter("g_attn", None)
            self.register_parameter("g_ffn", None)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        a = self.attn(self.ln1(x), key_padding_mask)
        if self.g_attn is not None:
            a = a * self.g_attn
        x = x + a
        f = self.ffn(self.ln2(x))
        if self.g_ffn is not None:
            f = f * self.g_ffn
        return x + f


class LinearAttnSASRec(nn.Module):
    def __init__(self, cfg: SASRecConfig, side_module: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(cfg.n_items + 1, cfg.d, padding_idx=0)
        self.use_pos_emb = not cfg.use_rope
        if self.use_pos_emb:
            self.pos_emb = nn.Embedding(cfg.max_len, cfg.d)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([LinearLiGRBlock(cfg) for _ in range(cfg.n_blocks)])
        self.final_ln = nn.LayerNorm(cfg.d)
        self.side = side_module
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0.0)
        if self.use_pos_emb:
            nn.init.normal_(self.pos_emb.weight, std=0.02)

    @property
    def output_embedding(self) -> torch.Tensor:
        return self.item_emb.weight

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        pad_mask = input_ids == 0
        x = self.item_emb(input_ids)
        if self.side is not None:
            x = x + self.side(input_ids)
        if self.use_pos_emb:
            pos = torch.arange(L, device=input_ids.device)
            x = x + self.pos_emb(pos).unsqueeze(0)
        x = self.emb_drop(x)
        for blk in self.blocks:
            x = blk(x, pad_mask)
        return self.final_ln(x)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encode(input_ids)

    @torch.no_grad()
    def score_all(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.encode(input_ids)
        return h[:, -1, :] @ self.output_embedding.T

    def score_pairs(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        neg_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(input_ids)
        E = self.output_embedding
        pos_logits = (h * E[target_ids]).sum(dim=-1)
        neg_logits = torch.einsum("bld,blkd->blk", h, E[neg_ids])
        return pos_logits, neg_logits, (input_ids != 0)
