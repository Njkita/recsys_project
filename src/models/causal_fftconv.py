"""CausalFFTConvRec — strictly-causal FFT-based sequential recommender.

Motivation. Standard FFT/FMLP models (FMLP-Rec, FNet) are NOT causal: a single
FFT over the sequence axis mixes future positions into past ones, so at training
time position t can already "see" input[t+1..L-1]. This leakage typically yields
strong train loss but weak test NDCG.

This model parametrizes a learnable kernel K of length L in the *time* domain
(depthwise per channel). The forward computes the causal convolution
    y[t] = sum_{s=0}^{t} x[s] * K[t - s]      (zero outside [0, L-1])
via FFT for O(L log L) speed (zero-pad to 2L, FFT, multiply, irFFT, take first L).
Strict causality by construction. Same trick as Hyena (Poli et al., 2023) and
S4/S4D (Gu et al.).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CausalFFTConvConfig:
    n_items: int
    d: int = 256
    n_blocks: int = 3
    max_len: int = 200
    dropout: float = 0.2
    ffn_mult: int = 4
    use_side: bool = False
    side_use_genome: bool = False


class CausalLongConv(nn.Module):
    def __init__(self, d: int, max_len: int):
        super().__init__()
        self.max_len = max_len
        std = max_len ** -0.5
        self.kernel = nn.Parameter(torch.randn(d, max_len) * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        n = 1
        while n < 2 * L:
            n *= 2
        x_t = x.transpose(1, 2)
        k = self.kernel[:, :L]
        Xf = torch.fft.rfft(x_t.float(), n=n, dim=-1)
        Kf = torch.fft.rfft(k.float(),   n=n, dim=-1)
        Yf = Xf * Kf.unsqueeze(0)
        y = torch.fft.irfft(Yf, n=n, dim=-1)
        y = y[..., :L]
        return y.to(x.dtype).transpose(1, 2)


class SwiGLUFFN(nn.Module):
    def __init__(self, d: int, mult: int = 4, dropout: float = 0.2):
        super().__init__()
        hidden = ((mult * d * 2 // 3 + 7) // 8) * 8
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class CausalFFTConvBlock(nn.Module):
    def __init__(self, cfg: CausalFFTConvConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d)
        self.conv = CausalLongConv(cfg.d, cfg.max_len)
        self.drop1 = nn.Dropout(cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d)
        self.ffn = SwiGLUFFN(cfg.d, mult=cfg.ffn_mult, dropout=cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.conv(self.ln1(x)))
        x = x + self.ffn(self.ln2(x))
        return x


class CausalFFTConvRec(nn.Module):
    def __init__(self, cfg: CausalFFTConvConfig, side_module: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(cfg.n_items + 1, cfg.d, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([CausalFFTConvBlock(cfg) for _ in range(cfg.n_blocks)])
        self.final_ln = nn.LayerNorm(cfg.d)
        self.side = side_module
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0.0)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    @property
    def output_embedding(self) -> torch.Tensor:
        return self.item_emb.weight

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        x = self.item_emb(input_ids)
        if self.side is not None:
            x = x + self.side(input_ids)
        pos = torch.arange(L, device=input_ids.device)
        x = x + self.pos_emb(pos).unsqueeze(0)
        x = self.emb_drop(x)
        for blk in self.blocks:
            x = blk(x)
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
