"""NextItNet: causal dilated 1D conv stack with residual blocks.

Improvements over the WSDM'19 paper / RecBole baseline:
- Tied input/output embeddings (saves |I|·d params, +1..3% NDCG).
- GLU activation inside the residual block (wavenet-style) instead of plain ReLU.
- Full-position generative training: loss is applied at every valid position
  via the same `score_pairs` API as SASRec, so we can use gBCE / sampled softmax.
- Optional side info embedding (genres/decade/genome) added to item embedding.

Architecture summary (one ResidualBlock_b):
    x  →  causal_pad(L)  →  Conv1d(d→d, k, dilation=D)  →  LayerNorm  →  GLU
       →  causal_pad(L)  →  Conv1d(d→d, k, dilation=2D) →  LayerNorm  →  GLU
       →  + x
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NextItNetConfig:
    n_items: int
    d: int = 128
    kernel_size: int = 3
    block_num: int = 4                       # repeat dilations this many times
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.2
    max_len: int = 200
    use_glu: bool = True
    use_side: bool = False
    side_use_genome: bool = True


class CausalConv1d(nn.Module):
    """Left-padded 1D conv to preserve causality."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class GLU(nn.Module):
    """Standard GLU: split features in half, sigmoid-gate one half by the other."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)              # split along channel dim
        return a * torch.sigmoid(b)


class ResidualBlockB(nn.Module):
    """Two stacked dilated convs with normalisation + activation, residual added.

    With GLU we keep param count by emitting 2*d channels from each conv and
    halving via the gate.
    """

    def __init__(self, d: int, kernel_size: int, dilation: int, dropout: float, use_glu: bool):
        super().__init__()
        self.use_glu = use_glu
        out_mult = 2 if use_glu else 1
        self.conv1 = CausalConv1d(d, d * out_mult, kernel_size, dilation)
        self.ln1 = nn.LayerNorm(d * out_mult, eps=1e-5)
        self.act1 = GLU() if use_glu else nn.ReLU()
        self.conv2 = CausalConv1d(d, d * out_mult, kernel_size, dilation * 2)
        self.ln2 = nn.LayerNorm(d * out_mult, eps=1e-5)
        self.act2 = GLU() if use_glu else nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def _ln_over_channels(self, x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
        # x: [B, C, L] → LayerNorm expects last dim, so transpose.
        return ln(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C=d, L]
        h = self.conv1(x)
        h = self._ln_over_channels(h, self.ln1)
        h = self.act1(h)
        h = self.conv2(h)
        h = self._ln_over_channels(h, self.ln2)
        h = self.act2(h)
        return x + self.drop(h)


class NextItNet(nn.Module):
    def __init__(self, cfg: NextItNetConfig, side_module: nn.Module | None = None):
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(cfg.n_items + 1, cfg.d, padding_idx=0)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.side = side_module
        full_dilations = list(cfg.dilations) * cfg.block_num
        self.blocks = nn.ModuleList([
            ResidualBlockB(cfg.d, cfg.kernel_size, d, cfg.dropout, cfg.use_glu)
            for d in full_dilations
        ])
        self.final_ln = nn.LayerNorm(cfg.d)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0.0)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def output_embedding(self) -> torch.Tensor:
        return self.item_emb.weight

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.item_emb(input_ids)                          # [B, L, d]
        if self.side is not None:
            x = x + self.side(input_ids)
        x = self.emb_drop(x)
        # Conv1d expects [B, C, L]
        x = x.transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2)                                 # back to [B, L, d]
        return self.final_ln(x)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encode(input_ids)

    @torch.no_grad()
    def score_all(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.encode(input_ids)
        last = h[:, -1, :]
        return last @ self.output_embedding.T

    def score_pairs(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        neg_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(input_ids)
        E = self.output_embedding
        pos_emb = E[target_ids]
        pos_logits = (h * pos_emb).sum(dim=-1)
        neg_emb = E[neg_ids]
        neg_logits = torch.einsum("bld,blkd->blk", h, neg_emb)
        target_mask = (input_ids != 0)
        return pos_logits, neg_logits, target_mask
