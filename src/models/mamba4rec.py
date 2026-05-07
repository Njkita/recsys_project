"""Mamba4Rec — selective state-space model for sequential recommendation.

Reference: Liu et al., RelKD'24, arXiv:2403.03900
Paper code:  github.com/chengkai-liu/Mamba4Rec

Drop-in replacement for the attention block of SASRec; the rest of the
training pipeline (loss, eval, side info, EMA) stays identical.

DEPENDENCIES (installed via `scripts/install_mamba.sh`):
  pip install causal-conv1d==1.4.0 mamba-ssm==2.2.4

Wheels for torch 2.5 + cu12 + cp310 are published on the GitHub releases pages
of both libraries. The setup script picks the right wheel automatically.

If `mamba_ssm` is not importable, this module raises a helpful error at model
construction time — the rest of the codebase still works (registry skips the
import lazily).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba                  # type: ignore
    _HAS_MAMBA = True
    _MAMBA_IMPORT_ERROR = None
except Exception as e:                           # noqa: BLE001
    Mamba = None                                  # type: ignore
    _HAS_MAMBA = False
    _MAMBA_IMPORT_ERROR = e


@dataclass
class Mamba4RecConfig:
    n_items: int
    d: int = 128
    n_blocks: int = 2                # paper: 2 blocks works best on ML-1M
    d_state: int = 32
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1
    ffn_mult: int = 4
    max_len: int = 200
    use_side: bool = False
    side_use_genome: bool = True


class Mamba4RecBlock(nn.Module):
    def __init__(self, cfg: Mamba4RecConfig):
        super().__init__()
        if not _HAS_MAMBA:
            raise ImportError(
                "mamba_ssm is required for Mamba4Rec. Install via "
                "`scripts/install_mamba.sh`. Underlying error: "
                f"{_MAMBA_IMPORT_ERROR!r}"
            )
        self.ln1 = nn.LayerNorm(cfg.d)
        self.mamba = Mamba(
            d_model=cfg.d,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
        )
        self.drop1 = nn.Dropout(cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d)
        hidden = ((cfg.ffn_mult * cfg.d * 2 // 3 + 7) // 8) * 8
        self.ffn_w1 = nn.Linear(cfg.d, hidden, bias=False)
        self.ffn_w2 = nn.Linear(cfg.d, hidden, bias=False)
        self.ffn_w3 = nn.Linear(hidden, cfg.d, bias=False)
        self.drop2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mamba is causal by construction; no mask required.
        x = x + self.drop1(self.mamba(self.ln1(x)))
        h = self.ffn_w3(F.silu(self.ffn_w1(self.ln2(x))) * self.ffn_w2(self.ln2(x)))
        return x + self.drop2(h)


class Mamba4Rec(nn.Module):
    def __init__(self, cfg: Mamba4RecConfig, side_module: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(cfg.n_items + 1, cfg.d, padding_idx=0)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Mamba4RecBlock(cfg) for _ in range(cfg.n_blocks)])
        self.final_ln = nn.LayerNorm(cfg.d)
        self.side = side_module
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0.0)

    @property
    def output_embedding(self) -> torch.Tensor:
        return self.item_emb.weight

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.item_emb(input_ids)
        if self.side is not None:
            x = x + self.side(input_ids)
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
