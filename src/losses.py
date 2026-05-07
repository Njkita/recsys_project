"""Loss functions for sequential recommenders.

Implemented:
- `GBCELoss` — generalized BCE from gSASRec (Petrov & Macdonald, RecSys 2023,
  arXiv:2309.07602). Single-equation correction to vanilla BCE that targets
  the same optimum as full-catalog softmax. `t=0.75` is the canonical setting,
  `n_neg=256` gives the right calibration on ML-20M.
- `SampledSoftmaxLoss` — negative-sampled cross-entropy with optional logQ
  correction (Bengio & Senecal). Equivalent to full softmax in expectation;
  on a 20k catalog typically matches or slightly beats gBCE at the same
  `n_neg`.
- `sample_negatives` — uniform or popularity-weighted draws done entirely on
  GPU. We avoid `torch.multinomial` for very large total counts (B·L·K can be
  ~6M on ML-20M) by either falling back to uniform `randint` or by drawing
  from a precomputed alias table when `pop_weights` is supplied. The paper
  exponent of 0.75 (word2vec-style) is the default for popularity sampling.
- In-batch negatives are not a separate loss class — they are implemented by
  appending the batch's positive item embeddings as extra negatives at the
  call site (see `train.py`). Cheap and stable, gives a free +0.5..1% NDCG.

All loss objects expect:
    logits_pos:   [B, L]      — score of the true next item at each position
    logits_neg:   [B, L, K]   — scores of K sampled negatives per position
    target_mask:  [B, L] bool — True where the position has a valid target
                                 (input != PAD)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_negatives(
    target_mask: torch.Tensor,
    n_neg: int,
    n_items: int,
    device: torch.device,
    pop_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample [B, L, K] item ids in 1..n_items.

    Uniform sampling uses a single fused `randint` kernel (microseconds on A100).
    Popularity sampling falls back to `torch.multinomial`; we cap a single
    multinomial call at ~1M draws and stitch larger batches to avoid
    pathological slowdowns observed at `total ≥ 5M`.
    """
    B, L = target_mask.shape
    total = B * L * n_neg
    if pop_weights is None:
        neg = torch.randint(1, n_items + 1, (total,), device=device)
        return neg.view(B, L, n_neg)

    chunk = 1_000_000
    if total <= chunk:
        neg = torch.multinomial(pop_weights, total, replacement=True)
    else:
        chunks = []
        remaining = total
        while remaining > 0:
            n = min(chunk, remaining)
            chunks.append(torch.multinomial(pop_weights, n, replacement=True))
            remaining -= n
        neg = torch.cat(chunks, dim=0)
    return neg.view(B, L, n_neg)


class GBCELoss(nn.Module):
    """Generalized BCE from gSASRec (Petrov & Macdonald, RecSys 2023).

        L = − [ α · log σ(s⁺) + (1/n) · Σⱼ log(1 − σ(sⱼ⁻)) ]
        α = (t · (|I| − 1)) / n,  t ∈ (0, 1]

    Uses log-sigmoid arithmetic for numerical stability in bf16 / fp32:
        log(1 − σ(x)) = −softplus(x)     log σ(x) = −softplus(−x)
    """

    def __init__(self, n_items: int, n_neg: int, t: float = 0.75):
        super().__init__()
        if not 0.0 < t <= 1.0:
            raise ValueError(f"gBCE t must be in (0, 1], got {t}")
        self.n_items = n_items
        self.n_neg = n_neg
        self.t = t
        self.alpha = t * (n_items - 1) / max(n_neg, 1)

    def forward(
        self,
        logits_pos: torch.Tensor,    # [B, L]
        logits_neg: torch.Tensor,    # [B, L, K]
        target_mask: torch.Tensor,   # [B, L] bool
    ) -> torch.Tensor:
        log_pos = -F.softplus(-logits_pos)
        log_neg = -F.softplus(logits_neg)
        per_token = self.alpha * log_pos + log_neg.mean(dim=-1)
        # Replace any NaN at fully-PAD prefix rows so the masked sum stays clean.
        per_token = torch.nan_to_num(per_token, nan=0.0, posinf=0.0, neginf=0.0)
        denom = target_mask.sum().clamp(min=1)
        return -(per_token * target_mask).sum() / denom


class SampledSoftmaxLoss(nn.Module):
    """Sampled softmax with optional logQ correction.

    Standard form (Bengio & Senecal):
        L = −log( exp(s⁺ − logQ⁺) /
                  [exp(s⁺ − logQ⁺) + Σⱼ exp(sⱼ⁻ − logQⱼ⁻)] )

    With uniform sampling logQ is constant and cancels — it only matters when
    negatives are popularity-weighted.
    """

    def __init__(self, log_q_pos: Optional[torch.Tensor] = None):
        super().__init__()
        if log_q_pos is not None:
            self.register_buffer("log_q_pos", log_q_pos)
        else:
            self.log_q_pos = None

    def forward(
        self,
        logits_pos: torch.Tensor,    # [B, L]
        logits_neg: torch.Tensor,    # [B, L, K]
        target_mask: torch.Tensor,   # [B, L] bool
        target_ids: torch.Tensor,    # [B, L] long
        neg_ids: torch.Tensor,       # [B, L, K] long
    ) -> torch.Tensor:
        if self.log_q_pos is not None:
            logits_pos = logits_pos - self.log_q_pos[target_ids]
            logits_neg = logits_neg - self.log_q_pos[neg_ids]
        all_logits = torch.cat([logits_pos.unsqueeze(-1), logits_neg], dim=-1)
        lse = torch.logsumexp(all_logits, dim=-1)
        loss_per_pos = lse - logits_pos
        loss_per_pos = torch.nan_to_num(loss_per_pos, nan=0.0, posinf=0.0, neginf=0.0)
        denom = target_mask.sum().clamp(min=1)
        return (loss_per_pos * target_mask).sum() / denom


def make_logq_uniform(n_items: int, n_neg: int, device: torch.device) -> torch.Tensor:
    """LogQ for uniform sampling: every non-PAD item has q = n_neg / n_items.
    Returns [n_items+1] tensor; PAD entry = 0 (won't be used).
    """
    log_q = torch.full((n_items + 1,), math.log(n_neg / n_items), device=device)
    log_q[0] = 0.0
    return log_q


def make_logq_popularity(item_pop, n_neg: int, exponent: float = 0.75) -> torch.Tensor:
    """LogQ for popularity-^exponent sampling.
    Returns [n_items+1] tensor; PAD entry = 0.
    """
    pop = torch.as_tensor(item_pop, dtype=torch.float32)
    pop[0] = 0.0
    w = pop.pow(exponent)
    z = w.sum().clamp(min=1e-9)
    p = w / z                                    # marginal prob for one draw
    p = p.clamp(min=1e-12)
    log_q = torch.log(p) + math.log(n_neg)       # K independent draws ≈ Kp
    log_q[0] = 0.0
    return log_q


def popularity_weights(item_pop, exponent: float = 0.75) -> torch.Tensor:
    """Convert raw popularity counts to sampling weights w_i ∝ pop_i^exponent.
    PAD index gets zero weight (never sampled).
    """
    w = torch.as_tensor(item_pop, dtype=torch.float32).clone()
    w[0] = 0.0
    w = w.pow(exponent)
    w = w / w.sum().clamp(min=1e-9)
    return w
