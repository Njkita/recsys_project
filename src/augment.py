"""Sequence augmentations.

Implemented:
- `sse_pt_augment` — Stochastic Shared Embeddings (Wu et al., RecSys 2020).
  With probability `p` replace a non-PAD item id with a uniform random item id.
  Acts as a strong embedding-level regulariser, +2..3% NDCG on ML-20M for
  three lines of code. Default in our SASRec config.
- `cl4srec_mask` — random item masking (BERT-style) for contrastive variants.
  Not used by the main loss but available for CL4SRec / DuoRec experiments.
- `cl4srec_crop` — random contiguous crop of the input sequence.
- `cl4srec_reorder` — local reorder of a contiguous slice.

For the production runs we only enable SSE-PT — the paper-proven contrastive
augmentations need a contrastive head that is out of scope for the v1 ML-20M
sweep. They live here as building blocks for follow-up ablations.
"""
from __future__ import annotations

import torch


def sse_pt_augment(
    input_ids: torch.Tensor,
    n_items: int,
    p: float = 0.1,
) -> torch.Tensor:
    """Replace each non-PAD position with a random item id with probability p."""
    if p <= 0:
        return input_ids
    pad_mask = input_ids == 0
    corrupt = (torch.rand_like(input_ids, dtype=torch.float32) < p) & ~pad_mask
    if not corrupt.any():
        return input_ids
    rnd = torch.randint(1, n_items + 1, input_ids.shape, device=input_ids.device)
    return torch.where(corrupt, rnd, input_ids)


def cl4srec_mask(input_ids: torch.Tensor, p: float = 0.2, mask_token: int = 0) -> torch.Tensor:
    """BERT-style random masking. `mask_token` defaults to PAD (0)."""
    if p <= 0:
        return input_ids
    pad_mask = input_ids == 0
    corrupt = (torch.rand_like(input_ids, dtype=torch.float32) < p) & ~pad_mask
    return torch.where(corrupt, torch.full_like(input_ids, mask_token), input_ids)


def cl4srec_crop(input_ids: torch.Tensor, eta: float = 0.6) -> torch.Tensor:
    """Random contiguous crop, keep `eta` fraction of non-PAD tokens.
    The remaining left side is replaced by PAD to preserve `max_len`.
    """
    B, L = input_ids.shape
    out = input_ids.clone()
    for b in range(B):
        nz = (input_ids[b] != 0).nonzero(as_tuple=True)[0]
        if len(nz) < 2:
            continue
        start, end = nz[0].item(), nz[-1].item() + 1
        keep = max(1, int((end - start) * eta))
        cs = torch.randint(start, end - keep + 1, (1,)).item()
        ce = cs + keep
        new = torch.zeros_like(input_ids[b])
        new[L - keep:] = input_ids[b, cs:ce]
        out[b] = new
    return out


def cl4srec_reorder(input_ids: torch.Tensor, beta: float = 0.6) -> torch.Tensor:
    """Reorder a random contiguous span of length `beta * effective_len`."""
    B, L = input_ids.shape
    out = input_ids.clone()
    for b in range(B):
        nz = (input_ids[b] != 0).nonzero(as_tuple=True)[0]
        if len(nz) < 3:
            continue
        start, end = nz[0].item(), nz[-1].item() + 1
        span = max(2, int((end - start) * beta))
        cs = torch.randint(start, end - span + 1, (1,)).item()
        idx = torch.randperm(span)
        out[b, cs:cs + span] = input_ids[b, cs:cs + span][idx]
    return out
