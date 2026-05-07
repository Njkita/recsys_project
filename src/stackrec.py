"""StackRec — train deep sequential recommenders by iterative layer stacking.

Reference: Wang et al., SIGIR 2021, arXiv:2012.07598
            github.com/wangjiachun0426/StackRec

Idea: a trained shallow model has highly redundant representations between
adjacent layers (cosine similarity >0.9). Doubling the depth by *copying*
trained blocks gives a strong starting point — ~30-45% wall-clock saving on
the way to a deep model versus from-scratch deep training.

Two stacking modes:
- 'adjacent' / interleaved: blocks become (1, 1', 2, 2', 3, 3', ...). Better
  on transformer/CNN sequential rec per the paper (Table 4).
- 'sequential' / cross: blocks become (1, 2, ..., L, 1', 2', ..., L'). Slower
  to converge but still useful.

Usage:
    from src.stackrec import stack_blocks, stackrec_optim_config
    deeper = stack_blocks(trained_model)        # n_blocks → 2 * n_blocks
    cfg = stackrec_optim_config(stage=1)        # lr scale, warmup fraction

LiGR-gates compatible: deepcopy preserves the learned `g_attn`/`g_ffn`
parameters. Tied output projection survives because deepcopy uses a memo
dict and keeps shared-tensor identity intact within one call.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Literal

import torch
import torch.nn as nn

StackMode = Literal["adjacent", "sequential"]


@torch.no_grad()
def stack_blocks(
    model: nn.Module,
    mode: StackMode = "adjacent",
    blocks_attr: str = "blocks",
) -> nn.Module:
    """Return a new model with doubled block depth.

    The returned object is a `deepcopy` of `model` with its `blocks` ModuleList
    replaced. Item / positional embeddings, RoPE buffers, final LN, and the
    tied output projection are preserved bit-identical.
    """
    new_model = copy.deepcopy(model)
    old_blocks: nn.ModuleList = getattr(new_model, blocks_attr)
    L = len(old_blocks)

    new_list = nn.ModuleList()
    if mode == "adjacent":
        for i in range(L):
            new_list.append(old_blocks[i])
            new_list.append(copy.deepcopy(old_blocks[i]))
    elif mode == "sequential":
        twins = [copy.deepcopy(b) for b in old_blocks]
        new_list.extend(old_blocks)
        new_list.extend(twins)
    else:
        raise ValueError(f"unknown StackRec mode: {mode!r}")

    setattr(new_model, blocks_attr, new_list)
    if hasattr(new_model, "cfg") and hasattr(new_model.cfg, "n_blocks"):
        new_model.cfg.n_blocks = 2 * L

    # Sanity: tied output projection identity preserved
    if hasattr(new_model, "output_embedding") and hasattr(new_model, "item_emb"):
        out = new_model.output_embedding
        emb = new_model.item_emb.weight
        assert out.data_ptr() == emb.data_ptr(), \
            "Tied weights broken after stack_blocks — investigate."

    return new_model


def stackrec_optim_config(stage: int, base_lr: float = 1e-3) -> Dict[str, Any]:
    """Recipe from the paper §4.2 + standard transformer practice.

    stage 0 = initial training (e.g. n_blocks=4)
    stage 1 = after first stack (8 blocks)
    stage 2 = after second stack (16 blocks)
    """
    lr_mult = {0: 1.0, 1: 0.5, 2: 0.25}.get(stage, 0.125)
    warmup_frac = 0.0 if stage == 0 else 0.05
    return {
        "lr": base_lr * lr_mult,
        "warmup_frac": warmup_frac,
        "scheduler": "cosine",
    }
