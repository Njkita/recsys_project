"""Model registry — single source of truth that maps a config string
("model.name") to (config_class, model_class). Importing a model class is
done lazily so that optional dependencies (mamba-ssm) do not break the
registry import.

Usage from train.py:
    from src.models import build_model
    model = build_model(cfg_dict, n_items=N, side_module=side)
"""
from __future__ import annotations

import importlib
from dataclasses import asdict, fields
from typing import Any, Dict, Optional

import torch.nn as nn


_REGISTRY = {
    "sasrec":       (".sasrec",        "SASRecConfig",      "SASRec"),
    "nextitnet":    (".nextitnet",     "NextItNetConfig",   "NextItNet"),
    "fmlp":         (".fmlp",          "FMLPConfig",        "FMLPRec"),
    "linear_attn":  (".linear_attn",   "LinearAttnConfig",  "LinearAttnSASRec"),
    "fnet_hybrid":  (".fnet",          "FNetHybridConfig",  "FNetHybridSASRec"),
    "mamba4rec":    (".mamba4rec",     "Mamba4RecConfig",   "Mamba4Rec"),
}


def _filter_for_dataclass(cfg_cls: Any, kw: Dict[str, Any]) -> Dict[str, Any]:
    valid = {f.name for f in fields(cfg_cls)}
    return {k: v for k, v in kw.items() if k in valid}


def build_model(
    model_cfg: Dict[str, Any],
    n_items: int,
    side_module: Optional[nn.Module] = None,
) -> nn.Module:
    """`model_cfg` must contain key `name` plus the dataclass fields."""
    name = model_cfg["name"]
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'; known: {sorted(_REGISTRY)}")
    mod_path, cfg_cls_name, model_cls_name = _REGISTRY[name]
    mod = importlib.import_module(mod_path, package=__package__)
    cfg_cls = getattr(mod, cfg_cls_name)
    model_cls = getattr(mod, model_cls_name)

    kw = dict(model_cfg)
    kw.pop("name", None)
    kw["n_items"] = n_items
    kw = _filter_for_dataclass(cfg_cls, kw)
    cfg = cfg_cls(**kw)
    return model_cls(cfg, side_module=side_module)


__all__ = ["build_model"]
