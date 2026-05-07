"""Training utilities: seed, EMA, warmup-cosine LR scheduler, param-group split,
checkpointing, snapshot ensembling, and a tiny tensorboard-free metrics logger.

Each utility is intentionally minimal — no Lightning/Accelerate dependency.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch


# ----------------------------- Reproducibility ----------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_tf32():
    """Allow TF32 matmul on Ampere+. Free ~10% speedup, no quality loss."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# ----------------------------- EMA ----------------------------------------- #
class EMA:
    """Exponential moving average of model parameters.

    Usage:
        ema = EMA(model, decay=0.999)
        # after each optimizer.step():
        ema.update(model)
        # at eval:
        ema.apply_to(model)        # swap in EMA weights
        ... evaluate ...
        ema.restore(model)         # swap back original weights
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self._backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module):
        self._backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if name in self._backup:
                p.data.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: Dict[str, torch.Tensor], device: Optional[torch.device] = None):
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k] = v.to(device or self.shadow[k].device)


# ----------------------------- LR Scheduler -------------------------------- #
class WarmupCosineLR:
    """Linear warmup → cosine decay to `min_lr_ratio · base_lr`.

    Call `.step()` after each optimizer step.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.01,
    ):
        self.opt = optimizer
        self.warmup = max(1, warmup_steps)
        self.total = max(self.warmup + 1, total_steps)
        self.min_ratio = min_lr_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._step = 0

    def step(self):
        self._step += 1
        if self._step < self.warmup:
            scale = self._step / self.warmup
        else:
            progress = (self._step - self.warmup) / max(1, self.total - self.warmup)
            progress = min(1.0, progress)
            scale = self.min_ratio + 0.5 * (1 - self.min_ratio) * (
                1 + math.cos(math.pi * progress)
            )
        for g, base in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = base * scale

    def get_lr(self) -> float:
        return self.opt.param_groups[0]["lr"]

    def state_dict(self) -> Dict[str, Any]:
        return {"step": self._step, "base_lrs": self.base_lrs}

    def load_state_dict(self, sd: Dict[str, Any]):
        self._step = sd["step"]
        self.base_lrs = sd["base_lrs"]


# ----------------------------- Param Groups -------------------------------- #
def split_decay_params(model: torch.nn.Module, weight_decay: float):
    """Group params for AdamW: no decay on biases, LayerNorm, embeddings."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or "embedding" in name.lower() or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ----------------------------- Snapshot Ensemble --------------------------- #
class SnapshotEnsemble:
    """Save the best model snapshot every K eval rounds and at end of training,
    average their predictions at test time.

    Stores state_dicts on disk (CPU) under `{out_dir}/snapshot_{i}.pt`. Memory
    cost: a few MB per snapshot for our model sizes.

    Usage:
        snap = SnapshotEnsemble(out_dir)
        ... after each "good" eval round ...
        snap.maybe_save(model, epoch)
        ... at test time ...
        snap.average_into(model)   # in-place: model weights ← mean(snapshots)
    """

    def __init__(self, out_dir: str | Path, max_snapshots: int = 3):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_snapshots = max_snapshots
        self.paths: List[Path] = []

    def save(self, model: torch.nn.Module, tag: str):
        path = self.out_dir / f"snapshot_{tag}.pt"
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
        self.paths.append(path)
        # FIFO eviction
        while len(self.paths) > self.max_snapshots:
            oldest = self.paths.pop(0)
            try:
                oldest.unlink()
            except FileNotFoundError:
                pass

    @torch.no_grad()
    def average_into(self, model: torch.nn.Module):
        if not self.paths:
            return
        device = next(model.parameters()).device
        sds = [torch.load(p, map_location="cpu") for p in self.paths]
        avg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in sds[0].items()}
        for sd in sds:
            for k, v in sd.items():
                avg[k] += v.to(torch.float32)
        for k in avg:
            avg[k] /= len(sds)
            avg[k] = avg[k].to(model.state_dict()[k].dtype)
        model.load_state_dict({k: v.to(device) for k, v in avg.items()}, strict=True)


# ----------------------------- Lightweight Logger -------------------------- #
class JSONLogger:
    """Append-only JSONL log + a final `result.json` summary.

    The aggregate results script (`src/results.py`) reads `result.json` from
    every `runs/<exp>/` to build the comparative table.
    """

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "log.jsonl"
        self._t0 = time.time()

    def log(self, **kv):
        kv["t_elapsed_s"] = round(time.time() - self._t0, 2)
        with self.log_path.open("a") as f:
            f.write(json.dumps(kv, ensure_ascii=False) + "\n")

    def write_result(self, summary: Dict[str, Any]):
        with (self.out_dir / "result.json").open("w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


# ----------------------------- Param Counter ------------------------------- #
def count_params(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
