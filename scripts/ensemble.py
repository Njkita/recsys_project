"""Ensemble inference: average .score_all() logits across N trained checkpoints,
then run full-catalog leave-one-out eval with filter-seen.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import SeqEvalDataset, split_loo
from src.eval import evaluate_full_catalog, format_metrics
from src.models import build_model
from src.train import build_user_seen_split, get_processed, build_side_module


class EnsembleModel(nn.Module):
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    @torch.no_grad()
    def score_all(self, input_ids: torch.Tensor) -> torch.Tensor:
        total = None
        for m in self.models:
            s = m.score_all(input_ids)
            total = s if total is None else total + s
        return total / len(self.models)

    def eval(self):
        super().eval()
        for m in self.models:
            m.eval()
        return self


def load_checkpoint(ckpt_path: Path, n_items: int, data_dir: Path, device: str) -> nn.Module:
    print(f"[ensemble] loading {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    side = None
    if cfg["model"].get("use_side", False):
        proc_local = get_processed(data_dir)
        side = build_side_module(cfg, proc_local, data_dir, d=cfg["model"]["d"])
    model = build_model(cfg["model"], n_items=n_items, side_module=side)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


def main():
    members = [
        Path("ensemble_bases/sasrec/best.pt"),
        Path("ensemble_bases/sasrec_baseline/best.pt"),
    ]
    stackrec_ckpt = Path("ensemble_bases/stackrec_sasrec/stage2_L16/best.pt")
    if stackrec_ckpt.exists():
        members.append(stackrec_ckpt)

    missing = [p for p in members if not p.exists()]
    if missing:
        print(f"[ensemble] ERROR: missing checkpoints: {missing}", flush=True)
        sys.exit(1)

    out_dir = Path("runs/ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data").resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    proc = get_processed(data_dir)
    train_seq, val_target, test_target = split_loo(proc.user_seq)
    n_users_total = max(proc.n_users, max(proc.user_seq.keys()) + 1)

    max_len = 200
    test_ds = SeqEvalDataset(
        train_seq, test_target, max_len=max_len,
        prepend_seq={u: [v] for u, v in val_target.items()},
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=4,
        pin_memory=(device == "cuda"),
    )
    _, test_seen = build_user_seen_split(train_seq, val_target, n_users_total)

    print(f"[ensemble] loading {len(members)} checkpoints", flush=True)
    models = [load_checkpoint(p, n_items=proc.n_items, data_dir=data_dir, device=device)
              for p in members]
    ensemble = EnsembleModel(models).to(device).eval()

    print(f"[ensemble] evaluating ensemble of {len(models)} on full-catalog LOO test", flush=True)
    metrics = evaluate_full_catalog(
        ensemble, test_loader, test_seen, proc.n_items,
        ks=(5, 10, 20), device=device,
    )
    print(f"[ensemble] {format_metrics(metrics)}", flush=True)

    result = {
        "exp_name": "ensemble",
        "model": "ensemble",
        "n_blocks": "-",
        "d": "-",
        "loss": "logit-average",
        "members": [str(p) for p in members],
        "n_members": len(members),
        "best_epoch": "-",
        "test_metrics": metrics,
        "device": device,
    }
    out_path = out_dir / "result.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[ensemble] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
