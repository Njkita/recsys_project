"""Scan `runs/<exp>/log.jsonl` files for common training pathologies and print
a diagnostic report.

Heuristics flagged:
- training loss exploded (grew >2× over a window) or went NaN
- gradient norm too large (>5× the running median)
- LR scheduler stuck at 0 (warmup misconfigured)
- early stopping fired before warmup ended
- val NDCG@10 plateaus before reaching 0.1 (likely undertrained / wrong loss)
- val NDCG@10 oscillates (no smooth improvement — indicates lr too high)
- training time per epoch grew significantly (data loader bottleneck?)
- final test NDCG@10 below 0.10 (something deeply wrong; check preprocessing
  or filter-seen logic)

Run: `python -m src.diagnose [--runs-dir runs]`
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _filter(events: List[Dict[str, Any]], tag: str) -> List[Dict[str, Any]]:
    return [e for e in events if e.get("tag") == tag]


def diagnose_run(run_dir: Path) -> Dict[str, Any]:
    log = _read_jsonl(run_dir / "log.jsonl")
    result_path = run_dir / "result.json"
    failure_path = run_dir / "FAILURE.txt"
    issues: List[str] = []
    info: Dict[str, Any] = {"run": str(run_dir)}

    # 1. Crashed runs
    if failure_path.exists():
        issues.append(f"FAILED: see {failure_path}")
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text())
            if result.get("status") == "FAILED":
                issues.append(f"FAILED status in result.json: {result.get('error')}")
            info["test_NDCG@10"] = (result.get("test_metrics") or {}).get("NDCG@10")
            info["best_epoch"] = result.get("best_epoch")
        except Exception as e:
            issues.append(f"result.json unreadable: {e}")

    if not log:
        issues.append("no log.jsonl — training never started?")
        return {"info": info, "issues": issues}

    # 2. Train step pathologies
    steps = _filter(log, "train_step")
    if steps:
        losses = [s.get("loss", float("nan")) for s in steps]
        grads = [s.get("grad_norm", 0.0) for s in steps]
        if any(l != l for l in losses):  # NaN check
            issues.append("loss went NaN at some point")
        # Loss explosion: late-epoch loss > 2 × early-epoch loss
        if len(losses) >= 4:
            early = statistics.mean(losses[:max(2, len(losses) // 8)])
            late = statistics.mean(losses[-max(2, len(losses) // 8):])
            if late > 2 * early and early > 0:
                issues.append(f"loss appears to diverge: early={early:.3f} late={late:.3f}")
        # Gradient spike check
        if grads and max(grads) > 5 * (statistics.median(grads) or 1):
            issues.append(f"max grad_norm={max(grads):.2f} >> median={statistics.median(grads):.2f}")
        # LR stuck at 0
        lrs = [s.get("lr", 0) for s in steps]
        if lrs and max(lrs) == 0:
            issues.append("lr never left 0 — scheduler/warmup misconfigured")
        info["loss_first"] = round(losses[0], 4) if losses else None
        info["loss_last"] = round(losses[-1], 4) if losses else None
        info["lr_max"] = max(lrs) if lrs else None

    # 3. Val curve pathologies
    vals = _filter(log, "val")
    if vals:
        ndcgs = [v.get("NDCG@10", 0) for v in vals]
        info["val_ndcg10_first"] = round(ndcgs[0], 4)
        info["val_ndcg10_best"] = round(max(ndcgs), 4)
        info["val_epochs"] = len(ndcgs)
        # No improvement at all
        if max(ndcgs) <= ndcgs[0]:
            issues.append(f"val NDCG@10 never improved beyond initial {ndcgs[0]:.4f}")
        # Stuck below 0.10
        if max(ndcgs) < 0.10:
            issues.append(f"val NDCG@10 max {max(ndcgs):.4f} < 0.10 — model undertrained or eval broken")
        # Oscillation: high std relative to range
        if len(ndcgs) >= 5:
            std = statistics.stdev(ndcgs[-5:])
            mean = statistics.mean(ndcgs[-5:])
            if mean > 0 and std / mean > 0.1:
                issues.append(f"val NDCG@10 oscillates in last 5 epochs (cv={std/mean:.2%}) — lr may be too high")

    # 4. Per-epoch wallclock drift
    epochs = _filter(log, "train_epoch")
    if len(epochs) >= 3:
        wcs = [e.get("wallclock_s", 0) for e in epochs]
        if min(wcs) > 0 and max(wcs) > 1.5 * min(wcs):
            issues.append(f"epoch time grew {min(wcs):.0f}s → {max(wcs):.0f}s "
                          "— possible memory leak / data loader bottleneck")
        info["epochs_run"] = len(wcs)
        info["mean_epoch_s"] = round(statistics.mean(wcs), 1)

    # 5. Early stop sanity
    es = _filter(log, "early_stop")
    if es:
        ev = es[0]
        info["early_stop"] = ev
        warmup_epochs = max(1, info.get("epochs_run", 0) // 20)
        if ev.get("epoch", 999) < warmup_epochs:
            issues.append(f"early-stopped at epoch {ev.get('epoch')} — likely premature")

    # 6. NaN events
    nan_events = _filter(log, "nan_detected")
    if nan_events:
        issues.append(f"{len(nan_events)} NaN events captured in log")

    # 7. Test metrics very low
    if info.get("test_NDCG@10") is not None and info["test_NDCG@10"] < 0.10:
        issues.append(f"test NDCG@10={info['test_NDCG@10']:.4f} < 0.10 — investigate")

    return {"info": info, "issues": issues}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs", type=str)
    args = p.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    n_total = 0
    n_clean = 0
    print(f"# diagnose — scanning {runs_dir}\n")
    for d in sorted(runs_dir.iterdir() if runs_dir.exists() else []):
        if not d.is_dir():
            continue
        # Recurse into stackrec multi-stage subdirs too
        candidates = [d] + [s for s in d.iterdir() if s.is_dir() and (s / "log.jsonl").exists()]
        for c in candidates:
            if not (c / "log.jsonl").exists() and not (c / "result.json").exists():
                continue
            n_total += 1
            rep = diagnose_run(c)
            issues = rep["issues"]
            info = rep["info"]
            if not issues:
                n_clean += 1
                print(f"## {c.name}  CLEAN")
            else:
                print(f"## {c.name}")
                for it in issues:
                    print(f"  ⚠  {it}")
            relevant = {k: v for k, v in info.items() if k != "run"}
            if relevant:
                print("  info:", json.dumps(relevant, ensure_ascii=False))
            print()
    print(f"# summary: {n_clean}/{n_total} runs clean")


if __name__ == "__main__":
    main()
