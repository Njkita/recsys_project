"""Latency / throughput / VRAM benchmark across model variants.

Measures forward-only latency at four sequence lengths:
- L=200  — paper standard for ML-20M
- L=500  — long-tail user histories
- L=1000 — synthetic, where SASRec attention starts to bottleneck
- L=2000 — synthetic, where linear-attention / FMLP / Mamba pull ahead

Inputs are random integer sequences (no real data needed for kernel timing).
We run a warmup, then median of 30 timed iterations with `cuda.synchronize`.
VRAM is captured via `torch.cuda.max_memory_allocated`.

Output: CSV with columns (model, L, batch, d, n_blocks, latency_ms,
throughput_seq_per_s, vram_mb, params_total). Append mode supported so the
script can be called multiple times for optional dependencies.

Run: `python -m src.benchmark --configs configs/sasrec.yaml ... --lengths 200 500 1000 2000 --out runs/benchmark.csv`
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

from .models import build_model
from .utils import count_params, enable_tf32


def _measure(
    model: torch.nn.Module,
    L: int,
    batch: int,
    n_items: int,
    device: torch.device,
    dtype: torch.dtype,
    iters: int = 30,
) -> Dict[str, float]:
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.randint(1, n_items + 1, (batch, L), device=device, dtype=torch.long)

    autocast_ctx = torch.autocast(device_type=device.type, dtype=dtype) \
        if device.type == "cuda" and dtype != torch.float32 else torch.cuda.amp.autocast(enabled=False)

    with torch.no_grad():
        for _ in range(3):
            with autocast_ctx:
                _ = model.score_all(x)
        torch.cuda.synchronize()
        ts: List[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            with autocast_ctx:
                _ = model.score_all(x)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)

    median_ms = statistics.median(ts) * 1000.0
    p95_ms = sorted(ts)[int(0.95 * len(ts))] * 1000.0
    vram_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return {
        "latency_ms_median": round(median_ms, 3),
        "latency_ms_p95": round(p95_ms, 3),
        "throughput_seq_per_s": round(batch / (median_ms / 1000.0), 1),
        "vram_mb": round(vram_mb, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--lengths", nargs="+", type=int, default=[200, 500, 1000, 2000])
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--n-items", type=int, default=20000,
                   help="dummy vocabulary size for synthetic input")
    p.add_argument("--out", required=True)
    p.add_argument("--append", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA required for latency benchmark.")
    enable_tf32()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "model", "config", "L", "batch", "d", "n_blocks",
        "latency_ms_median", "latency_ms_p95", "throughput_seq_per_s",
        "vram_mb", "params_total", "dtype",
    ]
    mode = "a" if (args.append and out_path.exists()) else "w"
    with out_path.open(mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if mode == "w":
            w.writeheader()

        for cfg_path in args.configs:
            cfg = yaml.safe_load(Path(cfg_path).read_text())
            mcfg = cfg["model"]
            for L in args.lengths:
                # We pass max_len=L so the model's positional buffers / RoPE
                # tables fit. Most configs accept max_len in their model dict.
                mcfg_eff = dict(mcfg)
                mcfg_eff["max_len"] = L
                # Side info for benchmark is off — we measure backbone speed.
                mcfg_eff["use_side"] = False
                try:
                    model = build_model(mcfg_eff, n_items=args.n_items, side_module=None).to(device)
                except Exception as e:                                # noqa: BLE001
                    print(f"[benchmark] skip {mcfg['name']} L={L}: {e}")
                    continue

                m = _measure(model, L, args.batch, args.n_items, device, dtype)
                row = {
                    "model": mcfg["name"],
                    "config": cfg_path,
                    "L": L,
                    "batch": args.batch,
                    "d": mcfg.get("d", "?"),
                    "n_blocks": mcfg.get("n_blocks", "?"),
                    "params_total": count_params(model)["total"],
                    "dtype": str(dtype),
                    **m,
                }
                w.writerow(row)
                print(f"[benchmark] {mcfg['name']:>14}  L={L:5d}  "
                      f"lat={m['latency_ms_median']:7.2f}ms  "
                      f"thr={m['throughput_seq_per_s']:>9}seq/s  "
                      f"vram={m['vram_mb']:>7}MB")
                del model
                torch.cuda.empty_cache()

    print(f"[benchmark] {out_path}")


if __name__ == "__main__":
    main()
