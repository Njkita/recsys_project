"""Generate defense-ready plots from runs/ artifacts."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "plots"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"

COLORS = {
    "ensemble": "#FFD700", "sasrec": "#1f77b4", "causal_fftconv": "#2ca02c",
    "stackrec_sasrec": "#9467bd", "stage2_L16": "#9467bd", "stage1_L8": "#c4a5d8",
    "stage0_L4": "#dcc6e8", "sasrec_baseline": "#ff7f0e", "linear_attn": "#17becf",
    "fmlp": "#8c564b", "fnet_hybrid": "#e377c2", "nextitnet": "#7f7f7f",
}
RUS = {
    "ensemble": "Ансамбль (sasrec + baseline)",
    "sasrec": "SASRec flagship",
    "causal_fftconv": "Causal FFT-Conv (наша)",
    "stackrec_sasrec": "StackRec (поэтапная)",
    "stage2_L16": "StackRec stage2_L16",
    "stage1_L8": "StackRec stage1_L8",
    "stage0_L4": "StackRec stage0_L4",
    "sasrec_baseline": "SASRec baseline",
    "linear_attn": "Linear Attention",
    "fmlp": "FMLP-Rec",
    "fnet_hybrid": "FNet Hybrid",
    "nextitnet": "NextItNet",
}
def color_for(n): return COLORS.get(n, "#666666")
def rus(n): return RUS.get(n, n)

def collect_curves(metric_key="NDCG@10"):
    curves = {}
    for log_path in list(sorted(RUNS.glob("*/log.jsonl"))) + list(sorted(RUNS.glob("stackrec_sasrec/*/log.jsonl"))):
        name = log_path.parent.name
        epochs, vals = [], []
        try:
            for line in open(log_path):
                d = json.loads(line)
                if d.get("tag") == "val" and metric_key in d:
                    epochs.append(d["epoch"]); vals.append(d[metric_key])
        except Exception: continue
        if epochs and len(epochs) >= 2:
            curves[name] = (epochs, vals)
    return curves

def plot_learning_curves():
    curves = collect_curves("NDCG@10")
    if not curves: print("[plot] no curves"); return
    ordered = sorted(curves.items(), key=lambda kv: max(kv[1][1]), reverse=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, (eps, vs) in ordered:
        ax.plot(eps, vs, label=rus(name), color=color_for(name), linewidth=2.2,
                alpha=0.92, marker="o", markersize=3, markevery=max(1, len(eps)//20))
    ax.axhline(0.187, color="red", linestyle="--", linewidth=1.5, alpha=0.6,
               label="Anchor V1adls1aV = 0.187")
    ax.set_xlabel("Эпоха", fontsize=12)
    ax.set_ylabel("NDCG@10 (валидационная выборка)", fontsize=12)
    ax.set_title("Кривые обучения — NDCG@10 по эпохам", fontsize=14, pad=12)
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(OUT / "learning_curves.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print(f"[plot] learning_curves.png — {len(ordered)} models")

def plot_loss_curves():
    series = {}
    for log_path in sorted(RUNS.glob("*/log.jsonl")):
        name = log_path.parent.name
        steps, losses = [], []
        try:
            for line in open(log_path):
                d = json.loads(line)
                if d.get("tag") == "step" and "loss" in d:
                    steps.append(d.get("step", len(steps))); losses.append(d["loss"])
        except Exception: continue
        if steps and len(steps) >= 10: series[name] = (steps, losses)
    if not series: print("[plot] no loss"); return
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, (steps, losses) in series.items():
        ax.plot(steps, losses, label=rus(name), color=color_for(name), linewidth=1.6, alpha=0.85)
    ax.set_xlabel("Шаг обучения", fontsize=12)
    ax.set_ylabel("Функция потерь (sampled softmax)", fontsize=12)
    ax.set_title("Кривые функции потерь по шагам", fontsize=14, pad=12)
    ax.set_yscale("log"); ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "learning_curves_loss.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print(f"[plot] learning_curves_loss.png — {len(series)} models")

def collect_final_results():
    results = {}
    for r_path in sorted(RUNS.rglob("result.json")):
        name = r_path.parent.name
        if r_path == RUNS / "stackrec_sasrec" / "result.json": continue
        try:
            d = json.load(open(r_path))
            tm = d.get("test_metrics", {}) or {}
            ndcg = tm.get("NDCG@10")
            if ndcg is None: continue
            results[name] = {"NDCG@10": ndcg, "HR@10": tm.get("HR@10"),
                             "MRR@20": tm.get("MRR@20"),
                             "params": (d.get("params") or {}).get("total")}
        except Exception as e: print(f"  skip {r_path}: {e}")
    return results

def plot_final_results():
    results = collect_final_results()
    if not results: print("[plot] no results"); return
    items = sorted(results.items(), key=lambda kv: -kv[1]["NDCG@10"])
    names = [n for n, _ in items]; vals = [r["NDCG@10"] for _, r in items]
    fig, ax = plt.subplots(figsize=(11, max(5, 0.55 * len(items))))
    ax.barh([rus(n) for n in names], vals, color=[color_for(n) for n in names],
            edgecolor="black", linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=10)
    ax.axvline(0.187, color="red", linestyle="--", linewidth=1.4, alpha=0.7,
               label="Anchor V1adls1aV = 0.187")
    ax.axvline(0.1902, color="orange", linestyle=":", linewidth=1.4, alpha=0.7,
               label="Наш SASRec baseline = 0.1902")
    ax.invert_yaxis(); ax.set_xlabel("test NDCG@10", fontsize=12)
    ax.set_title("Финальные результаты на MovieLens-20M", fontsize=14, pad=12)
    ax.legend(loc="lower right", fontsize=10); ax.set_xlim(0, max(vals) * 1.15)
    ax.grid(True, alpha=0.25, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "final_results.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print(f"[plot] final_results.png")

def plot_relative_to_baseline():
    results = collect_final_results()
    if "sasrec_baseline" not in results: print("[plot] no baseline"); return
    base = results["sasrec_baseline"]["NDCG@10"]
    deltas = {n: 100.0 * (r["NDCG@10"] - base) / base for n, r in results.items() if n != "sasrec_baseline"}
    items = sorted(deltas.items(), key=lambda kv: -kv[1])
    names = [n for n, _ in items]; vals = [d for _, d in items]
    fig, ax = plt.subplots(figsize=(11, max(5, 0.55 * len(items))))
    ax.barh([rus(n) for n in names], vals,
            color=["#2ca02c" if v >= 0 else "#d62728" for v in vals],
            edgecolor="black", linewidth=0.5)
    for i, v in enumerate(vals):
        off = max(abs(min(vals)), max(vals)) * 0.02
        x = v + (off if v >= 0 else -off)
        ax.text(x, i, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=10, fontweight="bold")
    ax.axvline(0, color="black", linewidth=1.0); ax.invert_yaxis()
    ax.set_xlabel(f"% от SASRec-baseline (NDCG@10 = {base:.4f})", fontsize=12)
    ax.set_title("Относительный прирост NDCG@10 vs SASRec-baseline", fontsize=14, pad=12)
    ax.grid(True, alpha=0.25, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "relative_to_baseline.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print(f"[plot] relative_to_baseline.png")

def plot_table():
    results = collect_final_results()
    if not results: return
    items = sorted(results.items(), key=lambda kv: -kv[1]["NDCG@10"])
    base = results.get("sasrec_baseline", {}).get("NDCG@10", 0.1902)
    rows = []
    for name, r in items:
        ndcg = r["NDCG@10"]; hr = r.get("HR@10") or 0.0; mrr = r.get("MRR@20") or 0.0
        params = r.get("params")
        params_str = f"{params/1e6:.2f}M" if isinstance(params, (int, float)) and params > 0 else "—"
        delta = 100.0 * (ndcg - base) / base if name != "sasrec_baseline" else 0.0
        delta_str = "—" if name == "sasrec_baseline" else f"{delta:+.1f}%"
        rows.append([rus(name), f"{ndcg:.4f}", f"{hr:.4f}", f"{mrr:.4f}", delta_str, params_str])
    headers = ["Модель", "NDCG@10", "HR@10", "MRR@20", "vs baseline", "Параметры"]
    fig, ax = plt.subplots(figsize=(12, 0.5 * (len(rows) + 1) + 0.5))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 1.5)
    for i in range(len(headers)):
        cell = table[(0, i)]; cell.set_facecolor("#404040")
        cell.set_text_props(color="white", fontweight="bold")
    for i, (name, _) in enumerate(items, start=1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i == 1: cell.set_facecolor("#fff8dc")
            elif name == "sasrec_baseline": cell.set_facecolor("#ffe5cc")
            elif "causal_fftconv" in name: cell.set_facecolor("#d4f4d4")
    ax.set_title("Финальная таблица результатов — MovieLens-20M", fontsize=14, pad=10)
    fig.tight_layout(); fig.savefig(OUT / "final_table.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print(f"[plot] final_table.png")

if __name__ == "__main__":
    plot_learning_curves(); plot_loss_curves()
    plot_final_results(); plot_relative_to_baseline(); plot_table()
    print(f"\nAll plots saved to: {OUT}/")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
