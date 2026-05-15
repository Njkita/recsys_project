#!/usr/bin/env bash
# overnight_ensemble.sh — runs AFTER overnight_retune.sh finishes.
# 1. waits for retune + any other sweep
# 2. retrains sasrec + sasrec_baseline INTO ensemble_bases/ (does NOT touch runs/sasrec*)
# 3. runs scripts/ensemble.py
# 4. regenerates runs/results.md with new ensemble row
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==================== ensemble overnight start $(date) ===================="

while pgrep -f "src\.train"            >/dev/null \
   || pgrep -f "overnight_retune\.sh"  >/dev/null \
   || pgrep -f "overnight\.sh"         >/dev/null \
   || pgrep -f "train_all\.sh"         >/dev/null; do
  echo "[$(date +%H:%M)] retune/sweep still running, waiting 5 min..."
  sleep 300
done
echo "[$(date +%H:%M)] all prior work done — starting ensemble pipeline"

mkdir -p ensemble_bases

have_best() { [ -f "ensemble_bases/$1/best.pt" ]; }

train_base() {
  local name="$1"
  local cfg="configs/${name}.yaml"
  local out="ensemble_bases/${name}"
  if have_best "$name"; then
    echo "[$(date +%H:%M)] $name best.pt already present, skipping retrain"
    return 0
  fi
  for round in $(seq 1 20); do
    for bs in 256 160 96 48; do
      rm -rf "$out"
      sed -i "s/^  batch_size: .*/  batch_size: $bs/" "$cfg"
      echo "[$(date +%H:%M)] training $name @ batch_size=$bs (round $round)"
      python -m src.train --config "$cfg" --out "$out" --data-dir data
      if have_best "$name"; then
        echo "[$(date +%H:%M)] >>> $name best.pt READY @ batch=$bs"
        return 0
      fi
      echo "[$(date +%H:%M)] $name failed @ batch=$bs (likely OOM) — smaller"
    done
    echo "[$(date +%H:%M)] all batch sizes failed for $name — sleep 15 min"
    sleep 900
  done
  echo "[$(date +%H:%M)] ERROR: $name never finished"
  return 1
}

train_base sasrec_baseline
train_base sasrec

sed -i "s/^  batch_size: .*/  batch_size: 256/" configs/sasrec.yaml
sed -i "s/^  batch_size: .*/  batch_size: 128/" configs/sasrec_baseline.yaml

if have_best sasrec && have_best sasrec_baseline; then
  echo "[$(date +%H:%M)] running ensemble inference"
  python scripts/ensemble.py
else
  echo "[$(date +%H:%M)] ERROR: missing ensemble bases, skipping ensemble.py"
fi

python -m src.results || true

echo "==================== ENSEMBLE DONE $(date) ===================="
[ -f runs/ensemble/result.json ] && cat runs/ensemble/result.json | head -30
