#!/usr/bin/env bash
# overnight_causal_fft.sh — trains causal_fftconv AFTER retune + ensemble pipelines finish.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==================== causal_fft overnight start $(date) ===================="

while pgrep -f "src\.train"             >/dev/null \
   || pgrep -f "overnight_retune\.sh"   >/dev/null \
   || pgrep -f "overnight_ensemble\.sh" >/dev/null \
   || pgrep -f "overnight\.sh"          >/dev/null \
   || pgrep -f "train_all\.sh"          >/dev/null; do
  echo "[$(date +%H:%M)] earlier overnight still running, waiting 5 min..."
  sleep 300
done
echo "[$(date +%H:%M)] all prior work done — starting causal_fftconv training"

have_result() {
  [ -f "runs/$1/result.json" ] && ! grep -q '"status": *"FAILED"' "runs/$1/result.json"
}

name="causal_fftconv"
cfg="configs/${name}.yaml"
out="runs/${name}"

for round in $(seq 1 20); do
  for bs in 256 160 96 48; do
    rm -rf "$out"
    sed -i "s/^  batch_size: .*/  batch_size: $bs/" "$cfg"
    echo "[$(date +%H:%M)] training $name @ batch_size=$bs (round $round)"
    python -m src.train --config "$cfg" --out "$out" --data-dir data
    if have_result "$name"; then
      echo "[$(date +%H:%M)] >>> $name SUCCESS @ batch_size=$bs"
      break 2
    fi
    echo "[$(date +%H:%M)] $name failed @ batch_size=$bs (likely OOM) — smaller"
  done
  echo "[$(date +%H:%M)] all batch sizes failed — sleep 15 min"
  sleep 900
done

sed -i "s/^  batch_size: .*/  batch_size: 256/" "$cfg"
python -m src.results || true

echo "==================== causal_fft DONE $(date) ===================="
[ -f "runs/$name/result.json" ] && cat "runs/$name/result.json" | head -30
