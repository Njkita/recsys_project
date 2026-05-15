#!/usr/bin/env bash
# overnight.sh — finish nextitnet + fmlp (+ fnet_hybrid if it failed) overnight.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==================== overnight start $(date) ===================="

succeeded() {
  [ -f "runs/$1/result.json" ] && ! grep -q '"status": *"FAILED"' "runs/$1/result.json"
}

# 1) wait for the currently-running sweep to finish
while pgrep -f "src\.train" >/dev/null || pgrep -f "train_all\.sh" >/dev/null; do
  echo "[$(date +%H:%M)] current sweep still running, waiting 3 min..."
  sleep 180
done
echo "[$(date +%H:%M)] GPU clear of our jobs — starting retry loop"

# 2) retry loop: up to 40 rounds, 15 min apart (~10h patience)
for round in $(seq 1 40); do
  all_done=1
  for m in nextitnet fmlp fnet_hybrid; do
    succeeded "$m" && continue
    all_done=0
    echo "[$(date +%H:%M)] round $round -- $m"
    for bs in 256 160 96 48; do
      rm -rf "runs/$m"
      sed -i "s/^  batch_size: .*/  batch_size: $bs/" "configs/$m.yaml"
      echo "[$(date +%H:%M)]   try $m @ batch_size=$bs"
      python -m src.train --config "configs/$m.yaml" --out "runs/$m" --data-dir data
      if succeeded "$m"; then
        echo "[$(date +%H:%M)]   >>> $m SUCCESS @ batch_size=$bs"
        break
      fi
      echo "[$(date +%H:%M)]   $m failed @ batch_size=$bs (likely OOM) — trying smaller"
    done
  done
  if [ "$all_done" -eq 1 ]; then
    echo "[$(date +%H:%M)] ALL MODELS DONE"
    break
  fi
  echo "[$(date +%H:%M)] round $round incomplete — sleep 15 min for neighbours to free GPU"
  sleep 900
done

echo "[$(date +%H:%M)] aggregating results..."
python -m src.results || true
python -m src.diagnose || true
echo "==================== overnight DONE $(date) ===================="
