#!/usr/bin/env bash
# overnight_retune.sh — retune nextitnet, fmlp, fnet_hybrid with corrected configs.
# Strategy: backup current results outside runs/, try descending batch, retry on OOM.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==================== retune overnight start $(date) ===================="

# 1) wait for any currently-running training
while pgrep -f "src\.train" >/dev/null || pgrep -f "train_all\.sh" >/dev/null; do
  echo "[$(date +%H:%M)] something still running, waiting 3 min..."
  sleep 180
done
echo "[$(date +%H:%M)] GPU clear — retune begins"

# 2) safety backups OUTSIDE runs/ (survives rm -rf)
mkdir -p backup_pre_retune
for m in nextitnet fmlp fnet_hybrid; do
  if [ -f "runs/$m/result.json" ]; then
    cp "runs/$m/result.json" "backup_pre_retune/$m.result.json"
    [ -f "runs/$m/log.jsonl" ] && cp "runs/$m/log.jsonl" "backup_pre_retune/$m.log.jsonl"
    echo "[$(date +%H:%M)] backed up $m -> backup_pre_retune/"
  fi
done

# 3) retune retry loop
succeeded() {
  [ -f "runs/$1/result.json" ] && ! grep -q '"status": *"FAILED"' "runs/$1/result.json"
}

for round in $(seq 1 30); do
  all_done=1
  for m in nextitnet fmlp fnet_hybrid; do
    if [ -f "runs/$m/.retuned_ok" ]; then continue; fi
    all_done=0
    echo "[$(date +%H:%M)] round $round -- $m"
    for bs in 256 160 96 48; do
      rm -rf "runs/$m"
      sed -i "s/^  batch_size: .*/  batch_size: $bs/" "configs/$m.yaml"
      echo "[$(date +%H:%M)]   try $m @ batch_size=$bs"
      python -m src.train --config "configs/$m.yaml" --out "runs/$m" --data-dir data
      if succeeded "$m"; then
        echo "[$(date +%H:%M)]   >>> $m completed @ batch_size=$bs"
        touch "runs/$m/.retuned_ok"
        break
      fi
      echo "[$(date +%H:%M)]   $m failed @ batch_size=$bs (likely OOM) — smaller"
    done
  done
  if [ "$all_done" -eq 1 ]; then echo "[$(date +%H:%M)] ALL DONE"; break; fi
  echo "[$(date +%H:%M)] round $round incomplete — sleep 15 min"
  sleep 900
done

# 4) aggregate + side-by-side comparison
echo "[$(date +%H:%M)] aggregating + comparing to pre-retune backups"
python -m src.results || true

echo
echo "=== COMPARISON: pre-retune backup vs new (test NDCG@10) ==="
for m in nextitnet fmlp fnet_hybrid; do
  if [ -f "backup_pre_retune/$m.result.json" ] && [ -f "runs/$m/result.json" ]; then
    old=$(python -c "import json;d=json.load(open('backup_pre_retune/$m.result.json'));t=d.get('test_metrics',{});print(t.get('NDCG@10','?'))" 2>/dev/null)
    new=$(python -c "import json;d=json.load(open('runs/$m/result.json'));t=d.get('test_metrics',{});print(t.get('NDCG@10','?'))" 2>/dev/null)
    echo "  $m:  was=$old  ->  now=$new"
  fi
done

python -m src.diagnose 2>&1 | tail -50 || true
echo "==================== RETUNE DONE $(date) ===================="
echo
echo "If any model regressed, rollback with:"
echo "  cp backup_pre_retune/<model>.result.json runs/<model>/result.json"
echo "  cp backup_pre_retune/<model>.log.jsonl   runs/<model>/log.jsonl"
