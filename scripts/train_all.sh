#!/usr/bin/env bash
# Sweep всех моделей. baseline первым — валидирует anchor 0.187.
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source .venv/bin/activate

run() {
  local name="$1"
  local cfg="configs/${name}.yaml"
  local out="runs/${name}"
  if [ -f "$out/result.json" ]; then
    echo "[skip] $name already has result.json"
    return 0
  fi
  echo "[train_all] >>> $name  ($(date '+%F %T'))"
  PYTHONUNBUFFERED=1 bash scripts/train.sh "$cfg" "$out"
  local rc=$?
  [ $rc -ne 0 ] && echo "[train_all] !!! $name failed rc=$rc — продолжаем" >&2
  return 0
}

run sasrec_baseline    # 1й — валидация anchor (~30 мин, expect NDCG@10≈0.18)
run sasrec             # 2й — flagship с фикснутым warmup_frac=0.01
run nextitnet
run fmlp
run fnet_hybrid
run linear_attn
run stackrec_sasrec

if python -c "import mamba_ssm" 2>/dev/null; then run mamba; fi

echo "[train_all] aggregating..."
python -m src.results || true
python -m src.diagnose || true
