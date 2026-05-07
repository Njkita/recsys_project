#!/usr/bin/env bash
# Train every model variant sequentially. ~24-72h on A100-80GB total
# depending on early-stop. Skips Mamba if mamba_ssm is not installed.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source .venv/bin/activate

run() {
  local name="$1"
  local cfg="configs/${name}.yaml"
  local out="runs/${name}"
  if [ -f "$out/result.json" ]; then
    echo "[skip] $name already has result.json — delete to re-run"
    return
  fi
  echo "[train_all] >>> $name"
  bash scripts/train.sh "$cfg" "$out"
}

run sasrec_baseline
run sasrec
run nextitnet
run fmlp
run fnet_hybrid
run linear_attn
run stackrec_sasrec

if python -c "import mamba_ssm" 2>/dev/null; then
  run mamba
else
  echo "[skip] mamba — mamba_ssm not installed (run scripts/install_mamba.sh)"
fi

echo "[train_all] done. Aggregating..."
python -m src.results
echo
echo "[train_all] Diagnostic scan..."
python -m src.diagnose
