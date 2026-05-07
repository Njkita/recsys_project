#!/usr/bin/env bash
# Run latency / VRAM / throughput benchmark over all model configs and
# multiple sequence lengths (200 — paper, 500, 1000, 2000 — synthetic).
# Writes runs/benchmark.csv.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source .venv/bin/activate

mkdir -p runs
python -m src.benchmark \
  --configs configs/sasrec.yaml configs/nextitnet.yaml configs/fmlp.yaml \
            configs/fnet_hybrid.yaml configs/linear_attn.yaml \
  --lengths 200 500 1000 2000 \
  --out runs/benchmark.csv

if python -c "import mamba_ssm" 2>/dev/null; then
  python -m src.benchmark \
    --configs configs/mamba.yaml \
    --lengths 200 500 1000 2000 \
    --out runs/benchmark.csv \
    --append
fi

echo "[benchmark] runs/benchmark.csv written"
