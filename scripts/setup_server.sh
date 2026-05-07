#!/usr/bin/env bash
# One-shot bootstrap on the GPU server.
#
# Assumes: torch 2.5.1+cu121 already installed globally (it is on x32-gpu-01).
# We create a venv with --system-site-packages so torch is reused without a
# 3 GB reinstall, and install the rest of our deps into the venv.
#
# After this script runs you can:
#   bash scripts/download_data.sh
#   bash scripts/preprocess.sh
#   bash scripts/train.sh configs/sasrec.yaml runs/sasrec
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ ! -d .venv ]; then
  echo "[setup] creating venv with --system-site-packages"
  python3 -m venv --system-site-packages .venv
fi
source .venv/bin/activate

echo "[setup] upgrading pip"
pip install --upgrade pip

echo "[setup] installing project requirements"
# torch is reused from system; numpy/pandas/etc are installed locally
pip install -r requirements.txt

echo "[setup] verifying torch + CUDA"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible'; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0))"

echo "[setup] verifying project imports"
python -c "from src.models import build_model; from src.data import preprocess_ml20m; from src.eval import evaluate_full_catalog; from src.train import run_training; print('ok')"

echo "[setup] done. Mamba4Rec is optional; install via scripts/install_mamba.sh"
