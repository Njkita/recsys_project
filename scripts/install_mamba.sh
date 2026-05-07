#!/usr/bin/env bash
# Install mamba-ssm + causal-conv1d via pre-built wheels for torch 2.5 + cu12.
#
# Wheels are published on the GitHub releases of state-spaces/mamba and
# Dao-AILab/causal-conv1d. Direct pip install would attempt a from-source
# build that fails without nvcc on x32-gpu-01. The wheels avoid that.
#
# We pick the cp310 wheel by default (Ubuntu 22.04 system Python is 3.10).
# If your venv uses a different Python, override PYVER below.
set -euo pipefail
PYVER="${PYVER:-cp310}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source .venv/bin/activate

CCONV_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu12torch2.5cxx11abiFALSE-${PYVER}-${PYVER}-linux_x86_64.whl"
MAMBA_URL="https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-${PYVER}-${PYVER}-linux_x86_64.whl"

echo "[mamba] installing causal-conv1d (pre-built wheel)"
pip install "$CCONV_URL"

echo "[mamba] installing mamba-ssm 2.2.4 (pre-built wheel, --no-build-isolation)"
pip install "$MAMBA_URL" --no-build-isolation

echo "[mamba] verifying"
python -c "from mamba_ssm import Mamba; import torch; m = Mamba(d_model=64).cuda(); x = torch.randn(2, 16, 64, dtype=torch.bfloat16).cuda(); y = m(x); print('Mamba OK:', y.shape, y.dtype)"
