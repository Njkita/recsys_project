#!/usr/bin/env bash
# Run 5-core filtering, dense id remap, and pickle the result to
# data/processed.pkl. Re-runnable: it overwrites the cache.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source .venv/bin/activate

python -c "
from pathlib import Path
from src.data import preprocess_ml20m
preprocess_ml20m(
    Path('data/ml-20m/ratings.csv'),
    Path('data/processed.pkl'),
    min_user=5, min_item=5,
)
"
echo "[preprocess] done."
