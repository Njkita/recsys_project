#!/usr/bin/env bash
# Train one model: bash scripts/train.sh configs/sasrec.yaml runs/sasrec_modern
set -euo pipefail
CONFIG="${1:?config path required}"
OUT="${2:?out dir required}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source .venv/bin/activate

mkdir -p "$OUT"
echo "[train] config=$CONFIG out=$OUT"
python -m src.train --config "$CONFIG" --out "$OUT" --data-dir data 2>&1 | tee -a "$OUT/stdout.log"
