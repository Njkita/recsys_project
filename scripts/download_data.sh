#!/usr/bin/env bash
# Download MovieLens-20M into data/ml-20m/.
#
# ~190 MB zip → ~620 MB unpacked. Includes ratings.csv, movies.csv,
# genome-scores.csv (needed for the side-info module).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_DIR/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ -f ml-20m/ratings.csv ]; then
  echo "[data] ml-20m already extracted at $DATA_DIR/ml-20m, skipping download."
  exit 0
fi

if [ ! -f ml-20m.zip ]; then
  echo "[data] downloading ml-20m.zip (~190 MB)"
  curl -L -o ml-20m.zip https://files.grouplens.org/datasets/movielens/ml-20m.zip
fi

echo "[data] extracting"
unzip -q ml-20m.zip
ls ml-20m | head -10

echo "[data] done. Files in $DATA_DIR/ml-20m/"
