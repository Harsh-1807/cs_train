#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

INPUT="${1:-../lexus-large.xlsx}"

python -m src.train \
  --input "$INPUT" \
  --config config.yaml \
  --artifacts artifacts/ \
  "${@:2}"

echo "Training complete. Artifacts in artifacts/"
