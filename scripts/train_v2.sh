#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

INPUT="${1:-../lexus-large.xlsx}"

python -m src.train_v2 \
  --input "$INPUT" \
  --config config.v2.yaml \
  --artifacts artifacts/ \
  "${@:2}"

echo "v2 training complete. Artifacts in artifacts/"
