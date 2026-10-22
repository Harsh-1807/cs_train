#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

if [ $# -lt 1 ]; then
  echo "Usage: $0 <data.xlsx> [predictions_dir] [output_dir]"
  exit 1
fi

INPUT="$1"
PREDICTIONS="${2:-outputs/predictions}"
OUTPUT="${3:-outputs/reports}"

python -m src.evaluate \
  --input "$INPUT" \
  --predictions "$PREDICTIONS" \
  --artifacts artifacts/ \
  --config config.yaml \
  --output "$OUTPUT"

echo "Evaluation report saved to $OUTPUT"
