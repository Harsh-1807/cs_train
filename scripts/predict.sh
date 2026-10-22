#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

if [ $# -lt 1 ]; then
  echo "Usage: $0 <test.xlsx> [output_dir]"
  exit 1
fi

INPUT="$1"
OUTPUT="${2:-outputs/predictions}"

python -m src.predict \
  --input "$INPUT" \
  --artifacts artifacts/ \
  --config config.yaml \
  --output "$OUTPUT"

echo "Predictions saved to $OUTPUT"
