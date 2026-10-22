#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

INPUT="${1:-../lexus-large.xlsx}"
OUTPUT="${2:-outputs/eda}"

python -m src.data.eda --input "$INPUT" --output "$OUTPUT" --config config.yaml
echo "EDA plots saved to $OUTPUT"
