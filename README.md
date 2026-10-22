# AMI Communication Health & Root-Cause Intelligence

Unsupervised GNN pipeline for AMI smart-meter communication health analysis.

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for full methodology, sheet/column omit list, and dual training modes.

## Quick start (GPU machine — e.g. NVIDIA A50)

```bash
cd ami-intelligence
chmod +x scripts/*.sh
./scripts/setup.sh
./scripts/eda.sh ../lexus-large.xlsx
./scripts/train.sh ../lexus-large.xlsx              # mode=full (default)
./scripts/train.sh ../lexus-large.xlsx --mode split # feeder holdout
./scripts/evaluate.sh ../lexus-large.xlsx
```

## Predict on test data

```bash
./scripts/predict.sh /path/to/test.xlsx
./scripts/evaluate.sh /path/to/test.xlsx   # if GT columns present
```

## Training modes

| Mode | Command | Use case |
|------|---------|----------|
| `full` | `./scripts/train.sh data.xlsx --mode full` | Production model on all data |
| `split` | `./scripts/train.sh data.xlsx --mode split` | Internal validation (20% feeder holdout) |

Set default in `config.yaml` → `training.mode`.

## Outputs

- `outputs/eda/` — pre-training plots
- `outputs/predictions/consumer_predictions.csv` — per-consumer predictions
- `outputs/reports/metrics_report.json` — evaluation metrics
- `artifacts/` — saved models

## Requirements

- Python 3.10+
- CUDA GPU recommended (16GB+ VRAM for 18K nodes)
- Auto-detects CPU if no GPU
