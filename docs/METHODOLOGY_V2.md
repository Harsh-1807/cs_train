# AMI GNN v2 — SOTA Pipeline

Branch: **`ami-sota-v2`** — compare against v1 (`ami-comm-health`).

## Why v2 exists

v1 (Graph MAE + HGT-2L + HDBSCAN) is deployable but has gaps:
- No temporal modeling of DCU 90-day series
- No contrastive graph pretraining
- Root cause via density clustering + heuristics (not learned centroids)

v2 addresses all three for **defensible AIML** on POC 2 multi-level scenarios.

---

## Architecture

```
DCU_Daily_Status (90 days × 560 DTCs)
        │
        ▼
  GRU Temporal Encoder (pretrain 15 epochs)
        │
        ├─► fused into DTC graph nodes
        └─► fused into consumer tabular features

Network_Topology (full hierarchy: PP → Consumer)
        +
Asset_Comm_Health + Consumer roll-ups (all levels)
        │
        ▼
  HeteroData graph
        │
        ▼
  AMI GNN v2 (joint training, 100 epochs)
  ├── Graph MAE (masked reconstruction)
  ├── GRACE contrastive (dual view: feature mask + edge dropout)
  ├── DEC (5 learnable cluster centroids, KL loss after warmup)
  └── HGT encoder (3 layers, 4 heads, 128-dim)
        │
        ├─► RootCauseAnalyzerV2 (DEC soft assignment → 5 classes)
        └─► LightGBM silent-meter head (Jan–Feb → Mar proxy)
```

---

## POC 2 scenario coverage

| Scenario (POC README) | How v2 detects it |
|----------------------|-------------------|
| DCU / Gateway failure | DCU temporal encoder + DTC node features + HGT propagation |
| Upstream backhaul outage | Backhaul_Up in DCU sequences; substation/feeder roll-ups |
| Poor RF / coverage gap | RSSI, hop count, packet loss in consumer + network features |
| NIC removed / hardware fault | Events + NIC status + DEC cluster assignment |
| Firmware mismatch | Firmware events + post-upgrade comm drop patterns |
| Feeder tripping / supply | Feeder outage roll-ups, transient cluster |
| Tamper-linked NIC removal | Tamper event sequence before event 9 |
| Malfunction (silent data loss) | High heartbeat + low read success → anomaly score + cluster |

---

## Learning paradigm (v2)

| Component | Type |
|-----------|------|
| DCU GRU pretrain | Self-supervised (predict last-day DCU stats) |
| Graph MAE | Self-supervised |
| GRACE contrastive | Self-supervised |
| DEC clustering | Unsupervised (learned centroids) |
| Root-cause label text | Interpretation of DEC cluster centroids (no GT Defect_Type) |
| Silent meter LightGBM | Weakly supervised (Comm_Status proxy) |
| evaluate.sh | GT metrics only at evaluation time |

---

## Commands

```bash
./scripts/setup.sh
./scripts/train_v2.sh ../lexus-large.xlsx              # full mode
./scripts/train_v2.sh ../lexus-large.xlsx --mode split   # feeder holdout
./scripts/predict.sh test.xlsx                            # auto-detects v2 artifacts
./scripts/evaluate.sh ../lexus-large.xlsx
```

---

## Compare v1 vs v2

| Aspect | v1 (`ami-comm-health`) | v2 (`ami-sota-v2`) |
|--------|------------------------|---------------------|
| HGT layers | 2 | 3 |
| Training | MAE only | MAE + contrastive + DEC |
| DCU temporal | Scalar aggregates | 90-day GRU encoder |
| Clustering | HDBSCAN | DEC (5 centroids) |
| Config | `config.yaml` | `config.v2.yaml` |
| Train script | `train.sh` | `train_v2.sh` |

Run both on same data, compare `outputs/reports/metrics_report.json`.
