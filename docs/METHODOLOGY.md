# AMI Communication Health & Root-Cause Intelligence

## Problem statement

Build an **unsupervised** pipeline on AMI smart-meter data to:

1. **Classify NIC removal root cause** into 5 classes: Tamper, Network coverage, Hardware, Firmware, Transient
2. **Predict next-month silent meters** (risk score + temporal validation Jan–Feb → Mar)
3. **Attribute missed reads and estimated bills** (RNA) to root-cause categories

Training data: `lexus-large.xlsx` (POC 2 — AMI Communication Health, 18,100 consumers, Jan–Mar 2026).

---

## POC objective alignment (multi-level scenarios)

POC 2 defects occur at **every hierarchy level** — consumer meter, DTC/DCU, LT line, feeder meter, PT, substation backhaul. The graph encodes all of these:

| Level | Node type in graph | Observable features used |
|-------|-------------------|--------------------------|
| Consumer | `consumer` | AMI_Comm_Health, events, monthly readings |
| DTC / DCU | `dtc` | Asset_Comm_Health (DTC meter/DCU), DCU_Daily_Status (90-day), child meter aggregates |
| Feeder | `feeder` | Asset_Comm_Health (feeder meter), child consumer roll-ups, outage minutes |
| LT line | `lt_line` | Child consumer comm aggregates |
| Substation / PT | `substation`, `transformer` | Asset_Comm_Health, topology edges |
| Full hierarchy | `power_plant` → `consumer` | Network_Topology edges (20,454) |

HGT message passing propagates feeder/DTC failures to downstream consumer embeddings — so a **DCU failure affecting all RF meters on one DTC** or a **feeder trip** is visible in the learned graph representation without using GT labels.

---

## Learning paradigm (honest terminology)

| Stage | Paradigm | What it uses |
|-------|----------|--------------|
| Graph MAE + HGT | **Self-supervised** | Masked feature reconstruction; no labels |
| Isolation Forest + HDBSCAN | **Unsupervised** | Embedding density / anomaly scoring |
| Root-cause class names | **Unsupervised cluster + heuristic mapping** | Cluster centroids mapped to 5 classes via observable signatures (RSSI, events, NIC status) — **not** GT Defect_Type |
| Silent meter LightGBM | **Weakly supervised (proxy label)** | Target = observable `Comm_Status` (Non-Communicating/Intermittent), **not** GT Defect_Type |
| evaluate.sh metrics | **Supervised evaluation only** | Compares predictions vs GT columns that were **never** seen during training |

**We do NOT claim pure unsupervised end-to-end classification.** Root causes are discovered by clustering, then interpreted. Silent-meter prediction uses an operational proxy label available to any utility.

### Is this SOTA?

**Strong parts (aligned with 2024–2025 graph ML practice):**
- Heterogeneous Graph Transformer (HGT) for multi-relational AMI hierarchy
- Graph MAE self-supervised pretraining for label-scarce settings
- Feeder-level inductive split (avoids DCU leakage)

**Gaps vs absolute SOTA (future upgrades):**
- No temporal GNN (TGN/DySAT) on DCU 90-day series — currently aggregated to scalars
- Root-cause labels are heuristic, not a learned classifier (alternative: Deep Graph Clustering / BGRl + LLM-free signature validation)
- No explicit contrastive loss between normal/defective subgraph pairs

**Best alternative if more time/data:** Temporal HGT + Graph contrastive learning pretrain + multi-task heads (silent meter + anomaly) with observable proxies only — still no GT during training.

---

## SOTA GNN architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Observable features (no GT labels)                             │
│  AMI_Comm_Health + Events + DCU daily + Monthly readings        │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Heterogeneous Graph (PyG HeteroData)                           │
│  Nodes: consumer, dtc, feeder, substation, lt_line, ...         │
│  Edges: Network_Topology + consumer→dtc/feeder                  │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Graph MAE (self-supervised)                           │
│  Mask 30% node features → HGT encode → reconstruct              │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: HGT Encoder → 128-dim consumer embeddings             │
└───────────────┬─────────────────────────────┬───────────────────┘
                ▼                             ▼
┌───────────────────────────┐   ┌─────────────────────────────────┐
│  Anomaly: recon error +   │   │  Silent meter: LightGBM on      │
│  Isolation Forest         │   │  embeddings + tabular features  │
│  → HDBSCAN → 5-class RC   │   │  Jan-Feb → Mar temporal val     │
└───────────────────────────┘   └─────────────────────────────────┘
```

| Component | Model | Role |
|-----------|-------|------|
| Pretraining | **Graph MAE** | Masked feature reconstruction (SOTA self-supervised graph learning) |
| Encoder | **HGT** (Heterogeneous Graph Transformer) | Typed attention over AMI hierarchy |
| Anomaly | Reconstruction error + Isolation Forest | Unsupervised defect detection |
| Clustering | **HDBSCAN** | Root-cause discovery |
| Silent meter | **LightGBM** on GNN embeddings | Downstream head (GNN is core learner) |

---

## Sheets used vs omitted

### Used (labels stripped at load)

| Sheet | Rows | Purpose |
|-------|------|---------|
| `AMI_Comm_Health` | 18,100 | Primary consumer comm features |
| `Consumer` | 18,100 | Hierarchy, billing, PQ aggregates |
| `Consumer_Monthly_Readings` | 54,300 | Monthly PQ, billing, read metrics |
| `DCU_Daily_Status` | 50,400 | DCU daily time series |
| `Consumer_Event_Log` | 89,665 | Meter events |
| `Asset_Comm_Health` | ~1,600 | DTC/feeder/PT meter health |
| `Network_Topology` | 20,454 | Graph edges |

### Omitted entirely

| Sheet | Reason |
|-------|--------|
| `README` | Documentation; describes GT design |
| `Summary` | Aggregate defect counts (leakage) |
| `Event_Class` | Reference metadata |
| `Power_Plant` … `LT_Line` | Redundant (in Network_Topology) |
| `Feeder_Monthly_Audit` | GT columns |
| `Asset_Event_Log` | Covered by Consumer_Event_Log |

---

## Columns omitted (global blocklist)

Enforced in `src/data/loader.py` — never passed to training code:

- `Is_Defect`, `Defect_Category`, `Defect_Type`, `Defect_Start_Month`
- `Record_Label`, `Consumer_Is_Defect`, `Linked_Cause_GT`
- Any column matching `^GT_.*` or `.*_GT$`

### Per-sheet drops

| Sheet | Dropped columns |
|-------|-----------------|
| `AMI_Comm_Health` | Is_Defect, Defect_Category, Defect_Type, Defect_Start_Month |
| `Consumer` | Is_Defect, Record_Label, Defect_Category, Defect_Type, Defect_Start_Month |
| `Consumer_Monthly_Readings` | Is_Defect, Defect_Category, Defect_Type |
| `DCU_Daily_Status` | Is_Defect, Defect_Category, Defect_Type |
| `Asset_Comm_Health` | Is_Defect, Defect_Category, Defect_Type |
| `Consumer_Event_Log` | Linked_Cause_GT, Consumer_Is_Defect |

### Kept (observable, not hints)

`Comm_Status`, `NIC_Status`, `Billing_Basis`, `PQ_Data_Status`, all comm metrics, raw event classes.

---

## Dual training modes

Set in `config.yaml` → `training.mode` or CLI `--mode full|split`.

### Mode `full` (production default)

- All 18,100 consumers train Graph MAE + HGT
- Temporal validation: Jan–Feb features → Mar silent status
- No feeder holdout
- Use when deploying final model or external test xlsx arrives later

### Mode `split` (internal benchmarking)

- **Feeder-level 80/20 holdout** (~15 of 72 feeders)
- Stratified by observable Non-Communicating + Intermittent rate per feeder
- GNN trained on train subgraph only; inductive inference on test feeders
- Primary GT metrics on held-out 20%

```yaml
training:
  mode: split   # or "full"
  split:
    strategy: feeder_holdout
    holdout_ratio: 0.20
    seed: 2026
    manifest_path: artifacts/split_manifest.json
```

---

## Root-cause taxonomy (5 classes)

| Class | Signatures |
|-------|------------|
| **Tamper** | Event 9 + prior tamper events (28, 11, 12, …) |
| **Network coverage** | Low RSSI, high hop count, packet loss, DCU-up RF-fail |
| **Hardware** | NIC Faulty/Removed (no tamper prelude), DCU down cascade |
| **Firmware** | Events 8/30, post-upgrade comm drop |
| **Transient** | Power failures, outage spike, intermittent recovery |

---

## Evaluation layers

1. **Unsupervised**: reconstruction MSE, silhouette, Davies-Bouldin
2. **Observable-proxy**: Jan–Feb → Mar Comm_Status AUROC (no GT)
3. **GT-backed** (`evaluate.sh` only): macro-F1, confusion matrix vs Defect_Type mapped to 5 classes

Evaluation adapts to mode via `artifacts/split_manifest.json` and `artifacts/training_meta.json`.

---

## Shell scripts

```bash
./scripts/setup.sh                          # venv + dependencies
./scripts/eda.sh                            # pre-training plots
./scripts/train.sh                          # uses config.yaml mode
./scripts/train.sh --mode full              # entire dataset
./scripts/train.sh --mode split             # feeder holdout
./scripts/predict.sh path/to/test.xlsx      # predictions CSV
./scripts/evaluate.sh path/to/data.xlsx     # metrics (GT if present)
```

---

## Output schemas

### `consumer_predictions.csv`

| Column | Description |
|--------|-------------|
| `consumer_code` | Join key |
| `anomaly_score` | Combined recon + isolation score |
| `root_cause_class` | One of 5 classes |
| `nic_removal_root_cause` | Root cause if NIC removed / event 9 |
| `silent_risk_score` | 0–1 next-period silent risk |
| `silent_risk_rank` | Rank among all consumers |
| `missed_reads_q1` | Observable missed reads |
| `estimated_bill_attributed_cause` | Root cause for RNA billing |

### `attribution_summary.csv`

Cause × (missed_reads, rna_bill_count, consumer_count)
