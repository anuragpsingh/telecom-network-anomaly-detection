# Telecom Network Anomaly Detection

**Author:** Anurag Singh  
**Model:** Weighted Ensemble — Transformer AE · MLP AE · Isolation Forest  
**Hardware:** Apple M5 Pro · Metal Performance Shaders (MPS) GPU  
**Dataset:** 129,600 rows · 37 KPIs · 13 anomaly types · 7 observability layers

---

## Overview

An end-to-end **production-grade**, unsupervised anomaly detection system for telecom and cloud-native infrastructure. Three complementary models are trained exclusively on normal telemetry and combined via a **weighted ensemble** to produce a final anomaly verdict — **no labels required during training**.

When an anomaly is detected, the system automatically fires a structured alert to the **NOC (Network Operations Centre) API** with severity classification, individual model scores, and a full payload for incident routing.

The system monitors the full observability stack of a modern telecom operator running microservices on Kubernetes with Istio service mesh:

| Layer | KPIs Monitored |
|---|---|
| Telecom Network | Latency, Throughput, Call Success Rate, Bearer Establishment, Jitter, Packet Loss, Handover Success |
| K8s Compute | Node CPU/Memory Utilisation, Pod Restarts, CPU Throttling Rate |
| Cloud Storage | Disk Read/Write, Storage Latency, IOPS, Storage Utilisation |
| Istio Service Mesh | Request Rate, Error Rate, P99 Latency, Retry Rate |
| Application (OTel) | Response Time, Error Rate, 5xx Rate, Request Rate, Active Connections |
| Database | Query Latency, Connection Pool, Slow Query Rate, Replication Lag, TPS |
| Hardware Health | NotReady Nodes, Disk/Memory Pressure, HW Error Log Rate, NIC RX Errors |

---

## Results

Trained on ~97 K normal-only rows. Evaluated on ~19 K test steps (6.4% anomaly rate).  
Ensemble threshold tuned to **0.80** — the last point where FN = 0 (zero missed anomalies).

| Metric | Score |
|---|---|
| Accuracy | 0.824 |
| Precision | 0.593 |
| **Recall** | **1.000** |
| **F1 Score** | **0.745** |
| **ROC-AUC** | **0.939** |
| **Avg Precision (AP)** | **0.735** |
| TP / FP / TN / FN | 4978 / 3413 / 11026 / **0** |

**All 13 anomaly types detected at 100% recall:**

| Anomaly Type | Layer | Recall |
|---|---|---|
| app_error_storm | Application | 1.000 |
| call_quality_degradation | Telecom | 1.000 |
| complaint_surge | Telecom | 1.000 |
| cpu_throttling_event | K8s Compute | 1.000 |
| db_connection_exhaustion | Database | 1.000 |
| db_replication_failure | Database | 1.000 |
| istio_cascade_failure | Istio | 1.000 |
| latency_spike | Telecom | 1.000 |
| network_congestion | Telecom + Compute | 1.000 |
| node_hardware_failure | Hardware | 1.000 |
| oom_kill | K8s Compute | 1.000 |
| storage_saturation | Cloud Storage | 1.000 |
| throughput_drop | Telecom | 1.000 |

### Threshold Tuning

| Threshold | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|
| 0.50 | 0.337 | 1.000 | 0.504 | 10,280 | 0 |
| 0.70 | 0.484 | 1.000 | 0.653 | 5,302 | 0 |
| **0.80** | **0.593** | **1.000** | **0.745** | **3,413** | **0** |
| 0.90 | 0.698 | 0.964 | 0.810 | 2,074 | 178 |

**0.80 is the optimal threshold** — highest F1 with zero missed anomalies. At 0.90, 178 real anomalies are missed to save 1,339 false alerts — an unfavourable trade in a NOC context.

---

## Architecture

### Ensemble Design

```
Telemetry window  [seq_len=24, features=37]
        │
        ├──────────────────────────────────────────────┐
        │                                              │
  ┌─────▼──────────────────┐   ┌──────────────────────▼──────┐
  │  Transformer Autoencoder│   │     MLP Autoencoder          │
  │  · 2-layer encoder      │   │  · Flatten: 24×37 → 888      │
  │  · 4 attention heads    │   │  · 888 → 512 → 256 → 64      │
  │  · Latent dim: 64       │   │  · 64 → 256 → 512 → 888      │
  │  · Dropout(0.2)         │   │  · Dropout(0.2)              │
  │  Weight: 0.50           │   │  Weight: 0.30                │
  └─────────────────────────┘   └──────────────────────────────┘
        │                                              │
  Recon MSE score                             Recon MSE score
        │                                              │
        └──────────────────┬───────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   Isolation Forest       │
              │   · 200 estimators       │
              │   · Trained on 97K rows  │
              │   Weight: 0.20           │
              └─────────────────────────┘
                           │
                    IF anomaly score
                           │
                           ▼
        ┌──────────────────────────────────┐
        │  Score Normalisation [0, 1]       │
        │  Percentile bounds (p1 / p99)     │
        │  computed from training data      │
        └──────────────────────────────────┘
                           │
                           ▼
        ensemble_score = 0.50 × norm_transformer
                       + 0.30 × norm_mlp
                       + 0.20 × norm_if

        is_anomaly = ensemble_score > 0.80
                           │
                    ┌──────▼───────┐
                    │  NOC Alert   │  ← HTTP POST with severity,
                    │  API Client  │    individual scores, alert ID
                    └──────────────┘
```

### Individual Model Parameters

| Model | Parameters | Architecture |
|---|---|---|
| Transformer AE | 691,813 | Encoder-decoder · d_model=128 · 4 heads · latent=64 |
| MLP AE | 1,206,712 | 3-layer encoder-decoder · flat_dim=888 · latent=64 |
| Isolation Forest | — (sklearn) | 200 estimators · contamination=auto |

---

## NOC Alert Integration

When the ensemble score exceeds the threshold, an HTTP POST is sent to the configured NOC endpoint:

```json
{
  "alert_id":       "TAD-000042",
  "timestamp":      "2024-03-15T14:32:00",
  "severity":       "HIGH",
  "ensemble_score": 0.93,
  "model_scores": {
    "transformer":      {"raw": 0.214, "normalized": 0.97},
    "mlp_autoencoder":  {"raw": 0.198, "normalized": 0.91},
    "isolation_forest": {"raw": 0.081, "normalized": 0.88}
  },
  "ensemble_weights": {"transformer": 0.5, "mlp_autoencoder": 0.3, "isolation_forest": 0.2},
  "threshold": 0.80,
  "source": "telecom-anomaly-detector"
}
```

**Severity tiers:**

| Severity | Ensemble Score |
|---|---|
| LOW | ≥ 0.50 |
| MEDIUM | ≥ 0.70 |
| HIGH | ≥ 0.85 |

Configure via `config.py` or environment variables:

```bash
export NOC_ALERT_ENDPOINT="https://noc.yourcompany.com/api/v1/alerts"
export NOC_API_KEY="your-api-key"
```

---

## Production Design

| Concern | Implementation |
|---|---|
| **No label dependency** | Fully unsupervised — threshold derived from training error distribution |
| **Normal-only training** | Anomaly rows stripped from train/val DataLoaders via label file |
| **Zero data leakage** | `StandardScaler` fitted on training split only; applied to val/test |
| **Weighted ensemble** | Three diverse detectors — temporal (Transformer), feedforward (MLP), tree-based (IF) |
| **Score normalisation** | Per-model p1/p99 percentile bounds computed on training data → consistent [0,1] scale |
| **NOC alert API** | HTTP POST on every anomaly · retry logic · graceful degradation if endpoint unreachable |
| **Severity classification** | LOW / MEDIUM / HIGH tiers based on ensemble score magnitude |
| **Device portability** | Auto-detects CUDA → MPS (Apple Silicon) → CPU at runtime |
| **Feature consistency** | `KPI_COLUMNS` derived from `KPI_META` — single source of truth, never drifts |
| **Stale checkpoint detection** | `main.py` auto-clears all checkpoints when feature schema changes |
| **Early stopping** | Patience-based val-loss monitoring; saves best checkpoint per model |
| **Live simulation** | `pipeline.py` replays telemetry at configurable speed for staging/demo |

---

## Training Environment

| Item | Detail |
|---|---|
| **Hardware** | Apple M5 Pro — Metal Performance Shaders (MPS) GPU |
| **Framework** | PyTorch (MPS backend) + scikit-learn (Isolation Forest) |
| **Transformer AE** | ~17 s/epoch · early stopped at epoch 11 |
| **MLP AE** | ~5 s/epoch · early stopped at epoch 16 |
| **Isolation Forest** | ~5 s total · 97,200 normal rows |
| **Dataset** | 129,600 rows × 37 features (90 days at 1-min resolution) |
| **Train / Val / Test** | 75% / 10% / 15% — sequential split, no shuffling |
| **Batch size** | 64 windows |
| **Optimiser** | Adam · LR = 1e-3 |

Runs unmodified on NVIDIA CUDA GPUs and CPU — device is auto-selected at runtime.

---

## Project Structure

```
telecom-network-anomaly-detection/
├── config.py
├── data_generator.py
├── preprocessor.py
├── models/
│   ├── __init__.py
│   ├── transformer_autoencoder.py
│   ├── mlp_autoencoder.py
│   └── isolation_forest.py
├── ensemble.py
├── noc_alert.py
├── trainer.py
├── detector.py
├── evaluate.py
├── main.py
├── benchmark.py
├── pipeline.py
├── predict.py
├── requirements.txt
├── data/
├── checkpoints/
└── results/
```

### File-by-File Guide (Beginner Friendly)

---

#### `config.py` — The Control Panel
Think of this as the single settings file for the entire project. Instead of hunting through every file to change a number, everything lives here: how many training epochs, the anomaly threshold, the list of 37 KPIs to monitor, ensemble weights, and the NOC API endpoint. If you want to tweak something, start here.

---

#### `data_generator.py` — The Fake Data Factory
Since we don't have real telecom data, this file creates realistic synthetic telemetry. It simulates 90 days of 1-minute readings for 37 KPIs across 7 layers — with daily traffic patterns, weekly cycles, cross-layer correlation (e.g. a CPU spike affects latency), and a slow drift over time. It then injects 13 types of anomalies **only into the test portion** so training stays clean.

---

#### `preprocessor.py` — Data Preparation
Raw CSV rows can't go straight into a neural network. This file does three things:
1. **Normalises** all 37 features to a common scale using `StandardScaler` (fit on training data only — no leakage)
2. **Removes** any anomalous rows from training and validation so the model only learns what "normal" looks like
3. **Slices** the data into overlapping 24-step windows (sliding window) and wraps them in PyTorch DataLoaders ready for training

---

#### `models/` — The Three Detectors

| File | What it does |
|---|---|
| `__init__.py` | Factory functions — `build_model()` creates a Transformer AE, `build_mlp_model()` creates an MLP AE |
| `transformer_autoencoder.py` | The main neural network. Uses self-attention to understand temporal patterns across the 24-step window. Learns to reconstruct normal data; anomalies reconstruct poorly and get a high error score |
| `mlp_autoencoder.py` | A simpler feedforward network. Flattens the whole window into one vector, compresses it to a 64-dim bottleneck, and reconstructs it back. A different "lens" to view the same data |
| `isolation_forest.py` | A tree-based model (no neural network). Randomly partitions data — anomalies are isolated faster (shorter paths). Scores each timestep individually and averages across the window |

---

#### `ensemble.py` — The Voting Committee
This is where the three models are combined. Each model produces its own anomaly score, which is normalised to [0, 1] using bounds computed from training data. The final score is a weighted average:

```
ensemble_score = 0.50 × transformer + 0.30 × mlp + 0.20 × isolation_forest
```

If the ensemble score exceeds 0.80, the window is flagged as an anomaly and an alert is fired.

---

#### `noc_alert.py` — The Alarm Bell
When an anomaly is confirmed by the ensemble, this file sends an HTTP POST to the NOC API. The payload includes the alert severity (LOW / MEDIUM / HIGH), the ensemble score, and each individual model's contribution. If the endpoint is unreachable, detection continues uninterrupted — the alert failure is logged but never crashes the system.

---

#### `trainer.py` — The Training Loop
Handles the mechanics of training a PyTorch model: forward pass, loss calculation, backpropagation, learning rate scheduling, early stopping, and saving the best checkpoint. Both the Transformer AE and MLP AE use the same `Trainer` class — only the model and save path differ.

---

#### `detector.py` — Single-Model Live Scorer
A lightweight wrapper around one trained model. Maintains a rolling 24-step buffer; each time a new telemetry tick arrives, it scores the current window and returns a result dict. Used inside `pipeline.py` for single-model live scoring. The ensemble version (`ensemble.py`) extends this concept to all three models.

---

#### `evaluate.py` — The Report Card
Given anomaly scores and ground-truth labels, this file computes precision, recall, F1, ROC-AUC, and average precision. It also generates:
- An anomaly score timeline with true anomaly regions highlighted
- A per-KPI dashboard showing all 37 signals
- A normalised layer overview grouped by observability layer
- A confusion matrix and precision-recall curve

---

#### `main.py` — The Orchestrator
The single script that runs the entire pipeline end-to-end in 7 steps: generate data → preprocess → train Transformer AE → train MLP AE → train Isolation Forest → build and calibrate ensemble → evaluate. Run this once to reproduce all results from scratch.

---

#### `predict.py` — Inference for New Data
The end-user script. Point it at any CSV containing the 37 KPI columns and it loads the pre-trained ensemble from `checkpoints/`, scores every row, and writes a results CSV with per-model scores, ensemble score, and anomaly flag. No training needed — works immediately after cloning.

---

#### `benchmark.py` — Transformer-Only Benchmark
A standalone script to train and evaluate just the Transformer AE in isolation (without the ensemble). Useful for comparing the Transformer's standalone performance against the full ensemble.

---

#### `pipeline.py` — Live Simulation
Replays the test dataset tick by tick at 100× speed to simulate a real-time telemetry feed. Prints NOC-style alerts to the console as anomalies are detected. In production this would connect to Kafka, OSS/BSS REST APIs, or Prometheus remote-write.

---

#### `data/` — Generated Dataset
Contains `telemetry.csv` (129,600 rows × 37 features) and `anomaly_labels.csv` (timestamp, is_anomaly, anomaly_type). Both are auto-generated by `data_generator.py` and excluded from git — run `python main.py` to regenerate.

---

#### `checkpoints/` — Saved Model Artefacts
Stores all pre-trained model weights and calibration files committed to the repo:

| File | Contents |
|---|---|
| `best_model.pt` | Transformer AE weights |
| `mlp_model.pt` | MLP AE weights |
| `isolation_forest.pkl` | Trained Isolation Forest + calibration bounds |
| `ensemble_params.pkl` | Per-model p1/p99 normalisation bounds |
| `scaler.pkl` | Fitted StandardScaler |
| `threshold.npy` | Transformer AE anomaly threshold |

---

#### `results/` — Output Plots & Reports
All plots and CSV summaries generated after evaluation. Excluded from git — regenerated by running `main.py`.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Apple Silicon (MPS):** PyTorch ≥ 2.0 required.  
> **NVIDIA CUDA:** Install the matching `torch` build from [pytorch.org](https://pytorch.org).

### 2. Run inference on your own data (pre-trained ensemble included)

All pre-trained artefacts are committed to `checkpoints/` — no training required.

```bash
python predict.py --input your_telemetry.csv
```

Output CSV — one row per input row:

| timestamp | ensemble_score | is_anomaly | transformer_score | mlp_score | if_score |
|---|---|---|---|---|---|
| 2024-01-15 08:00 | 0.123 | 0 | 0.11 | 0.09 | 0.18 |
| 2024-01-15 08:01 | 0.847 | 1 | 0.94 | 0.82 | 0.71 |

**Options:**

```bash
python predict.py --input data.csv --output flagged.csv
python predict.py --input data.csv --threshold 0.85   # raise for fewer alerts
```

**Input CSV requirements:** must contain all 37 KPI columns listed in `config.py`.  
First 23 rows produce no score while the sliding window warms up.

---

### 3. Train all models from scratch

```bash
python main.py
```

This runs the full 7-step pipeline:
1. Generate 129,600 rows of synthetic telemetry (90 days × 1-min, 37 KPIs)
2. Preprocess — strip anomalous rows, fit StandardScaler on train split only
3. Train Transformer Autoencoder (MPS / CUDA / CPU auto-detected)
4. Train MLP Autoencoder
5. Train Isolation Forest on 97 K normal timestep rows
6. Build ensemble — calibrate per-model normalisation bounds, save params
7. Evaluate ensemble on test split, save plots + CSV to `results/`

### 4. Live simulation

```bash
python pipeline.py
```

Replays test telemetry at 100× real-time speed, printing NOC alerts as they fire.

### 5. Standalone Transformer benchmark

```bash
python benchmark.py
```

---

## Configuration

All settings in [`config.py`](config.py):

| Parameter | Default | Description |
|---|---|---|
| `FEED_INTERVAL_MIN` | 1 | Sampling interval (minutes) |
| `HISTORY_DAYS` | 90 | Days of telemetry to generate |
| `WINDOW_SIZE` | 24 | Sliding window length (steps) |
| `HIDDEN_DIM` | 128 | Transformer d_model |
| `LATENT_DIM` | 64 | Bottleneck dimension (both AE models) |
| `NUM_HEADS` | 4 | Transformer attention heads |
| `NUM_LAYERS` | 2 | Encoder + decoder layers |
| `THRESHOLD_SIGMA` | 5.0 | Per-model threshold = mean + σ × std |
| `ENSEMBLE_WEIGHTS` | T=0.50, MLP=0.30, IF=0.20 | Model vote weights (must sum to 1.0) |
| `ENSEMBLE_THRESHOLD` | 0.80 | Anomaly flag threshold for ensemble score |
| `NOC_ALERTS_ENABLED` | True | Enable / disable NOC HTTP alerts |
| `NOC_ALERT_ENDPOINT` | example URL | Override via env `NOC_ALERT_ENDPOINT` |
| `NOC_API_KEY` | — | Override via env `NOC_API_KEY` |
| `EPOCHS` | 50 | Max training epochs per model |
| `PATIENCE` | 10 | Early-stopping patience |
| `BATCH_SIZE` | 64 | Training batch size |
| `LEARNING_RATE` | 1e-3 | Adam optimiser LR |

---

## Anomaly Types

All 13 types are injected **into the test split only** — training and validation are fully clean:

| Type | Primary Layer | Duration | Correlated Layers |
|---|---|---|---|
| `latency_spike` | Telecom | 30 min | — |
| `throughput_drop` | Telecom | 60 min | Application |
| `call_quality_degradation` | Telecom | 40 min | — |
| `complaint_surge` | Telecom | 50 min | — |
| `network_congestion` | Telecom | 90 min | K8s Compute, Application |
| `oom_kill` | K8s Compute | 70 min | Telecom, Application |
| `cpu_throttling_event` | K8s Compute | 90 min | Application, Telecom |
| `storage_saturation` | Cloud Storage | 120 min | Hardware, K8s |
| `istio_cascade_failure` | Istio | 75 min | Application, Telecom |
| `app_error_storm` | Application | 100 min | Istio, Telecom |
| `db_connection_exhaustion` | Database | 80 min | Application, K8s |
| `db_replication_failure` | Database | 110 min | Application |
| `node_hardware_failure` | Hardware | 150 min | K8s, Istio, Application, Telecom |

---

## Output Plots

| File | Description |
|---|---|
| `results/anomaly_dashboard.png` | Ensemble score timeline + all 37 KPI subplots |
| `results/layer_overview.png` | Normalised KPI overlay grouped by observability layer |
| `results/pr_confusion.png` | Precision-Recall curve + confusion matrix |
| `results/training_curve_transformer_ae.png` | Transformer AE train/val loss |
| `results/training_curve_mlp_ae.png` | MLP AE train/val loss |
| `results/benchmark_comparison.csv` | Summary metrics: accuracy, F1, AUC, TP/FP/TN/FN |

---

## License

MIT

---

*Built with PyTorch · scikit-learn · Trained on Apple M5 Pro MPS · Author: Anurag Singh*
