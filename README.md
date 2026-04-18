# Telecom Network Anomaly Detection

**Author:** Anurag Singh  
**Model:** Transformer Autoencoder — unsupervised, reconstruction-error based  
**Hardware:** Apple M5 Pro · Metal Performance Shaders (MPS) GPU · ~17 s/epoch  
**Dataset:** 129,600 rows · 37 KPIs · 13 anomaly types · 7 observability layers

---

## Overview

An end-to-end **production-grade**, unsupervised anomaly detection system for telecom and cloud-native infrastructure. A Transformer Autoencoder is trained exclusively on normal telemetry; at inference time, windows whose reconstruction error exceeds a statistical threshold are flagged as anomalies — **no labels required during training**.

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

Trained on ~107 K normal-only windows. Evaluated on ~19 K test steps (6.4% anomaly rate).

| Metric | Score |
|---|---|
| Accuracy | 0.888 |
| Precision | 0.696 |
| **Recall** | **1.000** |
| **F1 Score** | **0.821** |
| **ROC-AUC** | **0.966** |
| **Avg Precision (AP)** | **0.827** |

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

---

## Architecture

```
Input window  [batch, seq_len=24, features=37]
        │
  Linear projection → d_model=128
        │
  Sinusoidal Positional Encoding
        │
  ┌─────────────────────────────────────┐
  │  Transformer Encoder                │
  │  · 2 layers, 4 attention heads      │
  │  · FFN dim = 256                    │
  │  · Pre-LayerNorm (training stable)  │
  │  · Dropout = 0.1 (within layers)    │
  └─────────────────────────────────────┘
        │
  Dropout(0.2)   ← prevents encoder memorisation
        │
  Bottleneck: Linear(128 → 64) → Linear(64 → 128)
        │
  ┌─────────────────────────────────────┐
  │  Transformer Decoder                │
  │  (learnable query tokens per step)  │
  │  · Same config as encoder           │
  └─────────────────────────────────────┘
        │
  Dropout(0.2)   ← prevents decoder memorisation
        │
  Linear projection → features=37
        │
Output reconstruction  [batch, seq_len=24, features=37]

Anomaly score  =  mean squared reconstruction error over the window
Threshold      =  mean(train errors) + 5 × std(train errors)
```

**Total trainable parameters: 691,813**

---

## Production Design

| Concern | Implementation |
|---|---|
| **No label dependency** | Fully unsupervised — threshold derived from training error distribution |
| **Normal-only training** | Anomaly labels loaded in preprocessor; anomalous rows removed from train and val DataLoaders |
| **Zero data leakage** | `StandardScaler` fitted on training split only; applied to val/test |
| **Device portability** | Auto-detects CUDA → MPS (Apple Silicon) → CPU at runtime |
| **Feature consistency** | `KPI_COLUMNS` derived from `KPI_META` — single source of truth, never drifts |
| **Stale checkpoint detection** | `main.py` auto-clears checkpoints when feature schema changes |
| **Early stopping** | Patience-based val-loss monitoring; saves best checkpoint |
| **Cross-layer anomaly realism** | Each anomaly type cascades across correlated KPIs (e.g. a node hardware failure affects pod restarts, Istio errors, app error rate, and throughput simultaneously) |
| **Modular pipeline** | Data generation, preprocessing, training, detection, and evaluation are fully decoupled |
| **Live simulation** | `pipeline.py` replays telemetry at configurable speed for staging/demo |

---

## Training Environment

| Item | Detail |
|---|---|
| **Hardware** | Apple M5 Pro — Metal Performance Shaders (MPS) GPU |
| **Framework** | PyTorch (MPS backend) |
| **Training time** | ~17 s/epoch · Early stopped at epoch 12 |
| **Dataset** | 129,600 rows × 37 features (90 days at 1-min resolution) |
| **Train / Val / Test** | 75% / 10% / 15% — sequential split, no shuffling |
| **Batch size** | 64 windows |
| **Optimiser** | Adam · LR = 1e-3 |

Runs unmodified on NVIDIA CUDA GPUs and CPU — device is auto-selected at runtime via `get_device()` in `config.py`.

---

## Project Structure

```
telecom-network-anomaly-detection/
├── config.py                       # All hyperparameters & KPI metadata (single source of truth)
├── data_generator.py               # Synthetic telemetry + multi-layer anomaly injection
├── preprocessor.py                 # Sliding-window DataLoader + StandardScaler (normal-only)
├── models/
│   ├── __init__.py                 # build_model() factory
│   └── transformer_autoencoder.py  # Transformer encoder-decoder with latent bottleneck
├── trainer.py                      # Training loop, early stopping, threshold computation
├── detector.py                     # AnomalyDetector — per-step sliding-window scoring
├── evaluate.py                     # Metrics, multi-panel dashboard, per-type recall report
├── main.py                         # Orchestrator: generate → preprocess → train → evaluate
├── benchmark.py                    # Standalone benchmark with dedicated result plots
├── pipeline.py                     # Live simulation (real-time telemetry replay)
├── predict.py                      # Inference script — score any telemetry CSV out of the box
├── requirements.txt                # Python dependencies
├── data/                           # telemetry.csv + anomaly_labels.csv  (auto-generated)
├── checkpoints/                    # best_model.pt, scaler.pkl, threshold.npy  (pre-trained)
└── results/                        # PNG dashboards + benchmark_comparison.csv
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Apple Silicon (MPS):** PyTorch ≥ 2.0 required.  
> **NVIDIA CUDA:** Install the matching `torch` build from [pytorch.org](https://pytorch.org).

### 2. Run inference on your own data (pre-trained model included)

Pre-trained weights (`best_model.pt`), the fitted scaler (`scaler.pkl`), and the anomaly threshold (`threshold.npy`) are committed to the repo under `checkpoints/` — no training required.

```bash
python predict.py --input your_telemetry.csv
```

This writes `your_telemetry_predictions.csv` with three columns:

| timestamp | anomaly_score | is_anomaly |
|---|---|---|
| 2024-01-15 08:00 | 0.012431 | 0 |
| 2024-01-15 08:01 | 0.341780 | 1 |

**Options:**

```bash
# Save predictions to a custom path
python predict.py --input data.csv --output flagged.csv

# Override the threshold (higher = fewer alerts, lower = more sensitive)
python predict.py --input data.csv --threshold 0.08
```

**Input CSV requirements:** must contain all 37 KPI columns listed in `config.py`. A `timestamp` column is optional but recommended.  
The first 23 rows (one window − 1) produce no score while the sliding window warms up.

---

### 3. Run the full training pipeline

```bash
python main.py
```

This will:
1. Generate 129,600 rows of synthetic telemetry (90 days × 1-min intervals, 37 KPIs)
2. Strip anomalous rows from train/val DataLoaders (normal-only training)
3. Fit `StandardScaler` on training split only — no leakage
4. Train the Transformer Autoencoder (MPS / CUDA / CPU auto-detected)
5. Compute anomaly threshold — mean + 5σ of training reconstruction errors
6. Evaluate on test split; save all plots + CSV to `results/`

### 4. Run the live simulation

```bash
python pipeline.py
```

Replays test telemetry at 100× real-time speed, printing anomaly alerts as they fire.

### 5. Standalone benchmark

```bash
python benchmark.py
```

Saves training curve, anomaly score timeline, PR curve, and confusion matrix to `results/`.

---

## Configuration

All settings live in [`config.py`](config.py):

| Parameter | Default | Description |
|---|---|---|
| `FEED_INTERVAL_MIN` | 1 | Sampling interval (minutes) |
| `HISTORY_DAYS` | 90 | Days of telemetry to generate |
| `WINDOW_SIZE` | 24 | Sliding window length (steps) |
| `HIDDEN_DIM` | 128 | Transformer d_model |
| `LATENT_DIM` | 64 | Bottleneck dimension |
| `NUM_HEADS` | 4 | Multi-head attention heads |
| `NUM_LAYERS` | 2 | Encoder + decoder layers each |
| `THRESHOLD_SIGMA` | 5.0 | Threshold = mean + σ × std(train errors) |
| `EPOCHS` | 50 | Max training epochs |
| `PATIENCE` | 10 | Early-stopping patience |
| `BATCH_SIZE` | 64 | Training batch size |
| `LEARNING_RATE` | 1e-3 | Adam optimiser LR |

---

## Anomaly Types

All 13 types are injected **into the test split only** — training and validation data are fully clean:

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
| `results/anomaly_dashboard.png` | Reconstruction error timeline + all 37 KPI subplots with anomaly highlighting |
| `results/layer_overview.png` | Normalised KPI overlay grouped by observability layer |
| `results/pr_confusion.png` | Precision-Recall curve + confusion matrix |
| `results/training_curve.png` | Train vs. validation loss per epoch |
| `results/benchmark_comparison.csv` | Summary metrics: accuracy, F1, AUC, TP/FP/TN/FN, threshold |

---

## License

MIT

---

*Built with PyTorch · Trained on Apple M5 Pro MPS · Author: Anurag Singh*
