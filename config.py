# config.py — Central configuration for all hyperparameters and settings

import torch


def get_device() -> str:
    """Return the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Config:

    # -- Data ------------------------------------------------
    FEED_INTERVAL_MIN   = 1           # telemetry feed every 1 minute (~130 K steps over 90 days)
    HISTORY_DAYS        = 90          # training data covers 90 days
    ANOMALY_RATIO       = 0.05        # 5% of test data injected as anomalies
    RANDOM_SEED         = 42

    # KPI columns fed from telemetry sources — grouped by observability layer
    # ── Telecom Network  : OSS/BSS probes + RAN counters ─────────
    # ── K8s Compute      : Prometheus node-exporter + kube-state ─
    # ── Cloud Storage    : CSI driver + cloud provider APIs ───────
    # ── Istio            : Envoy telemetry v2 (sidecar proxies) ──
    # ── Application      : OpenTelemetry traces + APM agent ───────
    # ── Database         : DB exporter + slow-query logs ──────────
    # ── Hardware Health  : Node conditions + IPMI / hardware logs ─
    # Single source of truth: insertion order defines the column order used by the
    # model. KPI_COLUMNS is derived from these keys — do not maintain separately.
    # Value tuple: (human label, chart colour, observability layer)
    KPI_META = {
        # Telecom
        "latency_ms":                        ("Latency (ms)",            "tab:blue",        "Telecom"),
        "throughput_mbps":                   ("Throughput (Mbps)",       "tab:green",       "Telecom"),
        "call_success_rate":                 ("Call Success (%)",         "tab:orange",      "Telecom"),
        "bearer_establishment_success_rate": ("Bearer Est. (%)",          "tab:purple",      "Telecom"),
        "complaint_rate":                    ("Complaints/hr",            "tab:red",         "Telecom"),
        "packet_loss_rate":                  ("Packet Loss (%)",          "tab:brown",       "Telecom"),
        "jitter_ms":                         ("Jitter (ms)",              "tab:pink",        "Telecom"),
        "handover_success_rate":             ("Handover Success (%)",     "tab:olive",       "Telecom"),
        # K8s Compute
        "node_cpu_utilization":              ("Node CPU (%)",             "tomato",          "K8s Compute"),
        "node_memory_utilization":           ("Node Memory (%)",          "coral",           "K8s Compute"),
        "pod_restart_count":                 ("Pod Restarts/hr",          "crimson",         "K8s Compute"),
        "pod_cpu_throttling_rate":           ("CPU Throttling (%)",       "firebrick",       "K8s Compute"),
        # Cloud Storage
        "node_disk_read_mbps":               ("Disk Read (Mbps)",         "steelblue",       "Storage"),
        "node_disk_write_mbps":              ("Disk Write (Mbps)",        "dodgerblue",      "Storage"),
        "storage_read_latency_ms":           ("Store Read Lat. (ms)",     "royalblue",       "Storage"),
        "storage_write_latency_ms":          ("Store Write Lat. (ms)",    "navy",            "Storage"),
        "storage_utilization_pct":           ("Storage Used (%)",         "slateblue",       "Storage"),
        "storage_iops":                      ("Storage IOPS",             "mediumpurple",    "Storage"),
        # Istio
        "istio_request_rate":                ("Istio Req/s",              "seagreen",        "Istio"),
        "istio_error_rate":                  ("Istio Error (%)",          "darkgreen",       "Istio"),
        "istio_p99_latency_ms":              ("Istio P99 Lat. (ms)",      "limegreen",       "Istio"),
        "istio_retry_rate":                  ("Istio Retry (%)",          "forestgreen",     "Istio"),
        # Application
        "app_response_time_ms":              ("App Response (ms)",        "darkorange",      "Application"),
        "app_error_rate":                    ("App Error (%)",            "orangered",       "Application"),
        "app_request_rate":                  ("App Req/s",                "goldenrod",       "Application"),
        "app_active_connections":            ("Active Connections",        "gold",            "Application"),
        "app_5xx_rate":                      ("App 5xx/s",                "darkgoldenrod",   "Application"),
        # Database
        "db_query_latency_ms":               ("DB Query Lat. (ms)",       "teal",            "Database"),
        "db_connection_pool_utilization":    ("DB Pool Used (%)",         "darkcyan",        "Database"),
        "db_slow_query_rate":                ("DB Slow Queries/s",        "cadetblue",       "Database"),
        "db_replication_lag_ms":             ("DB Repl. Lag (ms)",        "lightseagreen",   "Database"),
        "db_transaction_rate":               ("DB TPS",                   "mediumaquamarine","Database"),
        # Hardware Health
        "node_not_ready_count":              ("NotReady Nodes",           "black",           "Hardware"),
        "node_disk_pressure_count":          ("DiskPressure Nodes",       "dimgray",         "Hardware"),
        "node_memory_pressure_count":        ("MemPressure Nodes",        "darkgray",        "Hardware"),
        "hardware_error_log_rate":           ("HW Errors/hr",             "gray",            "Hardware"),
        "node_network_rx_errors":            ("NIC RX Errors/s",          "slategray",       "Hardware"),
    }

    # Derived from KPI_META keys — always in sync, never edit manually
    KPI_COLUMNS = list(KPI_META.keys())

    # -- Preprocessing ----------------------------------------
    WINDOW_SIZE  = 24   # 24 steps × 5 min = 2-hour sliding window
    STRIDE       = 1    # sliding window stride
    TRAIN_RATIO  = 0.75
    VAL_RATIO    = 0.10 # remaining 15% = test

    # -- Model ------------------------------------------------
    MODEL_TYPE   = "transformer"
    HIDDEN_DIM   = 128              # increased for 37-feature input
    NUM_LAYERS   = 2
    NUM_HEADS    = 4                # transformer only
    DROPOUT      = 0.1
    LATENT_DIM   = 64               # increased bottleneck for richer feature space

    # -- Training ---------------------------------------------
    EPOCHS       = 50
    BATCH_SIZE   = 64
    LEARNING_RATE= 1e-3
    PATIENCE     = 10               # early stopping

    # -- Anomaly Detection ------------------------------------
    # Threshold = mean(train_errors) + THRESHOLD_SIGMA × std(train_errors)
    THRESHOLD_SIGMA = 5.0

    # -- Ensemble ---------------------------------------------
    # Weighted vote across all three detectors.
    # Weights must sum to 1.0.
    ENSEMBLE_WEIGHTS = {
        "transformer":      0.50,   # highest weight — best temporal modelling
        "mlp_autoencoder":  0.30,   # complementary feedforward view
        "isolation_forest": 0.20,   # tree-based, catches point anomalies
    }
    # Ensemble score ∈ [0, 1]. Flag anomaly if score exceeds this.
    ENSEMBLE_THRESHOLD = 0.80

    # -- NOC Alert API ----------------------------------------
    NOC_ALERTS_ENABLED    = True
    NOC_ALERT_ENDPOINT    = "https://noc.example.com/api/v1/alerts"  # override via env NOC_ALERT_ENDPOINT
    NOC_API_KEY           = ""                                         # override via env NOC_API_KEY
    NOC_TIMEOUT_SEC       = 5
    NOC_SEVERITY_THRESHOLDS = {
        "LOW":    0.50,   # ensemble score ≥ 0.50
        "MEDIUM": 0.70,   # ensemble score ≥ 0.70
        "HIGH":   0.85,   # ensemble score ≥ 0.85
    }

    # -- Pipeline (live simulation) ----------------------------
    SIMULATE_SPEED_FACTOR = 100     # 1 real second = 100× feed time (for demo)

    # -- Paths -------------------------------------------------
    DATA_PATH         = "data/telemetry.csv"
    ANOMALY_PATH      = "data/anomaly_labels.csv"
    MODEL_PATH        = "checkpoints/best_model.pt"
    MLP_MODEL_PATH    = "checkpoints/mlp_model.pt"
    IF_PATH           = "checkpoints/isolation_forest.pkl"
    ENSEMBLE_PATH     = "checkpoints/ensemble_params.pkl"
    RESULTS_PATH      = "results/"
