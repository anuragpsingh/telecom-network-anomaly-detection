# data_generator.py — Synthetic multi-layer telemetry with realistic patterns + anomaly injection
#
# Observability layers modelled:
#   Telecom Network  | K8s Compute | Cloud Storage | Istio Service Mesh
#   Application      | Database    | Hardware / Node Health

import numpy as np
import pandas as pd
import os
from config import Config


# ─────────────────────────────────────────────────────────────
# Traffic load patterns
# ─────────────────────────────────────────────────────────────

def _daily_pattern(t_hours: np.ndarray) -> np.ndarray:
    """Bimodal traffic curve: morning and evening peaks."""
    hour = t_hours % 24
    return (
        0.25
        + 0.45 * np.exp(-0.5 * ((hour - 8.5)  / 2.0) ** 2)   # morning peak
        + 0.30 * np.exp(-0.5 * ((hour - 19.0) / 2.5) ** 2)   # evening peak
    )


def _weekly_pattern(t_hours: np.ndarray) -> np.ndarray:
    """Weekends (day 5-6) carry ~20% lower traffic."""
    day = (t_hours // 24).astype(int) % 7
    return np.where(day >= 5, 0.80, 1.0)


# ─────────────────────────────────────────────────────────────
# Base signal generation — all 37 KPIs
# ─────────────────────────────────────────────────────────────

def generate_base_signals(n_points: int, interval_min: int = 5) -> pd.DataFrame:
    """
    Generate 37-feature time series across 6 observability layers.

    Correlations baked in:
      - All metrics share a common traffic load component
      - Shared correlated noise models cross-layer congestion
      - Gradual drift models infrastructure ageing over 90 days
      - Each layer has its own independent noise on top
    """
    t_min   = np.arange(n_points) * interval_min
    t_hours = t_min / 60.0

    load         = _daily_pattern(t_hours) * _weekly_pattern(t_hours)
    shared_noise = np.random.randn(n_points) * 0.05   # cross-layer congestion
    drift        = np.linspace(0, 1.0, n_points)       # 0→1 over 90 days

    rng = np.random.randn  # shorthand

    # ── Telecom Network ───────────────────────────────────────
    latency_ms = (
        20 + 15 * load + drift * 8 + shared_noise * 4 + rng(n_points) * 1.2
    ).clip(0)

    throughput_mbps = (
        120 - 30 * load - drift * 4 - shared_noise * 8 + rng(n_points) * 3.0
    ).clip(0)

    call_success_rate = (
        98.5 - 1.5 * load - drift * 0.4 - shared_noise * 0.6 + rng(n_points) * 0.2
    ).clip(0, 100)

    bearer_establishment_success_rate = (
        97.0 - 2.0 * load - drift * 0.5 - shared_noise * 0.5 + rng(n_points) * 0.25
    ).clip(0, 100)

    complaint_rate = (
        1.5 + 2.5 * load + drift * 0.6 + shared_noise * 0.4 + rng(n_points) * 0.15
    ).clip(0)

    packet_loss_rate = (
        0.1 + 0.6 * load + drift * 0.1 + shared_noise * 0.15 + rng(n_points) * 0.05
    ).clip(0, 100)

    jitter_ms = (
        2.0 + 4.0 * load + drift * 0.5 + shared_noise * 1.0 + rng(n_points) * 0.3
    ).clip(0)

    handover_success_rate = (
        99.0 - 1.2 * load - drift * 0.3 - shared_noise * 0.4 + rng(n_points) * 0.15
    ).clip(0, 100)

    # ── Kubernetes Compute ────────────────────────────────────
    node_cpu_utilization = (
        20 + 40 * load + drift * 5 + shared_noise * 6 + rng(n_points) * 2.0
    ).clip(0, 100)

    node_memory_utilization = (
        40 + 15 * load + drift * 20 + shared_noise * 2 + rng(n_points) * 1.0
    ).clip(0, 100)

    pod_restart_count = (
        0.1 + 0.4 * load + drift * 0.2 + np.abs(rng(n_points)) * 0.1
    ).clip(0)

    pod_cpu_throttling_rate = (
        5 + 25 * load + drift * 3 + shared_noise * 4 + rng(n_points) * 1.5
    ).clip(0, 100)

    # ── Cloud Storage ─────────────────────────────────────────
    node_disk_read_mbps = (
        50 + 30 * load + drift * 2 + shared_noise * 5 + rng(n_points) * 2.0
    ).clip(0)

    node_disk_write_mbps = (
        30 + 20 * load + drift * 2 + shared_noise * 3 + rng(n_points) * 1.5
    ).clip(0)

    storage_read_latency_ms = (
        5 + 5 * load + drift * 1.0 + shared_noise * 0.8 + rng(n_points) * 0.4
    ).clip(0)

    storage_write_latency_ms = (
        8 + 8 * load + drift * 1.5 + shared_noise * 1.0 + rng(n_points) * 0.5
    ).clip(0)

    # Storage utilisation grows steadily (drift) and oscillates with load
    storage_utilization_pct = (
        45 + drift * 30 + 3 * load + rng(n_points) * 0.5
    ).clip(0, 100)

    storage_iops = (
        200 + 350 * load + drift * 20 + shared_noise * 30 + rng(n_points) * 15
    ).clip(0)

    # ── Istio Service Mesh ────────────────────────────────────
    istio_request_rate = (
        80 + 500 * load + drift * 10 + shared_noise * 20 + rng(n_points) * 8
    ).clip(0)

    istio_error_rate = (
        0.4 + 1.2 * load + drift * 0.1 + shared_noise * 0.2 + rng(n_points) * 0.08
    ).clip(0, 100)

    istio_p99_latency_ms = (
        40 + 90 * load + drift * 5 + shared_noise * 8 + rng(n_points) * 3
    ).clip(0)

    istio_retry_rate = (
        0.8 + 2.5 * load + drift * 0.15 + shared_noise * 0.3 + rng(n_points) * 0.1
    ).clip(0)

    # ── Application Layer ─────────────────────────────────────
    # Response time is strongly correlated with istio p99 and DB latency
    app_response_time_ms = (
        60 + 120 * load + drift * 8 + shared_noise * 10 + rng(n_points) * 5
    ).clip(0)

    app_error_rate = (
        0.3 + 1.5 * load + drift * 0.1 + shared_noise * 0.25 + rng(n_points) * 0.08
    ).clip(0, 100)

    app_request_rate = (
        60 + 400 * load + drift * 8 + shared_noise * 15 + rng(n_points) * 6
    ).clip(0)

    app_active_connections = (
        200 + 800 * load + drift * 30 + shared_noise * 40 + rng(n_points) * 15
    ).clip(0)

    app_5xx_rate = (
        0.1 + 0.8 * load + drift * 0.05 + shared_noise * 0.1 + rng(n_points) * 0.04
    ).clip(0)

    # ── Database Layer ────────────────────────────────────────
    db_query_latency_ms = (
        10 + 20 * load + drift * 3 + shared_noise * 2 + rng(n_points) * 0.8
    ).clip(0)

    db_connection_pool_utilization = (
        30 + 40 * load + drift * 10 + shared_noise * 5 + rng(n_points) * 2
    ).clip(0, 100)

    db_slow_query_rate = (
        0.2 + 1.0 * load + drift * 0.2 + shared_noise * 0.15 + rng(n_points) * 0.06
    ).clip(0)

    # Replication lag rises slightly with write load and drifts upward
    db_replication_lag_ms = (
        5 + 15 * load + drift * 8 + shared_noise * 2 + rng(n_points) * 1.0
    ).clip(0)

    db_transaction_rate = (
        50 + 300 * load + drift * 5 + shared_noise * 10 + rng(n_points) * 4
    ).clip(0)

    # ── Hardware / Node Health ────────────────────────────────
    # Baseline near-zero — discrete events, Poisson-like
    node_not_ready_count = (
        np.abs(rng(n_points)) * 0.05 + drift * 0.02
    ).clip(0)

    node_disk_pressure_count = (
        np.abs(rng(n_points)) * 0.03 + drift * 0.03
    ).clip(0)

    node_memory_pressure_count = (
        np.abs(rng(n_points)) * 0.03 + drift * 0.02
    ).clip(0)

    hardware_error_log_rate = (
        0.05 + drift * 0.1 + np.abs(rng(n_points)) * 0.04
    ).clip(0)

    node_network_rx_errors = (
        0.1 + 0.5 * load * drift + np.abs(rng(n_points)) * 0.05
    ).clip(0)

    timestamps = pd.date_range(
        start="2024-01-01 00:00:00",
        periods=n_points,
        freq=f"{interval_min}min",
    )

    return pd.DataFrame({
        "timestamp":                         timestamps,
        # Telecom
        "latency_ms":                        latency_ms,
        "throughput_mbps":                   throughput_mbps,
        "call_success_rate":                 call_success_rate,
        "bearer_establishment_success_rate": bearer_establishment_success_rate,
        "complaint_rate":                    complaint_rate,
        "packet_loss_rate":                  packet_loss_rate,
        "jitter_ms":                         jitter_ms,
        "handover_success_rate":             handover_success_rate,
        # K8s Compute
        "node_cpu_utilization":              node_cpu_utilization,
        "node_memory_utilization":           node_memory_utilization,
        "pod_restart_count":                 pod_restart_count,
        "pod_cpu_throttling_rate":           pod_cpu_throttling_rate,
        # Cloud Storage
        "node_disk_read_mbps":               node_disk_read_mbps,
        "node_disk_write_mbps":              node_disk_write_mbps,
        "storage_read_latency_ms":           storage_read_latency_ms,
        "storage_write_latency_ms":          storage_write_latency_ms,
        "storage_utilization_pct":           storage_utilization_pct,
        "storage_iops":                      storage_iops,
        # Istio
        "istio_request_rate":                istio_request_rate,
        "istio_error_rate":                  istio_error_rate,
        "istio_p99_latency_ms":              istio_p99_latency_ms,
        "istio_retry_rate":                  istio_retry_rate,
        # Application
        "app_response_time_ms":              app_response_time_ms,
        "app_error_rate":                    app_error_rate,
        "app_request_rate":                  app_request_rate,
        "app_active_connections":            app_active_connections,
        "app_5xx_rate":                      app_5xx_rate,
        # Database
        "db_query_latency_ms":               db_query_latency_ms,
        "db_connection_pool_utilization":    db_connection_pool_utilization,
        "db_slow_query_rate":                db_slow_query_rate,
        "db_replication_lag_ms":             db_replication_lag_ms,
        "db_transaction_rate":               db_transaction_rate,
        # Hardware Health
        "node_not_ready_count":              node_not_ready_count,
        "node_disk_pressure_count":          node_disk_pressure_count,
        "node_memory_pressure_count":        node_memory_pressure_count,
        "hardware_error_log_rate":           hardware_error_log_rate,
        "node_network_rx_errors":            node_network_rx_errors,
    })


# ─────────────────────────────────────────────────────────────
# Anomaly catalogue — 13 types across all layers
# ─────────────────────────────────────────────────────────────

ANOMALY_TYPES = {
    # ── Telecom ───────────────────────────────────────────────
    "latency_spike": {
        "desc": "Routing loop or DDoS — sudden latency surge + packet loss",
        "duration_steps": 30,    # 30 min at 1-min intervals
    },
    "throughput_drop": {
        "desc": "Link failure — throughput collapses, jitter rises",
        "duration_steps": 60,    # 60 min
    },
    "call_quality_degradation": {
        "desc": "Codec issue — call success and bearer rates drop",
        "duration_steps": 40,    # 40 min
    },
    "complaint_surge": {
        "desc": "Service outage visible to users — complaints explode",
        "duration_steps": 50,    # 50 min
    },
    "network_congestion": {
        "desc": "Full congestion — all telecom KPIs degraded + compute impact",
        "duration_steps": 90,    # 90 min
    },
    # ── K8s Compute ──────────────────────────────────────────
    "oom_kill": {
        "desc": "OOM kill event — memory exhaustion, pod restarts surge",
        "duration_steps": 70,    # 70 min
    },
    "cpu_throttling_event": {
        "desc": "CPU resource starvation — throttling spike, latency chain",
        "duration_steps": 90,    # 90 min
    },
    # ── Cloud Storage ─────────────────────────────────────────
    "storage_saturation": {
        "desc": "Storage full — write latency spikes, IOPS degrade, disk pressure",
        "duration_steps": 120,   # 2 hours
    },
    # ── Istio Service Mesh ────────────────────────────────────
    "istio_cascade_failure": {
        "desc": "Sidecar proxy cascade — error rate + retry storm + P99 blow-up",
        "duration_steps": 75,    # 75 min
    },
    # ── Application ───────────────────────────────────────────
    "app_error_storm": {
        "desc": "Application bug / bad deploy — 5xx storm, response time spikes",
        "duration_steps": 100,   # 100 min
    },
    # ── Database ──────────────────────────────────────────────
    "db_connection_exhaustion": {
        "desc": "DB connection pool full — queries slow, app response degrades",
        "duration_steps": 80,    # 80 min
    },
    "db_replication_failure": {
        "desc": "DB replica falling behind — replication lag grows, slow queries spike",
        "duration_steps": 110,   # 110 min
    },
    # ── Hardware ──────────────────────────────────────────────
    "node_hardware_failure": {
        "desc": "Node goes NotReady — hardware errors, NIC errors, pod restarts",
        "duration_steps": 150,   # 2.5 hours
    },
}


def _inject(df, idx, col, multiplier):
    df.loc[idx, col] = df.loc[idx, col] * multiplier


def inject_anomalies(df: pd.DataFrame, anomaly_ratio: float = 0.05):
    """
    Inject realistic multi-layer anomaly windows into the DataFrame.
    Returns (df_with_anomalies, labels_df).
    """
    df = df.copy()
    n  = len(df)
    labels            = np.zeros(n, dtype=int)
    anomaly_types_col = ["normal"] * n

    n_windows = max(1, int(n * anomaly_ratio / 15))
    starts    = np.random.choice(
        range(Config.WINDOW_SIZE, n - 40), size=n_windows, replace=False
    )

    for start in starts:
        atype_name = np.random.choice(list(ANOMALY_TYPES.keys()))
        dur = ANOMALY_TYPES[atype_name]["duration_steps"]
        end = min(start + dur, n)
        idx = slice(start, end)

        # ── Telecom ───────────────────────────────────────────
        if atype_name == "latency_spike":
            _inject(df, idx, "latency_ms",        4.5)
            _inject(df, idx, "jitter_ms",          3.0)
            _inject(df, idx, "packet_loss_rate",   5.0)
            _inject(df, idx, "complaint_rate",     2.5)

        elif atype_name == "throughput_drop":
            _inject(df, idx, "throughput_mbps",   0.25)
            _inject(df, idx, "jitter_ms",          2.5)
            _inject(df, idx, "complaint_rate",     3.0)
            _inject(df, idx, "app_request_rate",   0.6)

        elif atype_name == "call_quality_degradation":
            _inject(df, idx, "call_success_rate",                 0.82)
            _inject(df, idx, "bearer_establishment_success_rate", 0.80)
            _inject(df, idx, "jitter_ms",                          3.5)
            _inject(df, idx, "complaint_rate",                     4.0)

        elif atype_name == "complaint_surge":
            _inject(df, idx, "complaint_rate",     6.0)
            _inject(df, idx, "app_error_rate",     3.0)
            _inject(df, idx, "app_5xx_rate",       4.0)

        elif atype_name == "network_congestion":
            # Full cross-layer event
            _inject(df, idx, "latency_ms",                        3.0)
            _inject(df, idx, "throughput_mbps",                   0.4)
            _inject(df, idx, "call_success_rate",                 0.88)
            _inject(df, idx, "bearer_establishment_success_rate", 0.85)
            _inject(df, idx, "complaint_rate",                    5.0)
            _inject(df, idx, "packet_loss_rate",                  4.0)
            _inject(df, idx, "node_cpu_utilization",              1.4)
            _inject(df, idx, "istio_p99_latency_ms",              2.5)
            _inject(df, idx, "app_response_time_ms",              2.0)
            _inject(df, idx, "node_network_rx_errors",            8.0)

        # ── K8s Compute ───────────────────────────────────────
        elif atype_name == "oom_kill":
            df.loc[idx, "node_memory_utilization"]   = np.minimum(
                df.loc[idx, "node_memory_utilization"] * 1.5 + 40, 99.5
            )
            _inject(df, idx, "pod_restart_count",            12.0)
            _inject(df, idx, "node_memory_pressure_count",   20.0)
            _inject(df, idx, "node_cpu_utilization",          1.3)
            _inject(df, idx, "app_response_time_ms",          1.8)
            _inject(df, idx, "app_error_rate",                2.5)
            _inject(df, idx, "hardware_error_log_rate",       5.0)

        elif atype_name == "cpu_throttling_event":
            df.loc[idx, "node_cpu_utilization"]      = np.minimum(
                df.loc[idx, "node_cpu_utilization"] * 1.6 + 30, 98)
            _inject(df, idx, "pod_cpu_throttling_rate",      5.0)
            _inject(df, idx, "istio_p99_latency_ms",          3.0)
            _inject(df, idx, "app_response_time_ms",          2.5)
            _inject(df, idx, "db_query_latency_ms",           1.8)
            _inject(df, idx, "latency_ms",                    1.6)

        # ── Cloud Storage ─────────────────────────────────────
        elif atype_name == "storage_saturation":
            df.loc[idx, "storage_utilization_pct"]  = np.minimum(
                df.loc[idx, "storage_utilization_pct"] + 40, 99.0)
            _inject(df, idx, "storage_write_latency_ms",     10.0)
            _inject(df, idx, "storage_read_latency_ms",       5.0)
            _inject(df, idx, "storage_iops",                  0.3)
            _inject(df, idx, "node_disk_write_mbps",          0.4)
            _inject(df, idx, "node_disk_pressure_count",     15.0)
            _inject(df, idx, "db_query_latency_ms",           2.5)
            _inject(df, idx, "db_slow_query_rate",            4.0)

        # ── Istio ─────────────────────────────────────────────
        elif atype_name == "istio_cascade_failure":
            _inject(df, idx, "istio_error_rate",             10.0)
            _inject(df, idx, "istio_retry_rate",              8.0)
            _inject(df, idx, "istio_p99_latency_ms",          6.0)
            _inject(df, idx, "app_error_rate",                5.0)
            _inject(df, idx, "app_5xx_rate",                  8.0)
            _inject(df, idx, "call_success_rate",             0.90)
            _inject(df, idx, "bearer_establishment_success_rate", 0.90)
            _inject(df, idx, "app_response_time_ms",          3.0)

        # ── Application ───────────────────────────────────────
        elif atype_name == "app_error_storm":
            _inject(df, idx, "app_error_rate",               15.0)
            _inject(df, idx, "app_5xx_rate",                 20.0)
            _inject(df, idx, "app_response_time_ms",          4.0)
            _inject(df, idx, "complaint_rate",                5.0)
            _inject(df, idx, "db_slow_query_rate",            3.0)
            _inject(df, idx, "istio_error_rate",              4.0)
            _inject(df, idx, "istio_retry_rate",              3.0)

        # ── Database ──────────────────────────────────────────
        elif atype_name == "db_connection_exhaustion":
            df.loc[idx, "db_connection_pool_utilization"] = np.minimum(
                df.loc[idx, "db_connection_pool_utilization"] + 55, 99.0)
            _inject(df, idx, "db_query_latency_ms",          6.0)
            _inject(df, idx, "db_slow_query_rate",            8.0)
            _inject(df, idx, "app_response_time_ms",          3.5)
            _inject(df, idx, "app_error_rate",                4.0)
            _inject(df, idx, "app_5xx_rate",                  5.0)
            _inject(df, idx, "node_cpu_utilization",          1.3)

        elif atype_name == "db_replication_failure":
            _inject(df, idx, "db_replication_lag_ms",        25.0)
            _inject(df, idx, "db_slow_query_rate",            6.0)
            _inject(df, idx, "db_query_latency_ms",           3.0)
            _inject(df, idx, "db_transaction_rate",           0.5)
            _inject(df, idx, "app_response_time_ms",          2.0)
            _inject(df, idx, "app_error_rate",                2.5)

        # ── Hardware ──────────────────────────────────────────
        elif atype_name == "node_hardware_failure":
            df.loc[idx, "node_not_ready_count"]        += np.random.randint(1, 4)
            _inject(df, idx, "hardware_error_log_rate",      30.0)
            _inject(df, idx, "node_network_rx_errors",       20.0)
            _inject(df, idx, "pod_restart_count",            15.0)
            _inject(df, idx, "node_cpu_utilization",          1.5)
            _inject(df, idx, "node_memory_utilization",       1.3)
            _inject(df, idx, "node_not_ready_count",          8.0)
            _inject(df, idx, "istio_error_rate",              3.0)
            _inject(df, idx, "app_error_rate",                3.0)
            _inject(df, idx, "throughput_mbps",               0.7)

        labels[start:end] = 1
        for i in range(start, end):
            anomaly_types_col[i] = atype_name

    # Clip physical bounds after all injections — vectorised per group
    pct_cols = [
        "call_success_rate", "bearer_establishment_success_rate",
        "handover_success_rate", "node_cpu_utilization",
        "node_memory_utilization", "pod_cpu_throttling_rate",
        "storage_utilization_pct", "db_connection_pool_utilization",
        "istio_error_rate", "app_error_rate", "packet_loss_rate",
    ]
    df[pct_cols] = df[pct_cols].clip(0, 100)

    non_neg = [
        "latency_ms", "throughput_mbps", "complaint_rate", "jitter_ms",
        "packet_loss_rate", "pod_restart_count", "node_disk_read_mbps",
        "node_disk_write_mbps", "storage_read_latency_ms",
        "storage_write_latency_ms", "storage_iops",
        "istio_request_rate", "istio_p99_latency_ms", "istio_retry_rate",
        "app_response_time_ms", "app_request_rate", "app_active_connections",
        "app_5xx_rate", "db_query_latency_ms", "db_slow_query_rate",
        "db_replication_lag_ms", "db_transaction_rate",
        "node_not_ready_count", "node_disk_pressure_count",
        "node_memory_pressure_count", "hardware_error_log_rate",
        "node_network_rx_errors",
    ]
    df[non_neg] = df[non_neg].clip(lower=0)

    labels_df = pd.DataFrame({
        "timestamp":    df["timestamp"],
        "is_anomaly":   labels,
        "anomaly_type": anomaly_types_col,
    })
    return df, labels_df


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def generate_dataset(save: bool = True):
    cfg            = Config()
    np.random.seed(cfg.RANDOM_SEED)
    points_per_day = (24 * 60) // cfg.FEED_INTERVAL_MIN
    n_points       = cfg.HISTORY_DAYS * points_per_day

    print(f"Generating {n_points:,} data points "
          f"({cfg.HISTORY_DAYS} days × {points_per_day} pts/day, "
          f"{len(cfg.KPI_COLUMNS)} features)...")

    df = generate_base_signals(n_points, cfg.FEED_INTERVAL_MIN)

    # Inject anomalies only into the test split
    train_end        = int(n_points * cfg.TRAIN_RATIO)
    df_test, lbls    = inject_anomalies(
        df.iloc[train_end:].reset_index(drop=True), cfg.ANOMALY_RATIO
    )
    df_full = pd.concat([df.iloc[:train_end], df_test], ignore_index=True)

    train_labels = pd.DataFrame({
        "timestamp":    df.iloc[:train_end]["timestamp"],
        "is_anomaly":   0,
        "anomaly_type": "normal",
    })
    labels_full = pd.concat([train_labels, lbls], ignore_index=True)

    if save:
        os.makedirs("data", exist_ok=True)
        df_full.to_csv(cfg.DATA_PATH, index=False)
        labels_full.to_csv(cfg.ANOMALY_PATH, index=False)
        print(f"Saved telemetry  -> {cfg.DATA_PATH}")
        print(f"Saved labels     -> {cfg.ANOMALY_PATH}")
        pct = labels_full["is_anomaly"].mean() * 100
        print(f"Anomaly rate     : {pct:.2f}%")
        print("\nAnomaly type breakdown:")
        print(labels_full[labels_full["is_anomaly"] == 1]["anomaly_type"].value_counts())

    return df_full, labels_full


if __name__ == "__main__":
    df, labels = generate_dataset(save=True)
    print("\nSample (tail):")
    print(df[["timestamp", "latency_ms", "node_cpu_utilization",
              "app_error_rate", "db_replication_lag_ms",
              "node_not_ready_count"]].tail(5).to_string(index=False))
