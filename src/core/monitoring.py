"""Inference telemetry and statistically meaningful drift checks."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt


def population_stability_index(
    reference: npt.ArrayLike,
    current: npt.ArrayLike,
    *,
    bins: int = 10,
) -> float:
    """Calculate PSI using quantile buckets learned from reference values."""

    reference_array = np.asarray(reference, dtype=float)
    current_array = np.asarray(current, dtype=float)
    reference_array = reference_array[np.isfinite(reference_array)]
    current_array = current_array[np.isfinite(current_array)]
    if len(reference_array) < bins or len(current_array) < bins:
        raise ValueError(f"At least {bins} finite values are required in each sample")
    edges = np.unique(np.quantile(reference_array, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_histogram = np.histogram(reference_array, bins=edges)[0] / len(reference_array)
    current_histogram = np.histogram(current_array, bins=edges)[0] / len(current_array)
    epsilon = 1e-6
    reference_ratio = np.clip(reference_histogram, epsilon, None)
    current_ratio = np.clip(current_histogram, epsilon, None)
    return float(
        np.sum((current_ratio - reference_ratio) * np.log(current_ratio / reference_ratio))
    )


@dataclass(frozen=True, slots=True)
class DriftAlert:
    metric_name: str
    psi: float
    threshold: float
    timestamp: str


class MonitoringMetrics:
    """Keep a bounded in-process metric window for the local deployment."""

    def __init__(self, max_points_per_metric: int = 10_000) -> None:
        self._values: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=max_points_per_metric)
        )
        self.drift_alerts: list[DriftAlert] = []

    def track_metric(self, metric_name: str, value: float) -> None:
        if not np.isfinite(value):
            raise ValueError("Metric value must be finite")
        self._values[metric_name].append(float(value))

    def detect_drift(
        self,
        metric_name: str,
        reference: npt.ArrayLike,
        *,
        threshold: float = 0.2,
    ) -> DriftAlert | None:
        current = list(self._values.get(metric_name, ()))
        psi = population_stability_index(reference, current)
        if psi <= threshold:
            return None
        alert = DriftAlert(metric_name, psi, threshold, datetime.now(UTC).isoformat())
        self.drift_alerts.append(alert)
        return alert


class InferenceMonitor:
    """Aggregate bounded latency and error telemetry."""

    def __init__(self, max_points: int = 10_000) -> None:
        self._latencies: deque[float] = deque(maxlen=max_points)
        self.total_requests = 0
        self.error_count = 0

    def record_inference(self, latency_ms: float, success: bool) -> None:
        if latency_ms < 0 or not np.isfinite(latency_ms):
            raise ValueError("latency_ms must be a finite non-negative value")
        self.total_requests += 1
        self.error_count += int(not success)
        self._latencies.append(float(latency_ms))

    def summary(self) -> dict[str, Any]:
        if not self._latencies:
            return {"status": "no_data", "total_requests": self.total_requests}
        values = np.asarray(self._latencies)
        return {
            "status": "ok",
            "total_requests": self.total_requests,
            "window_requests": len(values),
            "error_rate": self.error_count / self.total_requests,
            "latency_ms_avg": float(np.mean(values)),
            "latency_ms_p95": float(np.percentile(values, 95)),
            "timestamp": datetime.now(UTC).isoformat(),
        }
