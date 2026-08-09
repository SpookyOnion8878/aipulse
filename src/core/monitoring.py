"""
Monitoring Module — Step 1: Core Skills (ML & Deep Learning)

AIPulse Enterprise AI Platform — Monitoring Layer
Handles model monitoring, drift detection, and performance tracking.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs', 'monitoring_report.json')),
    ],
)
logger = logging.getLogger("aipulse.monitoring")


class MonitoringMetrics:
    """Track monitoring metrics."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.metrics_history: List[Dict] = []
        self.drift_alerts: List[Dict] = []

    def track_metric(self, metric_name: str, value: float, metadata: Optional[Dict] = None) -> Dict:
        """Track a single metric."""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "metric_name": metric_name,
            "value": value,
            "metadata": metadata or {},
        }
        self.metrics_history.append(metric)
        logger.info(f"Tracked metric '{metric_name}': {value}")
        return metric

    def compute_rolling_average(self, window: int = 7) -> List[Dict]:
        """Compute rolling average for metrics."""
        logger.info(f"Computing rolling average (window={window})")
        results = []
        for metric_name in set(m["metric_name"] for m in self.metrics_history):
            values = [m["value"] for m in self.metrics_history if m["metric_name"] == metric_name]
            if len(values) >= window:
                rolling = []
                for i in range(len(values) - window + 1):
                    rolling.append({
                        "timestamp": self.metrics_history[i + window - 1]["timestamp"],
                        "value": np.mean(values[i:i + window]),
                        "window_size": window,
                    })
                results.extend(rolling)
        return results

    def detect_drift(self, current_history: Optional[List[Dict]] = None, threshold: float = 0.2) -> List[Dict]:
        """Detect data drift using PSI (Population Stability Index)."""
        if current_history is None:
            current_history = self.metrics_history

        drift_alerts = []
        for metric_name in set(m["metric_name"] for m in current_history):
            current_values = [m["value"] for m in current_history if m["metric_name"] == metric_name]
            if len(current_values) < 2:
                continue

            # Compare with baseline (first 70% of data)
            baseline = current_values[:len(current_values) // 3]
            if len(baseline) < 3:
                continue

            # Simple drift detection (mean shift)
            baseline_mean = np.mean(baseline)
            current_mean = np.mean(current_values)
            psie = np.sqrt(np.mean(((current_values - baseline_mean) ** 2) / baseline_mean))

            if psie > threshold:
                drift_alerts.append({
                    "metric_name": metric_name,
                    "psie": round(psie, 4),
                    "baseline_mean": round(baseline_mean, 4),
                    "current_mean": round(current_mean, 4),
                    "threshold": threshold,
                    "timestamp": datetime.now().isoformat(),
                })
                logger.warning(f"Drift detected for '{metric_name}': PSI={psie:.4f}")

        return drift_alerts

    def compute_model_health(self) -> Dict[str, Any]:
        """Compute model health score."""
        total = len(self.metrics_history)
        if total == 0:
            return {"health_score": 0.0, "status": "no_data", "reason": "No metrics tracked"}

        # Count positive metrics
        total = len(self.metrics_history)
        avg = np.mean([m["value"] for m in self.metrics_history])
        std = np.std([m["value"] for m in self.metrics_history])

        # Health score based on consistency
        consistency = max(0, 1 - std / avg) if avg > 0 else 0
        health_score = round(consistency * 100, 2)

        status = "healthy" if health_score > 80 else ("degraded" if health_score > 50 else "critical")

        return {
            "health_score": health_score,
            "status": status,
            "avg_value": round(avg, 4),
            "std_value": round(std, 4),
            "data_points": total,
            "timestamp": datetime.now().isoformat(),
        }

    def generate_report(self, output_path: str = "logs/monitoring_report.json"):
        """Generate monitoring report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics_history": self.metrics_history,
            "drift_alerts": self.drift_alerts,
            "health_score": self.compute_model_health(),
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Monitoring report saved to {output_path}")
        return report


class InferenceMonitor:
    """Monitor inference performance."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.inference_latency: List[Dict] = []
        self.error_count = 0
        self.total_requests = 0

    def record_inference(self, latency_ms: float, success: bool, metadata: Optional[Dict] = None) -> None:
        """Record inference metrics."""
        self.total_requests += 1
        if not success:
            self.error_count += 1
        self.inference_latency.append({
            "timestamp": datetime.now().isoformat(),
            "latency_ms": latency_ms,
            "success": success,
            "metadata": metadata or {},
        })

        # Log every 100 requests
        if self.total_requests % 100 == 0:
            self._log_summary()

    def _log_summary(self) -> None:
        """Log inference summary."""
        total = len(self.inference_latency)
        if total == 0:
            return
        avg_latency = np.mean([m["latency_ms"] for m in self.inference_latency])
        error_rate = self.error_count / total
        logger.info(f"Inference Summary: {total} requests, avg latency: {avg_latency:.2f}ms, error rate: {error_rate:.4f}")


# Entry point
def main():
    """Run monitoring module."""
    monitor = MonitoringMetrics()

    # Simulate tracking metrics
    for i in range(50):
        monitor.track_metric("accuracy", 0.85 + np.random.rand() * 0.1, {"iteration": i})

    # Detect drift
    drift_alerts = monitor.detect_drift()

    # Generate report
    report = monitor.generate_report("logs/monitoring_report.json")
    print(f"Monitoring report generated with {len(drift_alerts)} drift alerts")


if __name__ == "__main__":
    main()
