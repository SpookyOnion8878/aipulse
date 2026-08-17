from __future__ import annotations

import numpy as np

from src.core.monitoring import InferenceMonitor, population_stability_index


def test_psi_is_zero_for_identical_samples() -> None:
    values = np.linspace(0, 1, 100)
    assert population_stability_index(values, values) == 0.0


def test_inference_summary_has_p95_and_error_rate() -> None:
    monitor = InferenceMonitor()
    monitor.record_inference(10, True)
    monitor.record_inference(30, False)
    result = monitor.summary()
    assert result["error_rate"] == 0.5
    assert result["latency_ms_p95"] > result["latency_ms_avg"]
