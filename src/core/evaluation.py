"""Validated model evaluation and release gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

Array = npt.NDArray[np.float64] | npt.NDArray[np.int64]


def _as_array(values: Any) -> npt.NDArray[Any]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional target array")
    if len(array) == 0:
        raise ValueError("Target array must not be empty")
    return array


class EvaluationMetrics:
    """Compute classification and regression metrics through sklearn."""

    def classification_report(
        self,
        y_true: Any,
        y_pred: Any,
        y_score: Any | None = None,
    ) -> dict[str, float | str | None]:
        truth = _as_array(y_true)
        prediction = _as_array(y_pred)
        if truth.shape != prediction.shape:
            raise ValueError("y_true and y_pred must have the same shape")

        labels = np.unique(truth)
        average = "binary" if len(labels) == 2 else "weighted"
        report: dict[str, float | str | None] = {
            "accuracy": float(accuracy_score(truth, prediction)),
            "precision": float(
                precision_score(truth, prediction, average=average, zero_division=0)
            ),
            "recall": float(recall_score(truth, prediction, average=average, zero_division=0)),
            "f1_score": float(f1_score(truth, prediction, average=average, zero_division=0)),
            "auc_roc": None,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if y_score is not None and len(labels) == 2:
            score = _as_array(y_score)
            if score.shape != truth.shape:
                raise ValueError("y_score and y_true must have the same shape")
            report["auc_roc"] = float(roc_auc_score(truth, score))
        return report

    def regression_report(self, y_true: Any, y_pred: Any) -> dict[str, float | str]:
        truth = _as_array(y_true)
        prediction = _as_array(y_pred)
        if truth.shape != prediction.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        mse = float(mean_squared_error(truth, prediction))
        return {
            "mae": float(mean_absolute_error(truth, prediction)),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r_squared": float(r2_score(truth, prediction)),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def calculate_auc_roc(self, y_true: Any, y_score: Any) -> float:
        truth = _as_array(y_true)
        score = _as_array(y_score)
        if len(np.unique(truth)) != 2:
            raise ValueError("AUC-ROC requires exactly two classes")
        if truth.shape != score.shape:
            raise ValueError("y_true and y_score must have the same shape")
        return float(roc_auc_score(truth, score))

    def generate_report(
        self,
        y_true: Any,
        y_pred: Any,
        y_score: Any | None = None,
    ) -> dict[str, float | str | None]:
        return self.classification_report(y_true, y_pred, y_score)


@dataclass(frozen=True, slots=True)
class ModelValidator:
    """Apply explicit model release thresholds."""

    thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "accuracy": 0.80,
            "f1_score": 0.70,
            "auc_roc": 0.85,
        }
    )

    def validate_model(self, metrics: dict[str, Any]) -> dict[str, Any]:
        failures: list[str] = []
        warnings: list[str] = []
        for metric_name, minimum in self.thresholds.items():
            value = metrics.get(metric_name)
            if value is None:
                warnings.append(f"{metric_name} was not assessed")
                continue
            if not isinstance(value, (int, float)):
                failures.append(f"{metric_name} is not numeric")
                continue
            if float(value) < minimum:
                failures.append(f"{metric_name}={float(value):.4f} is below {minimum:.4f}")
        return {
            "passed": not failures,
            "failures": failures,
            "warnings": warnings,
            "thresholds": dict(self.thresholds),
            "timestamp": datetime.now(UTC).isoformat(),
        }
