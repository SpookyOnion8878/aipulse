"""Fairness, compliance evidence, and data-integrity checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import confusion_matrix

from .audit import AuditTrail


@dataclass(frozen=True, slots=True)
class FairnessResult:
    status: str
    metrics: dict[str, float]
    groups: list[str]
    warnings: list[str]
    timestamp: str


class BiasDetector:
    """Measure binary classification disparities across supplied groups."""

    def __init__(self, disparity_threshold: float = 0.1) -> None:
        if not 0 < disparity_threshold < 1:
            raise ValueError("disparity_threshold must be between zero and one")
        self.disparity_threshold = disparity_threshold

    @staticmethod
    def _rates(
        truth: npt.NDArray[Any],
        prediction: npt.NDArray[Any],
        groups: npt.NDArray[Any],
    ) -> dict[str, dict[str, float]]:
        rates: dict[str, dict[str, float]] = {}
        for raw_group in np.unique(groups):
            mask = groups == raw_group
            group_truth = truth[mask]
            group_prediction = prediction[mask]
            if len(group_truth) == 0:
                continue
            tn, fp, fn, tp = confusion_matrix(group_truth, group_prediction, labels=[0, 1]).ravel()
            rates[str(raw_group)] = {
                "positive_rate": float(np.mean(group_prediction == 1)),
                "true_positive_rate": float(tp / (tp + fn)) if tp + fn else 0.0,
                "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
                "positive_predictive_value": float(tp / (tp + fp)) if tp + fp else 0.0,
            }
        return rates

    def run_bias_audit(
        self,
        y_true: npt.ArrayLike,
        y_pred: npt.ArrayLike,
        sensitive_group: npt.ArrayLike | pd.Series | None,
    ) -> dict[str, Any]:
        if sensitive_group is None:
            return asdict(
                FairnessResult(
                    status="not_assessed",
                    metrics={},
                    groups=[],
                    warnings=["Sensitive-group data was not supplied"],
                    timestamp=datetime.now(UTC).isoformat(),
                )
            )
        truth = np.asarray(y_true)
        prediction = np.asarray(y_pred)
        groups = np.asarray(sensitive_group)
        if truth.ndim != 1 or truth.shape != prediction.shape or truth.shape != groups.shape:
            raise ValueError(
                "Targets, predictions, and groups must be aligned one-dimensional arrays"
            )
        if not set(np.unique(truth)).issubset({0, 1}):
            raise ValueError("Fairness audit currently supports binary targets only")
        rates = self._rates(truth, prediction, groups)
        if len(rates) < 2:
            return asdict(
                FairnessResult(
                    status="not_assessed",
                    metrics={},
                    groups=list(rates),
                    warnings=["At least two sensitive groups are required"],
                    timestamp=datetime.now(UTC).isoformat(),
                )
            )

        def disparity(name: str) -> float:
            values = [group_rates[name] for group_rates in rates.values()]
            return max(values) - min(values)

        metrics = {
            "demographic_parity_difference": disparity("positive_rate"),
            "equal_opportunity_difference": disparity("true_positive_rate"),
            "false_positive_rate_difference": disparity("false_positive_rate"),
            "predictive_parity_difference": disparity("positive_predictive_value"),
        }
        metrics["equalized_odds_difference"] = max(
            metrics["equal_opportunity_difference"],
            metrics["false_positive_rate_difference"],
        )
        warnings = [
            f"{name}={value:.4f} exceeds {self.disparity_threshold:.4f}"
            for name, value in metrics.items()
            if value > self.disparity_threshold
        ]
        return asdict(
            FairnessResult(
                status="warning" if warnings else "passed",
                metrics=metrics,
                groups=list(rates),
                warnings=warnings,
                timestamp=datetime.now(UTC).isoformat(),
            )
        )


class ComplianceChecker:
    """Evaluate explicit evidence instead of assuming compliance."""

    REQUIRED_CONTROLS: ClassVar[frozenset[str]] = frozenset(
        {
            "data_minimization",
            "deletion_workflow",
            "lawful_basis",
            "risk_assessment",
            "human_oversight",
            "security_testing",
            "audit_retention",
        }
    )

    def check_compliance(self, evidence: dict[str, bool | None]) -> dict[str, Any]:
        checks: dict[str, str] = {}
        for control in sorted(self.REQUIRED_CONTROLS):
            value = evidence.get(control)
            checks[control] = (
                "passed" if value is True else "failed" if value is False else "not_assessed"
            )
        failed = [name for name, result in checks.items() if result == "failed"]
        missing = [name for name, result in checks.items() if result == "not_assessed"]
        status = "failed" if failed else "not_assessed" if missing else "passed"
        return {
            "status": status,
            "checks": checks,
            "failed_controls": failed,
            "missing_evidence": missing,
            "timestamp": datetime.now(UTC).isoformat(),
        }


class ModelPoisoningDetector:
    """Compare numeric training features with a trusted reference dataset."""

    def detect(
        self,
        input_data: pd.DataFrame,
        target: pd.Series,
        reference_data: pd.DataFrame | None = None,
        *,
        standardized_mean_threshold: float = 0.5,
    ) -> dict[str, Any]:
        if len(input_data) != len(target):
            raise ValueError("input_data and target must be aligned")
        if reference_data is None:
            return {
                "status": "not_assessed",
                "anomalies": [],
                "warnings": ["Trusted reference data was not supplied"],
            }
        common = sorted(
            set(input_data.select_dtypes(include="number").columns)
            & set(reference_data.select_dtypes(include="number").columns)
        )
        if not common:
            return {
                "status": "not_assessed",
                "anomalies": [],
                "warnings": ["No common numeric features were found"],
            }
        anomalies: list[dict[str, float | str]] = []
        for column in common:
            reference_std = float(reference_data[column].std())
            if not np.isfinite(reference_std) or reference_std == 0:
                continue
            shift = abs(float(input_data[column].mean()) - float(reference_data[column].mean()))
            standardized_shift = shift / reference_std
            if standardized_shift > standardized_mean_threshold:
                anomalies.append(
                    {
                        "feature": column,
                        "standardized_mean_shift": standardized_shift,
                    }
                )
        return {
            "status": "suspicious" if anomalies else "passed",
            "anomalies": anomalies,
            "warnings": [],
            "timestamp": datetime.now(UTC).isoformat(),
        }


__all__ = ["AuditTrail", "BiasDetector", "ComplianceChecker", "ModelPoisoningDetector"]
