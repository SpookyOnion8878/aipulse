from __future__ import annotations

import numpy as np
import pytest

from src.core.evaluation import EvaluationMetrics, ModelValidator


def test_auc_roc_uses_probability_scores() -> None:
    metrics = EvaluationMetrics()
    auc = metrics.calculate_auc_roc(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.35, 0.8]),
    )
    assert auc == pytest.approx(0.75)


def test_auc_roc_rejects_single_class() -> None:
    with pytest.raises(ValueError, match="exactly two classes"):
        EvaluationMetrics().calculate_auc_roc(np.zeros(4), np.linspace(0.1, 0.9, 4))


def test_validator_marks_unassessed_auc_as_warning() -> None:
    result = ModelValidator().validate_model({"accuracy": 0.9, "f1_score": 0.8, "auc_roc": None})
    assert result["passed"] is True
    assert result["warnings"] == ["auc_roc was not assessed"]
