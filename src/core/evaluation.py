"""
Model Evaluation Module — Step 1: Core Skills (ML & Deep Learning)

AIPulse Enterprise AI Platform — Evaluation Layer
Handles model evaluation metrics, validation, and performance monitoring.
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

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, 'evaluation_report.log')),
    ],
)
logger = logging.getLogger("aipulse.evaluation")


class EvaluationMetrics:
    """Calculate and track evaluation metrics."""

    def __init__(self, config: Optional[Dict] = None):
        self.metrics = {}
        self.config = config or {}

    def calculate_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate accuracy."""
        correct = (y_pred == y_true.values).sum()
        return correct / len(y_true)

    def calculate_precision(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate precision."""
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def calculate_recall(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate recall."""
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def calculate_f1(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate F1 score."""
        p = self.calculate_precision(y_true, y_pred)
        r = self.calculate_recall(y_true, y_pred)
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    def calculate_auc_roc(self, y_true: pd.Series, y_prob: np.ndarray) -> float:
        """Calculate AUC-ROC."""
        # Sort by probability
        indices = np.argsort(-y_prob)
        y_true_sorted = y_true.values[indices]
        fpr = np.empty(len(y_true_sorted))
        tpr = np.empty(len(y_true_sorted))

        tp = 0
        fp = 0
        for i in range(len(y_true_sorted)):
            if y_true_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
            fpr[fp] = fp / len(y_true_sorted)
            tpr[fp] = tp / fp if fp > 0 else 0.0

        fpr = np.concatenate([[0], fpr])
        tpr = np.concatenate([[0], tpr])

        # AUC calculation using trapezoidal rule
        auc = 0.0
        for i in range(len(fpr) - 1):
            auc += (fpr[i + 1] - fpr[i]) * (tpr[i] + tpr[i + 1]) / 2
        return auc

    def calculate_mae(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(y_true.values - y_pred))

    def calculate_mse(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        return np.mean((y_true.values - y_pred) ** 2)

    def calculate_rmse(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(self.calculate_mse(y_true, y_pred))

    def calculate_r_squared(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate R-squared."""
        y_true_mean = y_true.values.mean()
        ss_total = np.sum((y_true.values - y_true_mean) ** 2)
        ss_residual = np.sum((y_true.values - y_pred) ** 2)
        return 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0

    def calculate_bias(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate bias."""
        return np.mean(y_pred - y_true.values)

    def calculate_variance(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate variance."""
        return np.var(y_pred - y_true.values)

    def calculate_mixed_metric(self, y_true: pd.Series, y_pred: np.ndarray,
                               beta: float = 1.0) -> float:
        """Calculate F-beta score."""
        p = self.calculate_precision(y_true, y_pred)
        r = self.calculate_recall(y_true, y_pred)
        return (1 + beta ** 2) * p * r / ((beta ** 2) * p + r) if (beta ** 2 * p + r) > 0 else 0.0

    def generate_report(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        """Generate a full evaluation report."""
        return {
            "accuracy": round(self.calculate_accuracy(y_true, y_pred), 4),
            "precision": round(self.calculate_precision(y_true, y_pred), 4),
            "recall": round(self.calculate_recall(y_true, y_pred), 4),
            "f1_score": round(self.calculate_f1(y_true, y_pred), 4),
            "auc_roc": round(self.calculate_auc_roc(y_true, y_pred), 4),
            "mae": round(self.calculate_mae(y_true, y_pred), 4),
            "mse": round(self.calculate_mse(y_true, y_pred), 4),
            "rmse": round(self.calculate_rmse(y_true, y_pred), 4),
            "r_squared": round(self.calculate_r_squared(y_true, y_pred), 4),
            "bias": round(self.calculate_bias(y_true, y_pred), 4),
            "variance": round(self.calculate_variance(y_true, y_pred), 4),
            "timestamp": datetime.now().isoformat(),
        }

    def compare_with_baseline(self, metrics: Dict, baseline: Dict) -> Dict:
        """Compare current metrics with baseline."""
        comparison = {}
        for key in metrics:
            if key in baseline and baseline[key] is not None:
                comparison[key] = {
                    "current": metrics[key],
                    "baseline": baseline[key],
                    "improvement": metrics[key] - baseline[key] if isinstance(metrics[key], (int, float)) and metrics[key] > baseline[key] else "N/A",
                }
            else:
                comparison[key] = {"current": metrics[key], "baseline": None}
        return comparison

    def log_report(self, report: Dict, output_path: str = "logs/evaluation_report.json"):
        """Log evaluation report to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Evaluation report saved to {output_path}")


class ModelValidator:
    """Validate model performance."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.thresholds = {
            "accuracy": 0.80,
            "f1_score": 0.70,
            "auc_roc": 0.85,
            "mse": 1.0,
        }

    def validate_model(self, metrics: Dict) -> Dict[str, Any]:
        """Validate model against thresholds."""
        results = {
            "passed": True,
            "warnings": [],
            "failures": [],
            "thresholds": self.thresholds,
            "timestamp": datetime.now().isoformat(),
        }

        # Check accuracy
        if metrics.get("accuracy", 0) < self.thresholds.get("accuracy", 0.80):
            results["passed"] = False
            results["failures"].append(f"Accuracy {metrics.get('accuracy', 0):.4f} below threshold {self.thresholds['accuracy']}")

        # Check F1 score
        if metrics.get("f1_score", 0) < self.thresholds.get("f1_score", 0.70):
            results["passed"] = False
            results["failures"].append(f"F1 score {metrics.get('f1_score', 0):.4f} below threshold {self.thresholds['f1_score']}")

        # Check AUC-ROC
        if metrics.get("auc_roc", 0) < self.thresholds.get("auc_roc", 0.85):
            results["passed"] = False
            results["failures"].append(f"AUC-ROC {metrics.get('auc_roc', 0):.4f} below threshold {self.thresholds['auc_roc']}")

        # Check MSE
        if metrics.get("mse", float('inf')) > self.thresholds.get("mse", 1.0):
            results["passed"] = False
            results["failures"].append(f"MSE {metrics.get('mse', 0):.4f} above threshold {self.thresholds['mse']}")

        # Check bias
        bias = metrics.get("bias", 0)
        if abs(bias) > 0.1:
            results["warnings"].append(f"Bias detected: {bias:.4f}")

        # Check variance
        var = metrics.get("variance", 0)
        if var > 0.5:
            results["warnings"].append(f"High variance: {var:.4f}")

        # Check confidence
        confidence = metrics.get("auc_roc", 0)
        if confidence > 0.95 and metrics.get("accuracy", 0) < 0.90:
            results["warnings"].append(f"High AUC-ROC but low accuracy — possible overfitting")

        return results


# Entry point
def main():
    """Run evaluation module."""
    metrics = EvaluationMetrics()
    report = metrics.generate_report(
        y_true=pd.Series([0, 1, 1, 0, 1, 0, 1, 0, 1, 1]),
        y_pred=np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 1]),
    )
    metrics.log_report(report, "logs/evaluation_report.json")
    print(f"Evaluation report: {report}")


if __name__ == "__main__":
    main()
