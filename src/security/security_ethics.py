"""
Security & Ethics — Step 4: Security & Ethics

AIPulse Enterprise AI Platform — Security & Ethics Layer
Handles bias detection, audit trails, compliance, and security.
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
        logging.FileHandler(os.path.join(LOG_DIR, 'security.log')),
    ],
)
logger = logging.getLogger("aipulse.security")


class BiasDetector:
    """Detect bias in AI models."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.thresholds = {
            "demographic_parity": 0.1,
            "equalized_odds": 0.1,
            "equal_opportunity": 0.1,
            "predictive_parity": 0.1,
        }

    def calculate_demographic_parity(self, y_true: pd.Series, y_pred: np.ndarray,
                                      demographic_group: pd.Series) -> float:
        """Calculate demographic parity."""
        total = len(y_true)
        group_predictions = {}
        group_labels = {}

        for group in demographic_group.unique():
            mask = demographic_group == group
            group_predictions[group] = y_pred[mask].mean()
            group_labels[group] = y_true[mask].mean()

        # Calculate parity
        parity = (
            group_predictions.get("group_A", 0) - group_predictions.get("group_B", 0)
        )
        return abs(parity)

    def calculate_equalized_odds(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate equalized odds."""
        # True positive rate for each group
        group_tpr = {}
        group_fpr = {}
        for group in y_true.unique():
            mask = y_true == group
            group_tpr[group] = (y_pred[mask] == 1).sum() / mask.sum()
            group_fpr[group] = (y_pred[mask] == 0).sum() / mask.sum()

        # Calculate odds ratio
        odds = {}
        for group in group_tpr:
            odds[group] = group_tpr[group] / group_fpr[group] if group_fpr[group] > 0 else 0

        return np.mean(list(odds.values()))

    def calculate_equal_opportunity(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate equal opportunity."""
        # True positive rate for each group
        group_tpr = {}
        for group in y_true.unique():
            mask = y_true == group
            group_tpr[group] = (y_pred[mask] == 1).sum() / mask.sum()

        return np.mean(list(group_tpr.values()))

    def calculate_predictive_parity(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate predictive parity."""
        # Positive predictive value for each group
        group_ppv = {}
        for group in y_true.unique():
            mask = y_true == group
            group_ppv[group] = (y_pred[mask] == 1).sum() / mask.sum() if mask.sum() > 0 else 0

        return np.mean(list(group_ppv.values()))

    def run_bias_audit(self, y_true: pd.Series, y_pred: np.ndarray,
                        demographic_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Run comprehensive bias audit."""
        audit = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "thresholds": self.thresholds,
            "warnings": [],
        }

        if demographic_data is None or "group" not in demographic_data.columns:
            groups = pd.Series(np.where(np.arange(len(y_true)) % 2 == 0, "group_A", "group_B"))
        else:
            groups = demographic_data["group"]

        audit["metrics"]["demographic_parity"] = self.calculate_demographic_parity(
            y_true, y_pred, groups
        )
        audit["metrics"]["equalized_odds"] = self.calculate_equalized_odds(y_true, y_pred)
        audit["metrics"]["equal_opportunity"] = self.calculate_equal_opportunity(y_true, y_pred)
        audit["metrics"]["predictive_parity"] = self.calculate_predictive_parity(y_true, y_pred)

        # Check if any metric exceeds threshold
        for metric_name, value in audit["metrics"].items():
            if value > self.thresholds.get(metric_name, 0.1):
                audit["warnings"].append(f"{metric_name} exceeds threshold: {value:.4f}")

        return audit

    def generate_fairness_report(self, audit: Dict) -> Dict:
        """Generate fairness report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "audit": audit,
            "overall_fairness": self._compute_fairness_score(audit),
        }
        return report

    def _compute_fairness_score(self, audit: Dict) -> float:
        """Compute fairness score."""
        metrics = audit.get("metrics", {})
        if not metrics:
            return 0.0

        # Simple fairness score (average of all metrics)
        scores = [v for v in metrics.values() if isinstance(v, (int, float))]
        if scores:
            return np.mean(scores)
        return 0.0


class AuditTrail:
    """Audit trail for AI decisions."""

    def __init__(self, audit_path: str = "logs/audit_trail.json"):
        self.audit_path = audit_path

    def log_ai_decision(self, decision: Dict) -> None:
        """Log AI decision to audit trail."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "user_id": decision.get("user_id", "anonymous"),
            "model_name": decision.get("model_name", "unknown"),
            "input_data": decision.get("input_data", {}),
            "output_data": decision.get("output_data", {}),
            "reasoning": decision.get("reasoning", ""),
            "context": decision.get("context", {}),
        }

        # Add to audit trail
        os.makedirs(os.path.dirname(self.audit_path), exist_ok=True)
        audit_file = self.audit_path

        # Load existing audit trail
        if os.path.exists(audit_file):
            with open(audit_file, "r") as f:
                audit_data = json.load(f)
        else:
            audit_data = []

        audit_data.append(log_entry)

        with open(audit_file, "w") as f:
            json.dump(audit_data, f, indent=2)

        logger.info(f"AI decision logged: {decision.get('model_name')} v{decision.get('version', 'N/A')}")

    def get_audit_log(self) -> List[Dict]:
        """Get audit log."""
        if not os.path.exists(self.audit_path):
            return []

        with open(self.audit_path, "r") as f:
            return json.load(f)

    def export_audit(self, output_path: str = "logs/audit_export.json") -> None:
        """Export audit trail to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.get_audit_log(), f, indent=2)
        logger.info(f"Audit trail exported to {output_path}")


class ComplianceChecker:
    """Compliance checking for AI systems."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.compliance_rules = {
            "GDPR": {
                "data_minimization": True,
                "right_to_deletion": True,
                "consent_required": True,
                "data_residency": True,
            },
            "EU_AI_Act": {
                "risk_assessment": True,
                "transparency": True,
                "human_overview": True,
                "human_override": True,
            },
            "AI_Act_2024": {
                "bias_check": True,
                "explainability": True,
                "fairness": True,
                "accountability": True,
            },
            "HIPAA": {
                "data_protection": True,
                "access_control": True,
                "audit_trail": True,
            },
        }

    def check_compliance(self, system_config: Dict, models: List[Dict]) -> Dict[str, Any]:
        """Check compliance for all models."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_compliance": True,
            "violations": [],
            "warnings": [],
            "checks": {},
        }

        for rule_set_name, rule_set in self.compliance_rules.items():
            for check_name, requirement in rule_set.items():
                # Simulate compliance check
                if requirement:
                    # Check if model satisfies the requirement
                    satisfies = self._check_rule(system_config, check_name, models)
                    results["checks"][f"{rule_set_name}:{check_name}"] = {
                        "satisfied": satisfies,
                        "requirement": requirement,
                    }

                    if not satisfies:
                        results["violations"].append(f"Compliance check failed: {rule_set_name}:{check_name}")

        # Overall compliance
        if results["violations"]:
            results["overall_compliance"] = False

        return results

    def _check_rule(self, system_config: Dict, check_name: str, models: List[Dict]) -> bool:
        """Check if a rule is satisfied."""
        # TODO: Implement actual compliance checks
        # For now, simulate
        if check_name == "bias_check":
            return True  # Model has bias check
        elif check_name == "explainability":
            return True  # Model has explainability
        elif check_name == "fairness":
            return True  # Model is fair
        elif check_name == "accountability":
            return True  # Model is accountable
        else:
            return True  # Default to satisfied

    def generate_compliance_report(self, compliance_results: Dict) -> Dict:
        """Generate compliance report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "compliance_checks": compliance_results,
            "overall_status": "compliant" if compliance_results["overall_compliance"] else "non_compliant",
            "recommendations": [],
        }

        # Generate recommendations
        for violation in compliance_results.get("violations", []):
            report["recommendations"].append(f"Address: {violation}")

        return report


class ModelPoisoningDetector:
    """Detect model poisoning attacks."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.suspicious_signals = [
            "distribution_shift",
            "label_noise",
            "target_manipulation",
            "data_leakage",
        ]
        self.thresholds = {
            "distribution_shift": 0.3,
            "label_noise": 0.01,
            "target_manipulation": 0.2,
        }

    def detect(self, input_data: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """Detect model poisoning."""
        detection_results = {
            "timestamp": datetime.now().isoformat(),
            "signals_checked": [],
            "anomalies": [],
            "overall_status": "clean",
            "confidence": 0.0,
        }

        # Check for distribution shift
        if "distribution_shift" in self.suspicious_signals:
            # Simple distribution check
            for col in input_data.select_dtypes(include=['number']).columns:
                mean_old = input_data[col].mean()
                mean_new = target.mean()
                shift = abs(mean_old - mean_new) / max(mean_old, 1e-8)
                if shift > self.thresholds["distribution_shift"]:
                    detection_results["signals_checked"].append({
                        "signal": "distribution_shift",
                        "severity": "high",
                        "score": shift,
                    })
                    detection_results["anomalies"].append(f"Distribution shift detected in {col}")

        # Check for label noise
        if "label_noise" in self.suspicious_signals:
            noise_rate = (input_data["label"] == 0).sum() / len(input_data) if len(input_data) > 0 else 0
            if noise_rate > self.thresholds["label_noise"]:
                detection_results["signals_checked"].append({
                    "signal": "label_noise",
                    "severity": "medium",
                    "score": noise_rate,
                })
                detection_results["anomalies"].append(f"Label noise detected: {noise_rate:.4f}")

        # Check for target manipulation
        if "target_manipulation" in self.suspicious_signals:
            target_mean = target.mean()
            if target_mean > 0.5:
                detection_results["signals_checked"].append({
                    "signal": "target_manipulation",
                    "severity": "low",
                    "score": target_mean,
                })
                detection_results["anomalies"].append(f"Target manipulation detected")

        # Overall confidence
        if detection_results["anomalies"]:
            detection_results["overall_status"] = "suspicious"
            detection_results["confidence"] = 0.5 + (len(detection_results["anomalies"]) * 0.1)
        else:
            detection_results["overall_status"] = "clean"
            detection_results["confidence"] = 0.9

        return detection_results


# Entry point
def main():
    """Run security & ethics checks."""
    # Bias detection
    detector = BiasDetector()
    y_true = pd.Series([0, 1, 1, 0, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
    audit = detector.run_bias_audit(y_true, y_pred, None)

    print(f"Bias audit: {audit}")

    # Audit trail
    audit_trail = AuditTrail()
    audit_trail.log_ai_decision({
        "model_name": "ai_model",
        "version": "1.0.0",
        "user_id": "admin_001",
        "input_data": {"features": [1.0, 2.0, 3.0]},
        "output_data": {"prediction": 1},
        "reasoning": "Model output based on inference",
        "context": {"request_id": "req_001"},
    })

    # Compliance check
    compliance = ComplianceChecker()
    compliance_results = compliance.check_compliance(
        {"environment": "production", "security_level": "high"},
        [{"name": "ai_model", "type": "classification"}],
    )

    print(f"Compliance check: {compliance_results}")


if __name__ == "__main__":
    main()
