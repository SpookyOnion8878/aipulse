from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.security.audit import AuditTrail
from src.security.security_ethics import BiasDetector, ComplianceChecker


def test_fairness_requires_real_sensitive_groups() -> None:
    result = BiasDetector().run_bias_audit(np.array([0, 1]), np.array([0, 1]), None)
    assert result["status"] == "not_assessed"


def test_fairness_reports_group_disparity() -> None:
    truth = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    prediction = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    groups = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    result = BiasDetector(disparity_threshold=0.2).run_bias_audit(truth, prediction, groups)
    assert result["status"] == "warning"
    assert result["metrics"]["equal_opportunity_difference"] == 1.0


def test_compliance_does_not_default_to_passed() -> None:
    result = ComplianceChecker().check_compliance({})
    assert result["status"] == "not_assessed"
    assert result["missing_evidence"]


def test_audit_redacts_secrets_and_chains_events(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    audit = AuditTrail(path, "integrity-secret")
    first = audit.append(
        actor="user-1",
        action="login",
        resource="api",
        outcome="success",
        metadata={"access_token": "do-not-store"},
    )
    second = audit.append(actor="user-1", action="model.read", resource="m:1", outcome="success")
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["metadata"]["access_token"] == "[REDACTED]"
    assert second["previous_hash"] == first["integrity_hash"]
