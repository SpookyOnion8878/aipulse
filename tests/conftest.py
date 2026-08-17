from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    logs_dir = tmp_path / "logs"
    value = Settings(
        environment="test",
        data_dir=data_dir,
        models_dir=models_dir,
        logs_dir=logs_dir,
        registry_path=models_dir / "registry.json",
        audit_log_path=logs_dir / "audit.jsonl",
        jwt_secret="test-secret-that-is-long-enough-for-local-tests",
        jwt_algorithm="HS256",
        jwt_issuer="aipulse-test",
        jwt_audience="aipulse-api-test",
        rate_limit_requests=20,
        rate_limit_window_seconds=60,
        cors_origins=(),
    )
    value.ensure_runtime_directories()
    return value
