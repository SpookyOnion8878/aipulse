"""Environment-backed application configuration.

Configuration is loaded explicitly instead of at import time so tests and
command-line tools can supply isolated settings without global side effects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _origins(raw: str) -> tuple[str, ...]:
    return tuple(origin.strip() for origin in raw.split(",") if origin.strip())


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings shared by API and services."""

    environment: str
    data_dir: Path
    models_dir: Path
    logs_dir: Path
    registry_path: Path
    audit_log_path: Path
    jwt_secret: str | None
    jwt_algorithm: str
    jwt_issuer: str
    jwt_audience: str
    rate_limit_requests: int
    rate_limit_window_seconds: int
    cors_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> Settings:
        data_dir = Path(os.getenv("AIPULSE_DATA_DIR", "data")).resolve()
        models_dir = Path(os.getenv("AIPULSE_MODELS_DIR", "models")).resolve()
        logs_dir = Path(os.getenv("AIPULSE_LOGS_DIR", "logs")).resolve()
        return cls(
            environment=os.getenv("AIPULSE_ENV", "development").strip().lower(),
            data_dir=data_dir,
            models_dir=models_dir,
            logs_dir=logs_dir,
            registry_path=Path(
                os.getenv("AIPULSE_REGISTRY_PATH", str(models_dir / "registry.json"))
            ).resolve(),
            audit_log_path=Path(
                os.getenv("AIPULSE_AUDIT_LOG_PATH", str(logs_dir / "audit_trail.jsonl"))
            ).resolve(),
            jwt_secret=os.getenv("AIPULSE_JWT_SECRET") or None,
            jwt_algorithm=os.getenv("AIPULSE_JWT_ALGORITHM", "HS256"),
            jwt_issuer=os.getenv("AIPULSE_JWT_ISSUER", "aipulse"),
            jwt_audience=os.getenv("AIPULSE_JWT_AUDIENCE", "aipulse-api"),
            rate_limit_requests=_positive_int("AIPULSE_RATE_LIMIT_REQUESTS", 60),
            rate_limit_window_seconds=_positive_int("AIPULSE_RATE_LIMIT_WINDOW_SECONDS", 60),
            cors_origins=_origins(os.getenv("AIPULSE_CORS_ORIGINS", "http://localhost:8080")),
        )

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    def ensure_runtime_directories(self) -> None:
        for path in (self.data_dir, self.processed_data_dir, self.models_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)
