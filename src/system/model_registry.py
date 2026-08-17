"""Atomic local model registry used by the modular-monolith deployment."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z.-]+))?$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_key(version: str) -> tuple[int, int, int, int, str]:
    match = SEMVER.fullmatch(version)
    if match is None:
        raise ValueError(f"Invalid semantic version: {version}")
    major, minor, patch, prerelease = match.groups()
    return int(major), int(minor), int(patch), 1 if prerelease is None else 0, prerelease or ""


@dataclass(frozen=True, slots=True)
class ModelVersion:
    model_name: str
    version: str
    metrics: dict[str, Any]
    artifacts: dict[str, str]
    checksum_sha256: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    tags: list[str] = field(default_factory=list)
    training_data_version: str | None = None


class ModelRegistry:
    """Persist registry metadata with process-local locking and atomic writes."""

    def __init__(self, registry_path: str | Path = "models/registry.json") -> None:
        self.registry_path = Path(registry_path).resolve()
        self._lock = threading.RLock()
        self._data = self._load()

    def _empty(self) -> dict[str, Any]:
        return {"schema_version": 1, "models": {}, "versions": []}

    def _load(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return self._empty()
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid model registry: {self.registry_path}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
            raise ValueError("Model registry has an invalid schema")
        data.setdefault("schema_version", 1)
        data.setdefault("versions", [])
        return data

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{self.registry_path.name}.",
            suffix=".tmp",
            dir=self.registry_path.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self._data, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.registry_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def register(
        self,
        model_name: str,
        version: str,
        metrics: dict[str, Any],
        artifacts: dict[str, str],
        *,
        checksum_sha256: str,
        tags: list[str] | None = None,
        training_data_version: str | None = None,
    ) -> ModelVersion:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", model_name):
            raise ValueError("model_name contains unsupported characters")
        _version_key(version)
        if not re.fullmatch(r"[0-9a-f]{64}", checksum_sha256):
            raise ValueError("checksum_sha256 must be a lowercase SHA-256 digest")
        record = ModelVersion(
            model_name=model_name,
            version=version,
            metrics=metrics,
            artifacts=artifacts,
            checksum_sha256=checksum_sha256,
            tags=tags or [],
            training_data_version=training_data_version,
        )
        key = f"{model_name}:{version}"
        with self._lock:
            models = self._data.setdefault("models", {})
            versions = self._data.setdefault("versions", [])
            model_versions = models.setdefault(model_name, {})
            if version in model_versions:
                raise ValueError(f"Model version already exists: {key}")
            model_versions[version] = asdict(record)
            versions.append(key)
            self._save()
        return record

    def get_model(self, model_name: str, version: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            versions = self._data.get("models", {}).get(model_name, {})
            if not versions:
                return None
            selected = version or max(versions, key=_version_key)
            record = versions.get(selected)
            return dict(record) if isinstance(record, dict) else None

    def list_models(self, model_name: str | None = None) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with self._lock:
            for name, versions in self._data.get("models", {}).items():
                if model_name is not None and name != model_name:
                    continue
                for version, record in versions.items():
                    if isinstance(record, dict):
                        records.append(dict(record, model_name=name, version=version))
        return sorted(records, key=lambda item: (item["model_name"], _version_key(item["version"])))

    def delete_model(self, model_name: str, version: str | None = None) -> bool:
        with self._lock:
            models = self._data.get("models", {})
            if model_name not in models:
                return False
            if version is None:
                del models[model_name]
                prefix = f"{model_name}:"
                self._data["versions"] = [
                    key for key in self._data.get("versions", []) if not key.startswith(prefix)
                ]
            else:
                if version not in models[model_name]:
                    return False
                del models[model_name][version]
                self._data["versions"] = [
                    key
                    for key in self._data.get("versions", [])
                    if key != f"{model_name}:{version}"
                ]
                if not models[model_name]:
                    del models[model_name]
            self._save()
        return True
