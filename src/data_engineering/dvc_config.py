"""Checksummed local dataset snapshots.

This is intentionally named a snapshot store rather than pretending to be a
DVC remote. A real DVC adapter can replace it behind the same boundary later.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SnapshotStore:
    """Create, verify, list, compare, and restore immutable local snapshots."""

    def __init__(self, root: str | Path = "data/snapshots") -> None:
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "manifest.json"
        self._lock = threading.RLock()

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"schema_version": 1, "snapshots": {}}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Snapshot manifest is invalid") from exc
        if not isinstance(data, dict) or not isinstance(data.get("snapshots"), dict):
            raise ValueError("Snapshot manifest has an invalid schema")
        return data

    def _save_manifest(self, data: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(prefix=".manifest.", suffix=".json", dir=self.root)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.manifest_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def create_snapshot(self, source: str | Path, version: str) -> dict[str, str]:
        if VERSION.fullmatch(version) is None:
            raise ValueError("Invalid snapshot version")
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Snapshot source not found: {source_path}")
        suffix = source_path.suffix.lower() or ".bin"
        destination = (self.root / f"{version}{suffix}").resolve()
        if not destination.is_relative_to(self.root):
            raise ValueError("Snapshot path escapes the store")
        with self._lock:
            manifest = self._load_manifest()
            if version in manifest["snapshots"]:
                raise ValueError(f"Snapshot already exists: {version}")
            self.root.mkdir(parents=True, exist_ok=True)
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{version}.", suffix=suffix, dir=self.root
            )
            os.close(descriptor)
            try:
                shutil.copy2(source_path, temp_name)
                os.replace(temp_name, destination)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
            record = {
                "version": version,
                "file": destination.name,
                "sha256": _sha256(destination),
                "created_at": datetime.now(UTC).isoformat(),
                "source_name": source_path.name,
            }
            manifest["snapshots"][version] = record
            self._save_manifest(manifest)
            return record

    def list_snapshots(self) -> list[dict[str, str]]:
        manifest = self._load_manifest()
        return [dict(item) for _, item in sorted(manifest["snapshots"].items())]

    def restore_snapshot(self, version: str, target: str | Path) -> Path:
        manifest = self._load_manifest()
        record = manifest["snapshots"].get(version)
        if not isinstance(record, dict):
            raise FileNotFoundError(f"Unknown snapshot: {version}")
        source = (self.root / record["file"]).resolve()
        if not source.is_relative_to(self.root) or not source.is_file():
            raise ValueError("Snapshot artifact is missing or unsafe")
        if _sha256(source) != record["sha256"]:
            raise ValueError("Snapshot checksum mismatch")
        target_path = Path(target).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_path)
        return target_path

    def compare_versions(self, version_a: str, version_b: str) -> dict[str, Any]:
        manifest = self._load_manifest()["snapshots"]
        if version_a not in manifest or version_b not in manifest:
            raise FileNotFoundError("Both snapshot versions must exist")
        return {
            "version_a": version_a,
            "version_b": version_b,
            "identical": manifest[version_a]["sha256"] == manifest[version_b]["sha256"],
            "sha256_a": manifest[version_a]["sha256"],
            "sha256_b": manifest[version_b]["sha256"],
        }


DVCSnapshot = SnapshotStore
