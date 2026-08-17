"""Append-only, integrity-linked security audit events."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SENSITIVE_FRAGMENTS = ("password", "secret", "token", "authorization", "cookie")


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]"
            if any(fragment in str(key).lower() for fragment in SENSITIVE_FRAGMENTS)
            else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    return value


class AuditTrail:
    """Write JSONL events with a hash chain and bounded metadata."""

    def __init__(self, path: str | Path, integrity_secret: str | None = None) -> None:
        self.path = Path(path).resolve()
        self._secret = integrity_secret.encode() if integrity_secret else None
        self._lock = threading.RLock()

    def _previous_hash(self) -> str:
        if not self.path.exists():
            return "0" * 64
        previous = "0" * 64
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict) and isinstance(item.get("integrity_hash"), str):
                        previous = item["integrity_hash"]
        return previous

    def append(
        self,
        *,
        actor: str,
        action: str,
        resource: str,
        outcome: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        safe_metadata = _redact(metadata or {})
        encoded = json.dumps(safe_metadata, separators=(",", ":"), sort_keys=True)
        if len(encoded.encode("utf-8")) > 16_384:
            raise ValueError("Audit metadata exceeds 16 KiB")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            event: dict[str, Any] = {
                "timestamp": datetime.now(UTC).isoformat(),
                "actor": actor,
                "action": action,
                "resource": resource,
                "outcome": outcome,
                "metadata": safe_metadata,
                "previous_hash": self._previous_hash(),
            }
            canonical = json.dumps(event, separators=(",", ":"), sort_keys=True).encode()
            if self._secret is None:
                digest = hashlib.sha256(canonical).hexdigest()
            else:
                digest = hmac.new(self._secret, canonical, hashlib.sha256).hexdigest()
            event["integrity_hash"] = digest
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return event
