"""Small HTTP client shared by the Streamlit applications."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import requests


class APIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class APIClient:
    def __init__(
        self, base_url: str, token: str | None = None, timeout_seconds: float = 10
    ) -> None:
        normalized = base_url.rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("API URL must be an absolute HTTP(S) URL")
        self.base_url = normalized
        self.token = token.strip() if token else None
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        headers = dict(kwargs.pop("headers", {}))
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                headers=headers,
                timeout=self.timeout_seconds,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise APIError(f"Cannot reach AIPulse API: {exc}") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise APIError(
                "AIPulse API returned a non-JSON response", response.status_code
            ) from exc
        if not response.ok:
            detail = (
                payload.get("detail", "Request failed")
                if isinstance(payload, dict)
                else "Request failed"
            )
            raise APIError(str(detail), response.status_code)
        if not isinstance(payload, dict):
            raise APIError("AIPulse API returned an invalid payload", response.status_code)
        return payload

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/health")

    def list_models(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/models")

    def train_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v1/models/train", json=payload)
