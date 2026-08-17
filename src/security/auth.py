"""JWT authentication and role authorization for the API boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError

from ..config import Settings

bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True, slots=True)
class Principal:
    subject: str
    roles: frozenset[str]


def _settings(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    if not isinstance(settings, Settings):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Not ready")
    return settings


def current_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> Principal:
    settings = _settings(request)
    if settings.jwt_secret is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured",
        )
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    subject = payload.get("sub")
    raw_roles = payload.get("roles", [])
    if not isinstance(subject, str) or not subject.strip():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid subject")
    if isinstance(raw_roles, str):
        roles = frozenset({raw_roles})
    elif isinstance(raw_roles, list) and all(isinstance(role, str) for role in raw_roles):
        roles = frozenset(raw_roles)
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid roles")
    return Principal(subject=subject, roles=roles)


def require_roles(*allowed_roles: str) -> Any:
    allowed = frozenset(allowed_roles)

    def dependency(principal: Principal = Depends(current_principal)) -> Principal:
        if principal.roles.isdisjoint(allowed):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return principal

    return dependency


def create_access_token(
    settings: Settings,
    subject: str,
    roles: list[str],
    *,
    lifetime: timedelta = timedelta(minutes=15),
) -> str:
    """Create a short-lived token for tests and trusted development tooling."""

    if settings.jwt_secret is None:
        raise ValueError("AIPULSE_JWT_SECRET is required")
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": subject,
        "roles": roles,
        "iat": now,
        "exp": now + lifetime,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
