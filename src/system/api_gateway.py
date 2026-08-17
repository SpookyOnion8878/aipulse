"""FastAPI application boundary for AIPulse."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from ..config import Settings
from ..core.model_training import TrainingConfig, TrainingService
from ..logging_config import configure_logging
from ..security.audit import AuditTrail
from ..security.auth import Principal, require_roles
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)

ViewerPrincipal = Annotated[
    Principal,
    Depends(require_roles("viewer", "admin", "ml_engineer", "data_scientist")),
]
TrainerPrincipal = Annotated[Principal, Depends(require_roles("admin", "ml_engineer"))]


class RateLimiter:
    """Bounded process-local sliding-window limiter for the single-worker API."""

    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
        *,
        max_principals: int = 10_000,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_principals = max_principals
        self._requests: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def consume(self, principal: str, endpoint: str) -> tuple[bool, int]:
        key = f"{principal}:{endpoint}"
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            entries = self._requests.setdefault(key, deque())
            while entries and entries[0] <= cutoff:
                entries.popleft()
            if len(entries) >= self.max_requests:
                retry_after = max(1, int(entries[0] + self.window_seconds - now) + 1)
                return False, retry_after
            entries.append(now)
            self._requests.move_to_end(key)
            while len(self._requests) > self.max_principals:
                self._requests.popitem(last=False)
            return True, 0


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dataset_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    model_name: str = Field(default="ai_model", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    model_type: Literal["logistic_regression", "random_forest"] = "logistic_regression"
    label_column: str = Field(default="label", pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
    validation_fraction: float = Field(default=0.2, ge=0.1, le=0.5)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    version: str
    authentication_configured: bool


def _state(request: Request, name: str) -> Any:
    value = getattr(request.app.state, name, None)
    if value is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Not ready")
    return value


def enforce_rate_limit(request: Request, principal: Principal) -> None:
    limiter: RateLimiter = _state(request, "rate_limiter")
    allowed, retry_after = limiter.consume(principal.subject, request.url.path)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        runtime_settings.ensure_runtime_directories()
        application.state.settings = runtime_settings
        application.state.registry = ModelRegistry(runtime_settings.registry_path)
        application.state.training_service = TrainingService(
            runtime_settings, application.state.registry
        )
        application.state.audit = AuditTrail(
            runtime_settings.audit_log_path, runtime_settings.jwt_secret
        )
        application.state.rate_limiter = RateLimiter(
            runtime_settings.rate_limit_requests,
            runtime_settings.rate_limit_window_seconds,
        )
        logger.info("AIPulse API started environment=%s", runtime_settings.environment)
        yield

    application = FastAPI(
        title="AIPulse API",
        version="0.2.0",
        docs_url="/docs" if runtime_settings.environment != "production" else None,
        redoc_url=None,
        lifespan=lifespan,
    )
    if runtime_settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(runtime_settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @application.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        configured = runtime_settings.jwt_secret is not None
        return HealthResponse(
            status="ok" if configured else "degraded",
            service="aipulse-api",
            version="0.2.0",
            authentication_configured=configured,
        )

    @application.get("/api/v1/models", tags=["models"])
    def list_models(request: Request, principal: ViewerPrincipal) -> dict[str, Any]:
        enforce_rate_limit(request, principal)
        registry: ModelRegistry = _state(request, "registry")
        models = registry.list_models()
        return {"items": models, "count": len(models)}

    @application.post(
        "/api/v1/models/train",
        status_code=status.HTTP_201_CREATED,
        tags=["models"],
    )
    def train_model(
        payload: TrainRequest,
        request: Request,
        principal: TrainerPrincipal,
    ) -> dict[str, Any]:
        enforce_rate_limit(request, principal)
        service: TrainingService = _state(request, "training_service")
        audit: AuditTrail = _state(request, "audit")
        try:
            result = service.train(
                payload.dataset_name,
                TrainingConfig(
                    model_name=payload.model_name,
                    model_type=payload.model_type,
                    label_column=payload.label_column,
                    validation_fraction=payload.validation_fraction,
                ),
            )
        except FileNotFoundError as exc:
            audit.append(
                actor=principal.subject,
                action="model.train",
                resource=payload.model_name,
                outcome="not_found",
                metadata={"dataset": payload.dataset_name},
            )
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ValueError as exc:
            audit.append(
                actor=principal.subject,
                action="model.train",
                resource=payload.model_name,
                outcome="rejected",
                metadata={"dataset": payload.dataset_name, "reason": str(exc)},
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        audit.append(
            actor=principal.subject,
            action="model.train",
            resource=f"{result['model']}:{result['version']}",
            outcome="success",
            metadata={
                "dataset": payload.dataset_name,
                "validation": result["validation"],
            },
        )
        return result

    return application


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("aipulse.system.api_gateway:app", host="0.0.0.0", port=8000, workers=1)


if __name__ == "__main__":
    run()
