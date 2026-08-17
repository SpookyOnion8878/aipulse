FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN groupadd --system --gid 10001 aipulse \
    && useradd --system --uid 10001 --gid aipulse --home-dir /app aipulse

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir . \
    && mkdir -p /app/data/processed /app/models /app/logs \
    && chown -R aipulse:aipulse /app

USER aipulse

EXPOSE 8000

CMD ["uvicorn", "aipulse.system.api_gateway:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-server-header"]
