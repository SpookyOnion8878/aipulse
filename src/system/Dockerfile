# Dockerfile — API Gateway Service

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

CMD ["uvicorn", "src.system.api_gateway:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
