# Dockerfile — Core Model Training Service

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/core/ ./src/core/
COPY models/ ./models/

EXPOSE 8001

ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

CMD ["python", "src.core.model_training.py"]
