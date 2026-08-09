# Dockerfile — Data Engineering Service

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/data_engineering/ ./src/data_engineering/

EXPOSE 8002

ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

CMD ["python", "src.data_engineering.data_pipeline.py"]
