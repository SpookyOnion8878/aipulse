# Dockerfile — Security & Ethics Service

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/security/ ./src/security/

EXPOSE 8003

ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

CMD ["python", "src.security.security_ethics.py"]
