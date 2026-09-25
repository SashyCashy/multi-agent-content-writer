# Slim base keeps the image well under Artifact Registry's 0.5 GB free tier.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so code changes don't invalidate the pip layer.
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY src ./src

# Cloud Run injects PORT and expects the server to bind to it on 0.0.0.0.
ENV PORT=8080
EXPOSE 8080

# `exec` form so uvicorn is PID 1 and receives SIGTERM directly — Cloud Run
# sends it on scale-down, and without this the container is killed instead of
# shutting down cleanly.
CMD exec uvicorn src.web:app --host 0.0.0.0 --port ${PORT}
