# ==============================================================================
# OutfitTransformer - Production Dockerfile
# Multi-stage build with NVIDIA CUDA for GPU inference
# ==============================================================================
# Build:  docker build -t outfit-transformer .
# Run:    docker run --gpus all -p 8000:8000 --env-file .env outfit-transformer
# ==============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — install Python deps and download model weights
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download FashionCLIP weights into HuggingFace cache
# This avoids a ~600MB download on first request at runtime
RUN pip install --no-cache-dir transformers torch && \
    python -c "\
from transformers import CLIPModel, CLIPProcessor; \
CLIPModel.from_pretrained('patrickjohncyh/fashion-clip'); \
CLIPProcessor.from_pretrained('patrickjohncyh/fashion-clip'); \
print('FashionCLIP weights downloaded')"


# ---------------------------------------------------------------------------
# Stage 2: Runtime — NVIDIA CUDA base with minimal footprint
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 + minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    libpython3.11 \
    curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy HuggingFace model cache (FashionCLIP weights, ~600MB)
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Set up application directory
WORKDIR /app

# Copy source code
COPY src/ ./src/

# Environment
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV PORT=8000

# Expose API port
EXPOSE 8000

# Health check — start period allows time for model loading + warmup
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# Run with ddtrace-run wrapping gunicorn + uvicorn workers
# ddtrace-run auto-instruments: FastAPI, Redis, requests, torch
# - 2 workers: each loads CLIP + SASRec models (~700MB VRAM per worker)
# - 120s timeout: hybrid search can take up to 15s, leave headroom
# - Graceful shutdown: 30s for in-flight requests to complete
CMD ["ddtrace-run", "gunicorn", "api.app:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-"]
