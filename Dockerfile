# =============================================================================
# TESA Defence AI - Dockerfile
# =============================================================================
# Multi-stage build: slim Python image with optional CUDA support.
#
# Build (CPU):
#   docker build -t tesa-defence .
#
# Build (GPU / CUDA 12):
#   docker build --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
#                -t tesa-defence-gpu .
#
# Run dashboard:
#   docker run -p 8501:8501 tesa-defence
#
# Run CLI:
#   docker run -it tesa-defence python cli.py
# =============================================================================

ARG BASE_IMAGE=python:3.11-slim

# ---------------------------------------------------------------------------
# Stage 1 – Builder (install dependencies)
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder

# Install system dependencies needed for OpenCV & build tools
# Also install python3 + pip for CUDA base images that don't ship them
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        python3-dev \
        python3-pip \
        python3-venv \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Ensure python/pip point to the right version
RUN ln -sf /usr/bin/python3 /usr/bin/python 2>/dev/null || true \
    && ln -sf /usr/bin/pip3 /usr/bin/pip 2>/dev/null || true

WORKDIR /build

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages --prefix=/install \
    -r requirements.txt 2>/dev/null \
    || pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 – Runtime
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime

# Runtime system packages
# NOTE: python3 is only needed for CUDA base images (python:*-slim already has it)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Ensure python exists
RUN ln -sf /usr/bin/python3 /usr/bin/python 2>/dev/null || true

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN groupadd -r tesa && useradd -r -g tesa -m -s /bin/bash tesa

# Set working directory
WORKDIR /app

# Copy project files (respects .dockerignore)
COPY --chown=tesa:tesa . .

# Create necessary output directories
RUN mkdir -p outputs/problem_1 outputs/problem_2 outputs/problem_3/final \
             submissions \
    && chown -R tesa:tesa outputs submissions

# Switch to non-root user
USER tesa

# Streamlit configuration – headless mode, no telemetry
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

EXPOSE 8501

# Default: launch Streamlit dashboard
ENTRYPOINT ["python", "-m", "streamlit", "run", "dashboard.py"]
