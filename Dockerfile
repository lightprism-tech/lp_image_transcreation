# ==============================================================================
# Image Transcreation Pipeline - Docker Image
# Stage 1: Base image with system dependencies
# ==============================================================================
FROM python:3.10-slim AS base

# Metadata
LABEL maintainer="LightPrism.tech"
LABEL org.opencontainers.image.title="Image Transcreation Pipeline"
LABEL org.opencontainers.image.description="Context-aware image transcreation: perception, cultural reasoning, and visual realization"
LABEL org.opencontainers.image.version="0.1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for OpenCV and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Utilities
    wget \
    curl \
    # Terminal utilities
    bash \
    bash-completion \
    vim \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# ==============================================================================
# Stage 2: Python Dependencies
# ==============================================================================
FROM base AS dependencies

WORKDIR /app

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install Python packages with no pip cache
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 3: Application
# ==============================================================================
FROM dependencies AS application

WORKDIR /app

# Copy project configuration files
COPY pyproject.toml README.md ./

# Copy application code (perception, reasoning, realization)
COPY src/ ./src/

# Copy data layout (input, knowledge_base; data/output in .dockerignore)
COPY data/ ./data/

# Install the package in editable mode (no pip cache)
RUN pip install --no-cache-dir -e .

# Ensure full directory layout (volumes override at run)
RUN mkdir -p \
    /app/models/yolo \
    /app/models/caption_model \
    /app/models/classifier_model \
    /app/data/input/samples \
    /app/data/output/debug \
    /app/data/output/json \
    /app/data/knowledge_base \
    /app/cache \
    && chmod -R 755 /app

# Set CPU optimization environment variables
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    VECLIB_MAXIMUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    FLAGS_use_mkldnn=false \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# Default app paths (override via .env when using docker-compose)
ENV MODELS_DIR=/app/models \
    DATA_DIR=/app/data \
    CACHE_DIR=/app/cache \
    OUTPUT_DIR=/app/data/output

# Expose port for potential API service
EXPOSE 8000

# Set working directory
WORKDIR /app

# Default command: interactive bash (override: docker-compose run pipeline python -m perception /app/data/input/samples/testing_image.jpg)
CMD ["/bin/bash"]
