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
ARG PRELOAD_VIT_DETECTOR=0
ARG VIT_DETECTOR_MODEL=google/owlv2-large-patch14-ensemble
ARG PRELOAD_BLIP_MODEL=0
ARG BLIP_MODEL=Salesforce/blip-image-captioning-large

# Default app paths (override via .env when using docker-compose).
# HF_HOME/HF_HUB_CACHE/TRANSFORMERS_CACHE: Hugging Face / Transformers model cache.
# TORCH_HOME: torch.hub checkpoints.
# HOME: PaddleOCR 2.x uses ~/.paddleocr (expanduser); must live under the mounted cache volume.
# XDG_CACHE_HOME/PADDLE_HOME/PADDLEOCR_HOME/YOLO_CONFIG_DIR keep runtime downloads and tool config reusable.
ENV MODELS_DIR=/app/models \
    DATA_DIR=/app/data \
    CACHE_DIR=/app/cache \
    OUTPUT_DIR=/app/data/output \
    HF_HOME=/app/cache/huggingface \
    HF_HUB_CACHE=/app/cache/huggingface/hub \
    TRANSFORMERS_CACHE=/app/cache/huggingface/transformers \
    TORCH_HOME=/app/cache/torch \
    XDG_CACHE_HOME=/app/cache/xdg \
    PADDLE_HOME=/app/cache/paddle \
    PADDLEOCR_HOME=/app/cache/paddleocr \
    YOLO_CONFIG_DIR=/app/cache/ultralytics \
    MPLCONFIGDIR=/app/cache/matplotlib \
    HOME=/app/cache/home \
    BLIP_MODEL=${BLIP_MODEL} \
    VIT_DETECTOR_MODEL=${VIT_DETECTOR_MODEL}

# Copy project configuration files
COPY pyproject.toml README.md ./

# Copy application code (perception, reasoning, realization)
COPY src/ ./src/

# Copy data layout (input, knowledge_base; data/output in .dockerignore)
COPY data/ ./data/

# Install the package in editable mode (no pip cache)
RUN pip install --no-cache-dir -e .

# Ensure full directory layout (volumes override at run).
# Under CACHE_DIR: Hugging Face (BLIP, CLIP), PyTorch Hub, PaddleOCR (~/.paddleocr via HOME),
# and other user-cache tools persist when the host mounts ./cache:/app/cache (docker-compose.yml).
RUN mkdir -p \
    /app/models/yolo \
    /app/models/vit \
    /app/models/detr \
    /app/models/caption_model \
    /app/models/classifier_model \
    /app/data/input/samples \
    /app/data/output/debug \
    /app/data/output/json \
    /app/data/knowledge_base \
    /app/cache \
    /app/cache/huggingface \
    /app/cache/huggingface/hub \
    /app/cache/huggingface/transformers \
    /app/cache/torch \
    /app/cache/xdg \
    /app/cache/paddle \
    /app/cache/paddleocr \
    /app/cache/ultralytics \
    /app/cache/matplotlib \
    /app/cache/home \
    /app/cache/home/.cache \
    /app/cache/home/.paddleocr \
    && chmod -R 755 /app

# Optional: preload BLIP and open-vocabulary detector weights into the same cache layout
# used at runtime. Keep disabled by default because these images can become very large.
RUN if [ "${PRELOAD_BLIP_MODEL}" = "1" ]; then \
      python -c "from transformers import BlipProcessor, BlipForConditionalGeneration; model='${BLIP_MODEL}'; BlipProcessor.from_pretrained(model); BlipForConditionalGeneration.from_pretrained(model)"; \
    fi && \
    if [ "${PRELOAD_VIT_DETECTOR}" = "1" ]; then \
      python -c "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection; model='${VIT_DETECTOR_MODEL}'; AutoProcessor.from_pretrained(model); AutoModelForZeroShotObjectDetection.from_pretrained(model)"; \
    fi

# Set CPU optimization environment variables
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    VECLIB_MAXIMUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    FLAGS_use_mkldnn=false \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# Expose port for potential API service
EXPOSE 8000

# Set working directory
WORKDIR /app

# Default command: interactive bash (override: docker-compose run pipeline python -m perception /app/data/input/samples/testing_image.jpg)
CMD ["/bin/bash"]
