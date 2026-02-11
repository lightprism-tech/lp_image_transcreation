# ==============================================================================
# Stage 1: Base Image with System Dependencies
# ==============================================================================
FROM python:3.10-slim as base

# Metadata
LABEL maintainer="your-email@example.com"
LABEL description="Stage-1 Perception Pipeline - AI-powered image analysis"
LABEL version="1.0"

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
FROM base as dependencies

WORKDIR /app

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 3: Application
# ==============================================================================
FROM dependencies as application

WORKDIR /app

# Copy project configuration files
COPY pyproject.toml README.md ./

# Copy application code (new src/perception structure)
COPY src/ ./src/

# Copy data schemas
COPY data/ ./data/

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Create necessary directories with proper permissions
RUN mkdir -p \
    models/yolo \
    models/caption_model \
    models/classifier_model \
    outputs/scene_json \
    outputs/debug_images \
    input_images \
    && chmod -R 755 /app

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

# Health check (optional - modify if you add an API)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD python -c "import sys; sys.exit(0)"

# Set working directory
WORKDIR /app

# Default command - interactive bash shell
CMD ["/bin/bash"]
