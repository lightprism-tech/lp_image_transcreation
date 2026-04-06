# Quick Start Guide

Get the Image Transcreation Pipeline running in minutes.

## Prerequisites

- Python 3.8+ or Docker
- 8GB+ RAM recommended
- 10GB disk space (for models)

## Option 1: Docker (Recommended)

**Fastest way to get started:**

```bash
# 1. Clone repository
git clone <repository-url>
cd image-transcreation-pipeline

# 2. Build and run
docker-compose build
docker-compose run --rm pipeline python -m perception /app/data/input/samples/testing_image.jpg

# 3. View output: data/output/debug/ (images), data/output/json/ (if you used --output)
```

**Interactive shell:**
```bash
docker-compose run --rm pipeline /bin/bash
```

**Caches on disk:** Compose mounts `./cache` into the container. Hugging Face / Transformers and PyTorch Hub use `HF_HOME` and `TORCH_HOME` under `/app/cache`; PaddleOCR uses `~/.paddleocr`, and `HOME` is set to `/app/cache/home` so that path stays on the same volume. YOLO and SAM weights use `./models` (`MODELS_DIR`). After the first successful run, keep `./cache` and `./models` to avoid re-downloading.

## Option 2: Local Installation

### Step 1: Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd image-transcreation-pipeline

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install requirements
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Step 3: Run Pipeline

**Full pipeline (Stages 1–3) locally** — runs perception, then reasoning (LLM), then realization in one process. Create `.env` from `.env.example` and set your LLM provider and API key first; Stage 2 will fail without them.

Only `--img` and `--target` are required. Defaults: `--kg` is `data/knowledge_base/countries_graph.json`, `--output-dir` is `data/output`, `--run-name` is `my_run`. Override any of these when needed.

```bash
python src/main.py \
  --img data/input/samples/your_image.jpg \
  --target India
```

Optional explicit paths (same effect as defaults above):

```bash
python src/main.py \
  --img data/input/samples/your_image.jpg \
  --target India \
  --kg data/knowledge_base/countries_graph.json \
  --output-dir data/output \
  --run-name my_run
```

Artifacts are written under `data/output/my_run/` by default (for example `json/your_image_stage1_perception.json`, `json/your_image_stage2_reasoning.json`, and `images/your_image_stage3_realized.png`; exact basenames follow your input filename).

**Stage 1 only (perception):**

```bash
# Process an image
python -m perception data/input/samples/testing_image.jpg

# With custom output
python -m perception path/to/image.jpg --output data/output/json/result.json
```

### Step 4: Run Cultural Reasoning (Stage 2)

```bash
# Generate a transcreation plan (use Stage 1 JSON from --output)
python src/reasoning/main.py \
  --input data/output/json/image_analysis.json \
  --target "Japan" \
  --kg data/knowledge_base/countries_graph.json \
  --output data/output/plan.json
```

### Step 5: Run Visual Realization (Stage 3)

```bash
# Execute the transcreation plan
python -m src.realization.main \
  --img data/input/samples/testing_image.jpg \
  --plan data/output/plan.json \
  --output data/output/final_result.png
```

## First Run

**NOTE: First run takes longer** (often several minutes on CPU) while weights are fetched:
- YOLOv8x (order of 100MB+, under `MODELS_DIR`)
- BLIP captioning model (hundreds of MB; Hugging Face cache)
- CLIP (large ViT checkpoints are multi-GB; Hugging Face cache)
- PaddleOCR inference models (order of 10–20MB each; under `HOME/.paddleocr` locally, or `/app/cache/home/.paddleocr` in Docker)
- SAM and other `MODELS_DIR` assets as configured

**Docker:** Downloads land in the bind-mounted `./cache` and `./models` directories on the host (see Option 1). Re-running `docker-compose run` reuses them.

**Subsequent runs are much faster** (~5–15 seconds per image on CPU once caches are warm)

## Sample Images

Try the pipeline with the included sample images:

```bash
python -m perception data/input/samples/testing_image.jpg
python -m perception data/input/samples/testing_image2.jpg
```

## Configuration

### Basic Configuration

Create `.env` file:

```bash
# Copy example
cp .env.example .env

# Edit settings
PERCEPTION_ENV=development
DEBUG=true
SAVE_DEBUG=true
LOG_LEVEL=INFO
```

### Key Settings

```bash
# Detection thresholds
OBJECT_THRESHOLD=0.5      # Object detection confidence
TEXT_THRESHOLD=0.6        # Text detection confidence

# Paths
MODELS_DIR=./models       # Local weights (YOLO, SAM, …)
CACHE_DIR=./cache         # App cache; Docker also uses HF_HOME / TORCH_HOME / HOME under /app/cache
OUTPUT_DIR=./data/output  # Output location

# Optional (local runs): align with Docker if you want a single cache tree
# HF_HOME=./cache/huggingface
# TORCH_HOME=./cache/torch
# Use a dedicated HOME only if you know the side effects for your shell

# Performance
BATCH_SIZE=1              # Images per batch
OCR_GPU=false            # Enable GPU for OCR
```

## Usage Examples

### Command Line

```bash
# Basic usage
python -m perception image.jpg

# Specify output path
python -m perception image.jpg --output result.json

# Process multiple images
for img in data/input/samples/*.jpg; do
    python -m perception "$img"
done
```

### Python API

```python
import logging
from perception.main import main

logger = logging.getLogger(__name__)

# Process single image
result = main("path/to/image.jpg")

# Access results
logger.info("%s", result["scene_description"])
logger.info("Found %s objects", len(result["objects"]))
logger.info("Found %s text regions", len(result["text_regions"]))
```

### Advanced API Usage

```python
from perception.detectors import ObjectDetector
from perception.ocr import OCREngine
from perception.understanding import SceneSummarizer

# Use individual components
detector = ObjectDetector(confidence_threshold=0.5)
objects = detector.detect("image.jpg")

ocr = OCREngine(languages=['en'])
text = ocr.extract_text("image.jpg")

summarizer = SceneSummarizer()
description = summarizer.summarize("image.jpg", objects)
```

## Output Structure

The pipeline generates:

### 1. JSON Output

When using `--output`, JSON is written to the given path (e.g. `data/output/json/`).

```json
{
  "image_path": "image.jpg",
  "image_type": "photograph",
  "scene_description": "A bustling city street...",
  "objects": [
    {
      "id": 1,
      "label": "person",
      "confidence": 0.95,
      "bbox": [100, 200, 300, 500],
      "caption": "A person wearing a red jacket",
      "attributes": {"color": "red", "clothing": "jacket"}
    }
  ],
  "text_regions": [
    {
      "text": "STOP",
      "confidence": 0.98,
      "bbox": [50, 50, 150, 100]
    }
  ]
}
```

### 2. Debug Images (`data/output/debug/`)

Annotated images with:
- Bounding boxes around detected objects
- Text region highlights
- Confidence scores
- Object labels

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Solution: Reinstall package
pip install -e .
```

**Model Download Failures**
```bash
# Local: point Hugging Face and app cache to writable directories
export HF_HOME=/path/to/writable/huggingface
export TORCH_HOME=/path/to/writable/torch
export CACHE_DIR=./cache

# Docker: use Compose from the repo root so ./cache is mounted and HF_HOME,
# TORCH_HOME, and HOME are set (see README.md). For plain docker run, mount
# -v "$(pwd)/cache:/app/cache" and pass the same -e HF_HOME, TORCH_HOME, HOME
# as in the root README Deployment section.
```

**Out of Memory**
```bash
# Solution: Reduce image size in config/settings.yaml (image.max_size)
MAX_IMAGE_SIZE = (1280, 720)  # Instead of (1920, 1080)
```

**Docker Build Issues**
```bash
# Solution: Clean rebuild
docker-compose down
docker-compose build --no-cache
```

### Performance Tips

**CPU Optimization:**
```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

**GPU Acceleration:**
```bash
# Enable GPU for OCR
export OCR_GPU=true

# Ensure CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

## Next Steps

1. **Read the full documentation**: [README](../README.md)
2. **Explore the API**: [Perception README](../src/perception/README.md)
3. **Learn about the research**: [AI.md](AI.md)
4. **Deploy to production**: See [Deployment](#deployment) below
5. **Contribute**: See [Contributing Guidelines](../CONTRIBUTING.md)

## Getting Help

- **Documentation**: [Full README](../README.md)
- **AI Research**: [Technical Foundation](AI.md)
- **Perception API**: [API Documentation](../src/perception/README.md)
- **Issues**: [GitHub Issues](https://github.com/lightprism/image-transcreation/issues)
- **Discussions**: [GitHub Discussions](https://github.com/lightprism/image-transcreation/discussions)
- **Support**: support@lightprism.tech

---

## Deployment

### Docker Production Build

```bash
# Build (use Dockerfile in repo root, or your own -f Dockerfile.prod when available)
docker build -t transcreation:prod .

# Run with resource limits; include cache volume and env vars so hub/Paddle weights persist
docker run -d \
  --name transcreation-prod \
  --memory="16g" \
  --cpus="4.0" \
  -e HF_HOME=/app/cache/huggingface \
  -e TORCH_HOME=/app/cache/torch \
  -e HOME=/app/cache/home \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/cache:/app/cache \
  transcreation:prod
```

### Cloud Deployment

**AWS (ECS/Fargate):**
```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/transcreation:latest
```

**GCP (Cloud Run):**
```bash
# Deploy to Cloud Run
gcloud run deploy transcreation \
  --image gcr.io/<project>/transcreation:latest \
  --platform managed \
  --memory 16Gi
```

**Azure (ACI):**
```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group transcreation-rg \
  --name image-transcreation-pipeline \
  --image <registry>.azurecr.io/transcreation:latest \
  --cpu 4 --memory 16
```

---

**Ready to transcreate?** Run the full pipeline from perception to realization!
