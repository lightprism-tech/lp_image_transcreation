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

**NOTE: First run takes longer** (~2-5 minutes) due to model downloads:
- YOLOv8x (~130MB)
- BLIP-2 (~1GB)
- CLIP (~500MB)
- PaddleOCR (~100MB)

**Subsequent runs are fast** (~5-15 seconds per image on CPU)

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
MODELS_DIR=./models       # Model storage
OUTPUT_DIR=./data/output  # Output location

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
# Solution: Set cache directory
export HF_HOME=/path/to/cache
export CACHE_DIR=./cache
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
# Build optimized image
docker build -f Dockerfile.prod -t transcreation:prod .

# Run with resource limits
docker run -d \
  --name transcreation-prod \
  --memory="16g" \
  --cpus="4.0" \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
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
