<div align="center">

#  Image Transcreation Pipeline

### A Structured Pipeline for Context-Aware Image Transcreation
*Explicit Reasoning, Cultural Grounding, and Controlled Visual Realization*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

**Developed by [LightPrism.tech]**

[Quick Start](docs/QUICKSTART.md) • [AI Research](docs/AI.md) • [API Docs](src/perception/README.md) • [Contributing](#contributing)

</div>

---

## Overview

Image transcreation adapts visual content for different cultural contexts while preserving semantic intent and layout. The pipeline is organized as explicit stages: Perception, Reasoning, and Realization.

## Current updates (Apr 2026)

- Added unified full-pipeline entrypoint in `src/main.py` with two run modes:
  - full run: Stage 1 -> Stage 2 -> Stage 3
  - realization-only run from existing Stage-2 JSON via `--stage2-json`.
- Full pipeline CLI: only `--img` and `--target` are required; defaults are `--kg data/knowledge_base/countries_graph.json`, `--output-dir data/output`, `--run-name my_run`.
- Added stage and model caching controls in full pipeline CLI:
  - `--no-cache` disables stage output reuse
  - `--no-model-cache` disables in-process model/engine reuse.
- Improved Stage-2 output robustness by normalizing object fields (`class_name`, `original_class_name`, `bbox`) for downstream compatibility.
- Added Stage-3 fail-fast checks for non-actionable plans (unless `--allow-empty-plan` is set).
- Added better Stage-3 fallback rendering:
  - per-instance tinted bbox + label overlays when replacements contain bboxes
  - simple adaptation label overlay when no bbox-based action is available.

### Key Features (Current)

- Structured 3-stage pipeline with JSON handoff between stages.
- Grounded cultural reasoning with KB candidates and avoid lists.
- Infographic-aware policy to reduce unsafe/object-drift substitutions.
- OCR-region-aware text rewrite with layout constraints.
- Stage-3 mask-based edit/inpaint flow with quality gates.

---

## Quick Start

Compose loads `.env` from the project root. Copy `.env.example` to `.env` and set LLM/API keys before a full run (Stage 2 requires them). With the provided `docker-compose.yml`, downloaded hub and PaddleOCR weights persist under `./cache` on the host (see [Installation](#docker-recommended)).

### Docker: full pipeline (Stages 1–3)

From the repository root, put your image under `data/input/...` on the host (mounted at `/app/data` in the container). Outputs are written under `data/output/<run-name>/` on the host.

Full pipeline: only `--img` and `--target` are required. Defaults (from `/app` in the container): `--kg` is `data/knowledge_base/countries_graph.json`, `--output-dir` is `data/output`, `--run-name` is `my_run`. Override any of these only when needed.

```bash
docker-compose build
docker-compose run --rm pipeline python src/main.py \
  --img /app/data/input/samples/your_image.jpg \
  --target India
```

Explicit knowledge graph only (output dir and run name still use defaults):

```bash
docker-compose run --rm pipeline python src/main.py \
  --img /app/data/input/samples/your_image.jpg \
  --target India \
  --kg /app/data/knowledge_base/countries_graph.json
```

One line (all defaults):

```bash
docker-compose run --rm pipeline python src/main.py --img /app/data/input/samples/your_image.jpg --target India
```

Realization-only from an existing Stage-2 JSON (path matches a prior full run under `--run-name`, e.g. `my_run`):

```bash
docker-compose run --rm pipeline python src/main.py \
  --stage2-json /app/data/output/my_run/json/your_image_stage2_reasoning.json
```

Use a different `--run-name` if you want Stage 3 outputs in a separate folder from the default `my_run`.

### Docker: Stage 1 only

```bash
docker-compose build
docker-compose run --rm pipeline python -m perception /app/data/input/samples/testing_image.jpg
# Interactive shell: docker-compose run --rm pipeline /bin/bash
```

### Local

```bash
python -m venv venv
# Windows: venv\Scripts\activate   |   Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
pip install -e .
python src/main.py --img data/input/samples/your_image.jpg --target India
# Optional: --kg, --output-dir (default data/output), --run-name (default my_run)
# Stage 1 only: python -m perception data/input/samples/testing_image.jpg
```

**Full guide:** [docs/QUICKSTART.md](docs/QUICKSTART.md)

---

## Architecture

Current implemented flow:

```mermaid
graph LR
    A[Input Image] --> B[Stage 1: Perception]
    B --> C[Stage 2: Reasoning]
    C --> D[Stage 3: Realization]
    G[Knowledge Base] --> C
```

| Stage | Purpose | Status |
|-------|---------|--------|
| **1. Perception** | Structured scene, OCR, infographic semantics | Implemented |
| **2. Reasoning** | KB-grounded transformations and text rewrite plan | Implemented |
| **3. Realization** | Inpaint/edit + text rendering + quality gates | Implemented |

**Technical details:** [docs/AI.md](docs/AI.md)

---

## Project Structure

```
image-transcreation-pipeline/
├── src/
│   ├── main.py                  # Full pipeline CLI (Stages 1–3, optional Stage 3–only)
│   ├── perception/              # Stage 1: Context Extraction
│   │   ├── schemas/             # JSON schemas for validation
│   │   │   ├── object_schema.json
│   │   │   └── scene_schema.json
│   │   ├── config/              # Configuration and settings
│   │   ├── core/                # Pipeline orchestration
│   │   ├── detectors/           # Object and text detection
│   │   ├── understanding/      # Captioning and attributes
│   │   ├── ocr/                 # Text extraction
│   │   ├── builders/            # JSON output builders
│   │   └── utils/               # Bbox, image load, logging
│   ├── reasoning/               # Stage 2: Cultural Reasoning
│   │   ├── engine.py
│   │   ├── knowledge_loader.py
│   │   ├── llm_client.py
│   │   ├── main.py
│   │   └── schemas.py
│   └── realization/             # Stage 3: Visual Realization
│       ├── engine.py
│       ├── main.py
│       ├── models.py
│       ├── schema.py
│       └── README.md
├── data/
│   ├── input/                   # Input images (e.g. input/samples/)
│   ├── knowledge_base/          # Cultural knowledge graph (.json, .pkl, .jsonl)
│   └── output/
│       ├── debug/               # Debug visualizations (bboxes)
│       └── json/                # JSON output when saved via CLI
├── scripts/
│   └── knowledge_graph/         # KG generator (countries.json, generator.py)
├── models/                      # Model weights (auto-downloaded)
├── cache/                       # Hub + PaddleOCR caches (HF_HOME, HOME/.paddleocr; Docker bind-mount)
├── tests/                       # Unit and integration tests
└── docs/                        # Documentation (AI.md, QUICKSTART.md, etc.)
```

Root files: `docker-compose.yml`, `Dockerfile`, `pyproject.toml`, `requirements.txt`, `.env.example` (copy to `.env` for config; Docker loads `.env`).

**Note:** JSON outputs are returned by the API or saved to user-specified paths (e.g. `--output` or `data/output/json/`).

---

## Installation

### Prerequisites
- Python 3.8+ or Docker
- 8GB+ RAM
- 10GB disk space

### Docker (Recommended)

Docker Compose loads your local `.env` file into the container. Create it from the example before the first run:

```bash
# Copy .env.example to .env and edit if needed (API keys, thresholds, etc.)
cp .env.example .env   # Linux/Mac
# Windows: copy .env.example .env

docker-compose build

# Full pipeline (recommended): perception -> reasoning -> realization (defaults: kg, output-dir, run-name)
docker-compose run --rm pipeline python src/main.py --img /app/data/input/samples/your_image.jpg --target India

# Stage 1 only
docker-compose run --rm pipeline python -m perception /app/data/input/samples/testing_image.jpg

# Interactive shell
docker-compose run --rm pipeline /bin/bash
```

**Docker model and hub cache:** Compose mounts `./cache` to `/app/cache`. The Dockerfile and `docker-compose.yml` set `HF_HOME`, `TORCH_HOME`, and `HOME` so Hugging Face / Transformers, PyTorch Hub, and PaddleOCR (which uses `~/.paddleocr` from `HOME`) write under that volume. Local weights (YOLO, SAM, etc.) stay under `./models` (`MODELS_DIR`). The first run downloads each missing artifact once; later runs reuse `./cache` and `./models` if you keep those folders. Do not set `HOME` or `HF_HOME` in `.env` to paths under `/root` when using Docker, or caches will not persist across containers.

### Local Setup

```bash
# Clone repository
git clone <repository-url>
cd image-transcreation-pipeline

# Install
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

---

## Usage

### Full pipeline (CLI)

Runs Stage 1, then Stage 2 (LLM + knowledge graph), then Stage 3 (realization). Use `--no-cache` / `--no-model-cache` to force recomputation; see `python src/main.py --help`.

Required arguments for a full run: `--img` and `--target`. Optional: `--kg` (default `data/knowledge_base/countries_graph.json`), `--output-dir` (default `data/output`), `--run-name` (default `my_run`).

```bash
python src/main.py \
  --img data/input/samples/your_image.jpg \
  --target India
```

Docker (paths inside the container; same defaults under `/app`):

```bash
docker-compose run --rm pipeline python src/main.py --img /app/data/input/samples/your_image.jpg --target India
```

### Command Line (Stage 1)

```bash
# Process an image (output to stdout or --output path)
python -m perception data/input/samples/testing_image.jpg
python -m perception image.jpg --output data/output/json/result.json
```

### Python API

```python
import logging
from perception.main import main

logger = logging.getLogger(__name__)

# Process image
result = main("image.jpg")

# Access results
logger.info("%s", result["scene_description"])
logger.info("Objects: %s", len(result["objects"]))
logger.info("Text regions: %s", len(result["text_regions"]))
```

### Component-Level API

```python
from perception.detectors import ObjectDetector
from perception.ocr import OCREngine

# Use individual components
detector = ObjectDetector(confidence_threshold=0.5)
objects = detector.detect("image.jpg")

ocr = OCREngine(languages=['en'])
text = ocr.extract_text("image.jpg")
```

**Full API reference:** [src/perception/README.md](src/perception/README.md)

---

## Common Commands

| Action | Command |
|--------|---------|
| **Perception (Stage 1)** | `python -m perception data/input/samples/testing_image.jpg` |
| **Perception with output** | `python -m perception image.jpg --output data/output/json/result.json` |
| **Reasoning (Stage 2)** | `python src/reasoning/main.py --input data/output/json/Japan_stage1_perception.json --target India --kg data/knowledge_base/countries_graph.json --output data/output/json/Japan_stage2_reasoning.json` |
| **Realization (Stage 3)** | `python -m src.realization.main --img data/input/samples/Japan.jpg --plan data/output/json/Japan_stage2_reasoning.json --output data/output/final_india.png` |
| **Full pipeline** | `python src/main.py --img data/input/samples/Japan.jpg --target India` (optional: `--kg`, `--output-dir`, `--run-name`; defaults: `data/knowledge_base/countries_graph.json`, `data/output`, `my_run`) |
| **Realization-only from Stage 2** | `python src/main.py --stage2-json data/output/my_run/json/Japan_stage2_reasoning.json` (optional `--output-dir` / `--run-name`; same defaults) |
| **Docker: full pipeline** | `docker-compose run --rm pipeline python src/main.py --img /app/data/input/samples/Japan.jpg --target India` |
| **Docker: realization-only** | `docker-compose run --rm pipeline python src/main.py --stage2-json /app/data/output/my_run/json/Japan_stage2_reasoning.json` |
| **Docker: Stage 1 (perception)** | `docker-compose run --rm pipeline python -m perception /app/data/input/samples/testing_image.jpg` |
| **Docker: shell** | `docker-compose run --rm pipeline /bin/bash` |
| **Regenerate knowledge graph** | `python scripts/knowledge_graph/generator.py` |
| **Tests** | `pytest` or `pytest tests/unit/` |

---

## Output Format

- Stage-1 JSON: perception output (`*_stage1_perception.json`)
- Stage-2 JSON: adapted scene + `edit_plan` + optional `edit_text` (`*_stage2_reasoning.json`)
- Stage-3 image: realized image (`*_stage3_realized.png`)

### Debug Visualizations

Annotated images with bounding boxes, labels, and confidence scores saved to `data/output/debug/`

---

## Configuration

Use `.env` for runtime configuration. Recommended Stage-2 LLM setup:

```bash
# Copy template and edit
cp .env.example .env   # Linux/Mac  |  copy .env.example .env   # Windows
```

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_key
LLM_MODEL=llama-3.3-70b-versatile
```

Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `PERCEPTION_ENV` | Environment (development/production) | development |
| `DEBUG` | Enable debug output | true |
| `OBJECT_THRESHOLD` | Object detection confidence | 0.5 |
| `TEXT_THRESHOLD` | Text detection confidence | 0.6 |
| `MODELS_DIR` | Model weights directory (YOLO, SAM, …) | ./models |
| `CACHE_DIR` | Application cache directory | ./cache |
| `OUTPUT_DIR` | Output directory | ./data/output |
| `HF_HOME` | Hugging Face Hub cache (set in Docker image / Compose) | (local default: platform-specific) |
| `TORCH_HOME` | PyTorch Hub cache (set in Docker image / Compose) | (local default: platform-specific) |
| `HOME` | In Docker, set to `/app/cache/home` so PaddleOCR uses the mounted cache | (your system default) |
| `LLM_PROVIDER` | Reasoning provider (`groq` or `openai`) | groq/openai |
| `GROQ_API_KEY` | Groq API key | (set in .env) |
| `LLM_API_KEY` | Generic key fallback (Groq/OpenAI) | (set in .env) |

**All options:** [src/perception/config/settings.yaml](src/perception/config/settings.yaml)

---

## Development

### Running Tests

```bash
pytest                              # All tests
pytest tests/unit/                  # Unit tests only
pytest --cov=perception             # With coverage
```

### Code Style

```bash
black src/ tests/                   # Format
flake8 src/ tests/                  # Lint
mypy src/                           # Type check
```

---

## Deployment

### Docker Production

```bash
# Build and tag
docker build -t image-transcreation-pipeline:latest .

# One-off full pipeline (writes to host ./data/output via the data volume)
docker run --rm --env-file .env --memory="16g" --cpus="4.0" \
  -e HF_HOME=/app/cache/huggingface \
  -e TORCH_HOME=/app/cache/torch \
  -e HOME=/app/cache/home \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/cache:/app/cache \
  -w /app \
  image-transcreation-pipeline:latest \
  python src/main.py --img /app/data/input/samples/your_image.jpg --target India

# Long-running container (default CMD is bash; override as needed)
docker run -d --name image-transcreation-pipeline \
  --env-file .env \
  --memory="16g" --cpus="4.0" \
  -e HF_HOME=/app/cache/huggingface \
  -e TORCH_HOME=/app/cache/torch \
  -e HOME=/app/cache/home \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/cache:/app/cache \
  image-transcreation-pipeline:latest
```

On Windows PowerShell, replace `$(pwd)` with `${PWD}` (or use `%cd%` in `cmd.exe`) for volume paths.

### Cloud Platforms

- **AWS**: ECS/Fargate deployment
- **GCP**: Cloud Run deployment  
- **Azure**: ACI deployment

**Full deployment guide:** [docs/QUICKSTART.md#deployment](docs/QUICKSTART.md)

---

## Current Technical Notes

- Stage-1 now includes infographic-focused icon semantics and OCR style extraction.
- Stage-2 includes constrained OCR text rewrite candidate selection.
- Stage-3 includes richer quality gates (distribution + SSIM + optional CLIP local consistency).

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to set up your development environment, coding standards, and pull request process.

### Quick Steps

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get running in 3 minutes
- **[AI Research](docs/AI.md)** - Technical foundation and architecture
- **[Perception API](src/perception/README.md)** - Stage 1 implementation details
- **[Reasoning](src/reasoning/README.md)** - Stage 2 cultural reasoning
- **[Realization](src/realization/README.md)** - Stage 3 visual realization
- **[Knowledge Graph](docs/knowledge_graph.md)** - Knowledge base and generation
- **[LICENSE](LICENSE)** - MIT License with trademark notice

---

## References

1. **Khanuja et al.** (2024). *Image Transcreation for Cultural Relevance.* arXiv:2404.01247v3
2. **Radford et al.** (2021). *CLIP: Learning Transferable Visual Models.* ICML
3. **Li et al.** (2023). *BLIP-2: Bootstrapping Language-Image Pre-training.* NeurIPS

**Full references:** [docs/AI.md#references](docs/AI.md#references)

---

## License

MIT License - See [LICENSE](LICENSE) for details

**Trademark Notice:** "LightPrism" and "LightPrism.tech" are trademarks of LightPrism.tech

---

## Acknowledgments

- Ultralytics (YOLOv8)
- Salesforce (BLIP)
- OpenAI (CLIP)
- PaddlePaddle (PaddleOCR)
- Stability AI (Stable Diffusion)

---

