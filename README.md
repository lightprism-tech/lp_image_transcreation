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
  - text-region replacement fallback when `edit_text` actions are present
  - mock text rendering fallback (`_apply_mock_text_changes`) that draws translated strings into original text bboxes when the full inpaint backend is unavailable
  - simple adaptation label overlay only when no bbox/text action is available.
- Improved Stage-3 text rendering quality:
  - preserves OCR-inferred style family and weight more reliably
  - auto-fits translated text to original text bbox to reduce overflow/layout drift
  - high-contrast foreground color auto-selection against the sampled background.
- Added a text-aware post-realization quality gate with one automatic retry:
  - checks bbox occupancy, foreground/background contrast ratio, and color delta
  - retries with adjusted font size and color before accepting the edit
  - SSIM / CLIP-local checks are now disabled for text edits (configurable via `realization_config.json -> quality_gate.text_use_ssim` and `text_use_clip_local`).
- Improved transcreation prompt quality:
  - inpaint prompt refinement now enforces explicit target-culture grounding
  - non-grounded LLM prompt outputs are rejected and replaced with safe fallback prompts (`_is_culturally_grounded_prompt`).
- Strengthened Stage-2 reasoning:
  - explicit "Decision policy" and "KG-first planning rubric" in the LLM prompt
  - normalization enforces grounded, actionable decisions, mapping LLM `target_object` back to KG candidates (or the top candidate) instead of free-text output
  - confidence parsing and safer defaulting for weak / ambiguous LLM responses.
- Removed hardcoded object names, food lists, and weekday tokens from Stage-2:
  - cultural-type inference is driven entirely by KG `label_to_type` + detected attributes
  - food terms are discovered dynamically from KG `label_to_type` and KB substitutions for `FOOD` entries
  - infographic row anchors are detected geometrically from OCR regions via `_infer_row_label_bboxes` instead of a fixed day-of-week list.
- Added `region_replace` action type for infographic / icon-grid layouts:
  - `build_region_replacements` uses KG candidates plus an LLM pass to pick the best source food term and the best target replacement
  - `region_replace` actions flow through `adapt_plan_to_edit_format` and are honored by Stage-3 even when Stage-1 returned zero detected objects.

### Key Features (Current)

- Structured 3-stage pipeline with JSON handoff between stages.
- Grounded cultural reasoning with KB candidates and avoid lists; no hardcoded object/food/day lists.
- KG-first planning rubric with LLM-assisted candidate selection for both object and region replacements.
- Infographic-aware policy with dynamic row/region detection to reduce unsafe/object-drift substitutions.
- OCR-region-aware text rewrite with layout constraints.
- Style-family-aware text replacement (font family/weight/size fit) in realization.
- Stage-3 mask-based edit/inpaint flow with object and text quality gates, including an automatic retry pass for text edits.
- Robust multi-level Stage-3 fallback (bbox overlay -> mock text -> adaptation label) so runs produce a meaningful output even without a full inpaint backend.

---

## Quick Start

Compose loads `.env` from the project root. Copy `.env.example` to `.env` and set LLM/API keys before a full run (Stage 2 requires them). With the provided `docker-compose.yml`, downloaded BLIP/CLIP/OWL-ViT, Hugging Face, PyTorch, Ultralytics, and PaddleOCR artifacts persist under `./cache` on the host (see [Installation](#docker-recommended)).

### Docker: full pipeline (Stages 1–3)

From the repository root, put your image under `data/input/...` on the host (mounted at `/app/data` in the container). Outputs are written under `data/output/<run-name>/` on the host.

Full pipeline: only `--img` and `--target` are required. Defaults (from `/app` in the container): `--kg` is `data/knowledge_base/countries_graph.json`, `--output-dir` is `data/output`, `--run-name` is `my_run`. Override any of these only when needed.

```bash
docker-compose build
docker-compose run --rm pipeline python src/main.py \
  --img /app/data/input/samples/your_image.jpg \
  --target India
```

Rebuild the image after changing `Dockerfile`, `requirements.txt`, or Docker build args. Python source changes under `src/` are mounted by Compose and usually do not require a rebuild.

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

**Docker model and hub cache:** Compose mounts `./cache` to `/app/cache`. The Dockerfile and `docker-compose.yml` set `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME`, `PADDLE_HOME`, `PADDLEOCR_HOME`, `YOLO_CONFIG_DIR`, `MPLCONFIGDIR`, and `HOME` so Hugging Face / Transformers, PyTorch Hub, PaddleOCR, Ultralytics, and related runtime downloads write under that volume. Local weights (YOLO, SAM, etc.) stay under `./models` (`MODELS_DIR`). The first run downloads each missing artifact once; later runs reuse `./cache` and `./models` if you keep those folders. Do not set `HOME`, `HF_HOME`, or related Docker cache variables in `.env` to paths under `/root`, or caches will not persist across containers.

Optional build-time preloading is available via `.env`:

```bash
PRELOAD_BLIP_MODEL=1
PRELOAD_VIT_DETECTOR=1
docker-compose build pipeline
```

Keep these disabled for normal development because they increase image size. Runtime downloads are still cached and reused through `./cache`.

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
- Stage-2 JSON: adapted scene + `edit_plan` + optional `edit_text` + optional `region_replace` (`*_stage2_reasoning.json`)
- Stage-3 image: realized image (`*_stage3_realized.png`)

Stage-2 `edit_plan` action kinds:
- `replace` / `preserve` on detected objects (KG-grounded target).
- `edit_text` on OCR regions with translated strings and preserved style metadata.
- `region_replace` on geometrically-inferred regions (e.g. infographic cells) with a KG-selected target term, used when Stage-1 detection is sparse but OCR anchors exist.

Stage-3 fallback behavior (when no real generated image file is returned):
- If replace actions contain bbox -> draw per-instance replacement overlays.
- Else if `region_replace` entries exist -> overlay the target term into each inferred region.
- Else if `edit_text` exists -> apply bbox text replacements in mock mode with style-family and contrast preservation.
- Else -> apply adaptation label overlay.

Stage-3 text quality gate (see `data/config/realization_config.json -> quality_gate`):
- `text_use_ssim` (default `false`) and `text_use_clip_local` (default `false`) skip image-level similarity checks for text edits.
- Text edits are accepted based on bbox occupancy, contrast ratio, and color delta; one automatic retry is attempted with adjusted font size / color before falling back.

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
| `BLIP_MODEL` | BLIP caption/scene model used by Stage 1 | Salesforce/blip-image-captioning-large |
| `CLIP_MODEL` | CLIP model for image-type and semantic analysis | openai/clip-vit-large-patch14 |
| `VIT_DETECTOR_MODEL` | OWL-ViT/open-vocabulary detector model | google/owlv2-large-patch14-ensemble |
| `PRELOAD_BLIP_MODEL` | Docker build-time BLIP preload into image cache (`0`/`1`) | 0 |
| `PRELOAD_VIT_DETECTOR` | Docker build-time open-vocabulary detector preload (`0`/`1`) | 0 |
| `HF_HOME` | Hugging Face root cache (Docker: `/app/cache/huggingface`) | (local default: platform-specific) |
| `HF_HUB_CACHE` | Hugging Face Hub artifact cache (Docker: `/app/cache/huggingface/hub`) | (local default) |
| `TRANSFORMERS_CACHE` | Transformers cache (Docker: `/app/cache/huggingface/transformers`) | (local default) |
| `TORCH_HOME` | PyTorch Hub cache (Docker: `/app/cache/torch`) | (local default) |
| `XDG_CACHE_HOME` | General Linux cache root for libraries (Docker: `/app/cache/xdg`) | (local default) |
| `PADDLE_HOME` | Paddle cache (Docker: `/app/cache/paddle`) | (local default) |
| `PADDLEOCR_HOME` | PaddleOCR cache (Docker: `/app/cache/paddleocr`) | (local default) |
| `YOLO_CONFIG_DIR` | Ultralytics config/cache directory (Docker: `/app/cache/ultralytics`) | (local default) |
| `MPLCONFIGDIR` | Matplotlib config/cache directory (Docker: `/app/cache/matplotlib`) | (local default) |
| `HOME` | In Docker, set to `/app/cache/home` so tools using `~` stay under mounted cache | (your system default) |
| `LLM_PROVIDER` | Reasoning provider (`groq` or `openai`) | groq/openai |
| `GROQ_API_KEY` | Groq API key | (set in .env) |
| `LLM_API_KEY` | Generic key fallback (Groq/OpenAI) | (set in .env) |

Realization quality-gate tuning lives in `data/config/realization_config.json` under `quality_gate`:

| Key | Purpose | Default |
|-----|---------|---------|
| `text_use_ssim` | Enable SSIM check for text edits (usually noisy, off by default) | `false` |
| `text_use_clip_local` | Enable CLIP-local consistency check for text edits | `false` |
| `text_min_occupancy` | Minimum fraction of the text bbox that rendered glyphs must cover | tuned per-config |
| `text_min_contrast` | Minimum foreground/background contrast ratio accepted for a text edit | tuned per-config |
| `text_min_color_delta` | Minimum color delta between rendered text and sampled bg | tuned per-config |

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
  -e HF_HUB_CACHE=/app/cache/huggingface/hub \
  -e TRANSFORMERS_CACHE=/app/cache/huggingface/transformers \
  -e TORCH_HOME=/app/cache/torch \
  -e XDG_CACHE_HOME=/app/cache/xdg \
  -e PADDLE_HOME=/app/cache/paddle \
  -e PADDLEOCR_HOME=/app/cache/paddleocr \
  -e YOLO_CONFIG_DIR=/app/cache/ultralytics \
  -e MPLCONFIGDIR=/app/cache/matplotlib \
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
  -e HF_HUB_CACHE=/app/cache/huggingface/hub \
  -e TRANSFORMERS_CACHE=/app/cache/huggingface/transformers \
  -e TORCH_HOME=/app/cache/torch \
  -e XDG_CACHE_HOME=/app/cache/xdg \
  -e PADDLE_HOME=/app/cache/paddle \
  -e PADDLEOCR_HOME=/app/cache/paddleocr \
  -e YOLO_CONFIG_DIR=/app/cache/ultralytics \
  -e MPLCONFIGDIR=/app/cache/matplotlib \
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
- Stage-2 includes constrained OCR text rewrite candidate selection with stronger cultural rewrite rules (tone preservation, brand-name preservation, anti-stereotype guidance).
- Stage-2 decision-making is fully KG-driven: there are no hardcoded object names, food lists, or weekday/row tokens; cultural-type inference and food-term discovery flow from `label_to_type` and KB substitutions, and row anchors are inferred geometrically from OCR regions.
- Stage-2 emits an additional `region_replace` action kind, letting Stage-3 patch icon-grid / infographic regions even when Stage-1 detects zero objects in those cells. Target terms for `region_replace` are selected by an LLM pass restricted to KG candidates for the input culture.
- Stage-2 inpaint prompt refinement rejects non-grounded prompts (`_is_culturally_grounded_prompt`) and substitutes a deterministic culture-grounded fallback to avoid generic / off-target outputs.
- Stage-3 includes richer object-edit quality gates (distribution + SSIM + optional CLIP local consistency) and a separate, relaxed text-edit quality gate (bbox occupancy, contrast ratio, color delta) with one automatic retry and no SSIM/CLIP-local dependency.
- Stage-3 text edits now include better style-family matching, bold-variant selection, bbox-aware font fitting, and high-contrast foreground color picking against the sampled background.
- Stage-3 falls back gracefully through bbox overlays, `region_replace` overlays, and mock text rendering so a realized output is always produced even without a full inpainting backend.

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

