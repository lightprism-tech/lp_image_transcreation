# Stage 3: Visual Realization

The Visual Realization module applies Stage-2 edit plans to images with mask-based edit/inpaint behavior, text-region rendering, and quality gates.

## Current implementation

- Object replacement:
  - prefers proper diffusers inpaint pipeline (`image + mask + prompt`)
  - falls back to FLUX image-edit endpoint (`image + mask + prompt`)
  - falls back to mock overlay if no inpaint backend is available.
- Text replacement:
  - uses OCR region bbox + style metadata (`font_family`, `font_weight`, `font_size`, color/background) where available.
- Actionability checks:
  - fails fast when plan has no actionable edits unless `--allow-empty-plan` is passed
  - warns when replacements exist but no bbox is available for inpainting.
- Quality gates:
  - local mean/std distribution checks
  - optional SSIM neighborhood gate
  - optional CLIP local-consistency gate.
- Fallback visuals:
  - per-instance tinted bbox overlays with labels when replacement actions include bboxes
  - simple adaptation-label overlay when only non-actionable edits are present.

## Input and Output

**Input**

- **Image**: Path to the source image (e.g. Stage 1 input or any image to adapt).
- **Plan**: Path to the Edit-Plan JSON. It can be the reasoning output (Stage 2) with `target_culture`, `transformations`, and `preservations`; the module converts this to the internal Edit-Plan shape.

**Output**: rendered image path. If no real inpaint backend is configured, output is mock-style.

**Small example**

Input plan (`data/output/plan_japan.json`):

```json
{
  "target_culture": "Japan",
  "transformations": [
    {
      "original_object": "Hamburger",
      "original_type": "FOOD",
      "target_object": "Onigiri",
      "rationale": "Culturally appropriate casual food for the setting.",
      "confidence": 0.95
    }
  ],
  "preservations": [
    { "original_object": "person", "rationale": "Universal." }
  ]
}
```

Command:

```bash
python -m src.realization.main \
  --img data/input/samples/testing_image.jpg \
  --plan data/output/plan_japan.json \
  --output data/output/final_japan.png
```

Output: an image at `data/output/final_japan.png` (culturally adapted per plan; in mock mode, the same image with a small "Adapted for Japan" label).

## Usage

```bash
python -m src.realization.main \
  --img <path_to_input_image> \
  --plan <path_to_edit_plan.json> \
  --output <path_to_output_image>
```

Optional flags:

- `--config <path_to_config.json>` to control inpainting and quality gates.
- `--allow-empty-plan` to allow no-op/mock output when plan has no actionable edits.

### Example

```bash
python -m src.realization.main \
  --img data/input/samples/testing_image.jpg \
  --plan data/output/plan_japan.json \
  --output data/output/final_japan.png
```

### Docker (interactive shell)

Run these from the **project root** (where `Dockerfile` and `docker-compose.yml` are). Compose loads your local `.env` via `env_file` and mounts `data`, `models`, `cache`, and `src` into the container. Hub and PaddleOCR artifacts are stored under the mounted `./cache` via `HF_HOME`, `TORCH_HOME`, and `HOME` (see root `README.md`).

**Docker Compose** — open a shell in the pipeline service:

```bash
docker compose build
docker compose run --rm pipeline /bin/bash
```

If your CLI still uses the hyphenated plugin, use `docker-compose build` and `docker-compose run --rm pipeline /bin/bash` instead.

Inside the container, use **`/app/...`** paths (not host-relative `./data/...`). Example:

```bash
python -m src.realization.main \
  --img /app/data/input/samples/testing_image.jpg \
  --plan /app/data/output/plan_japan.json \
  --output /app/data/output/final_japan.png
```

**Plain `docker run`** — after building the image, start bash with the same layout and env file as Compose:

```bash
docker build -t image-transcreation-pipeline:latest .

docker run --rm -it \
  --env-file .env \
  -e MODELS_DIR=/app/models \
  -e DATA_DIR=/app/data \
  -e CACHE_DIR=/app/cache \
  -e OUTPUT_DIR=/app/data/output \
  -e HF_HOME=/app/cache/huggingface \
  -e TORCH_HOME=/app/cache/torch \
  -e HOME=/app/cache/home \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/cache:/app/cache" \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd)/pyproject.toml:/app/pyproject.toml" \
  image-transcreation-pipeline:latest \
  /bin/bash
```

On Windows **PowerShell**, you can replace `"$(pwd)/..."` with `"${PWD}/..."` for each volume source path, or use absolute paths.

## JSON Schema

You can inspect the expected structure of the Edit-Plan by running:

```python
import json
import logging
from src.realization.schema import get_edit_plan_schema

logger = logging.getLogger(__name__)
logger.info("%s", json.dumps(get_edit_plan_schema(), indent=2))
```

## Plan input expectations

Use Stage-2 JSON with:

- `edit_plan` (`target_culture`, `transformations`, `preservations`)
- `objects` carrying `original_class_name`, `class_name`, and `bbox` for inpainting.

Realization converts this automatically to internal `EditPlan`.

For best results, ensure Stage-2 output contains object-level `bbox` and `original_class_name` for each intended replacement.

```bash
python -m src.reasoning.main --input data/output/json/stage1.json --target India --kg data/knowledge_base/countries_graph.json --output data/output/json/stage2_adapted_india.json

python -m src.realization.main \
  --img data/input/samples/testing_image2.jpg \
  --plan data/output/json/stage2_adapted_india.json \
  --output data/output/realized_india.png
```

## Why output may still be mock-like

- `use_inpainting` disabled
- no working diffusers/FLUX backend
- plan has no actionable bbox replacements
- quality gate rejected generated region and fallback path was used.

## Inpainting (real object replacement)

To **generate an updated image** with clothing/object replacement instead of the mock overlay:

1. Install the image model dependency: `pip install diffusers`
2. Pass a config that enables inpainting, e.g. `--config data/config/realization_config.json` with:
   ```json
   {
     "use_inpainting": true,
     "inpaint_model": "FLUX.2-pro",
     "inpaint_steps": 25,
     "use_llm_prompt_refinement": false,
     "inpaint_mask_pad_pct": 0.35
   }
   ```
   `use_llm_prompt_refinement` is optional: it uses your LLM (from .env: LLM_PROVIDER, LLM_API_KEY, LLM_MODEL) to improve the inpainting prompt text; the image is still generated by the diffusion model, not by the LLM.
3. Use a plan file that contains `edit_plan` and `objects` with `original_class_name` and `bbox` (Stage 2 output from the current pipeline).

If `use_inpainting` is false or diffusers is not installed, realization falls back to the mock overlay.

### FLUX endpoint notes

FLUX path expects an image-edit style endpoint and sends `image + mask + prompt`.
Use env vars:

- `AZURE_FLUX_EDIT_URL` (preferred) or `AZURE_FLUX_IMAGE_URL`
- `AZURE_OPENAI_API_KEY`

## Better visual impact during testing

- Prefer source images with multiple culture-replaceable objects (food, clothing, sports gear, signage), not only abstract infographics.
- Re-run Stage 2 and verify `edit_plan.transformations` and `edit_text` are non-empty before Stage 3.

### Recommended run sequence

```bash
python -m src.reasoning.main \
  --input data/output/json/stage1.json \
  --target India \
  --kg data/knowledge_base/countries_graph.json \
  --output data/output/json/stage2_adapted_india.json

python -m src.realization.main \
  --img data/input/samples/Japan.jpg \
  --plan data/output/json/stage2_adapted_india.json \
  --config data/config/realization_config.json \
  --output data/output/final_india.png
```

Before running realization, verify `stage2_adapted_india.json` has non-empty `edit_plan.transformations`.

## Key files

- `engine.py`: edit execution, text rendering, and quality gates.
- `inpaint.py`: diffusers and FLUX backends.
- `models.py`: edit action schema (`EditTextAction` includes optional style metadata).
- `schema.py`: Stage-2 to Stage-3 plan adaptation.
- `main.py`: CLI orchestration.
