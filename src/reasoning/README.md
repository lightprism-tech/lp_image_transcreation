# Stage 2: Cultural Reasoning

This folder decides what to transform and what to preserve using Stage-1 perception output plus the cultural knowledge graph.

## Current behavior

- Uses grounded KB candidates first (`countries_graph.json` + mappings).
- Supports infographic-aware policy:
  - avoids COCO-style substitutions in infographic/poster modes
  - preserves ambiguous person detections unless confidence/context is strong.
- Builds OCR region-aware text edit actions for Stage-3.
- Enforces rewrite constraints for text:
  - candidate set generation
  - layout-aware length validation using bbox and style metadata
  - best-candidate selection by constraint score.

## LLM providers

- Preferred runtime path: `groq` or `openai`.
- Azure path is intentionally disabled in current `llm_client.py` build to avoid deployment mismatch failures.

Example `.env`:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_key
LLM_MODEL=llama-3.3-70b-versatile
```

## Docker

Stage 2 calls an LLM only; it does not load the vision weights used in Stages 1 and 3. Run it inside the same Compose service as the rest of the pipeline (root `README.md` / `docs/QUICKSTART.md`) so paths and `.env` match the container layout (`/app/data`, etc.).

## CLI usage

```bash
python src/reasoning/main.py \
  --input data/output/json/Japan_stage1_perception.json \
  --target India \
  --kg data/knowledge_base/countries_graph.json \
  --output data/output/json/Japan_stage2_reasoning.json
```

If `--output` is omitted, output is auto-generated under `--output-dir/<run-name or timestamp>/json/`.

```bash
python src/reasoning/main.py \
  --input data/output/json/Japan_stage1_perception.json \
  --target India \
  --kg data/knowledge_base/countries_graph.json \
  --output-dir data/output \
  --run-name run_check
```

Optional:

- `--avoid item1 item2 ...` to inject explicit avoid-list constraints.

## Input and output

- Input: Stage-1 JSON (`metadata`, `image_type`, `scene`, `objects`, `text`).
- Output: same structure with adapted values plus:
  - `edit_plan` (`transformations`, `preservations`, `target_culture`)
  - optional `edit_text` for OCR regions.
- During application, transformed objects keep `original_class_name` and updated `class_name` so Stage-3 can map bbox-based edits reliably.

## Important module notes

- `engine.py`: core policy and plan generation logic.
- `llm_client.py`: provider calls and JSON parsing.
- `knowledge_loader.py`: candidate retrieval, avoid/style/sensitivity priors.
- `schemas.py`: Pydantic models for reasoning I/O.
