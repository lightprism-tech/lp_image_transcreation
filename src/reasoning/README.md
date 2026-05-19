# Stage 2: Cultural Reasoning

Decides **what** to transform, **what** to preserve, and **how** to rewrite OCR text using Stage-1 perception output and the cultural knowledge graph.

**Full documentation:** [docs/REASONING.md](../../docs/REASONING.md)

## Quick summary

| Step | Component |
|------|-----------|
| Type inference | Label cues + KB tokens + `semantic_type` (see `reasoning.yaml` policy) |
| Decision | **LLM** (default `llm_first`) or **KG candidates then LLM** (`kg_first`) |
| Grounding | **KG** maps LLM target → catalog label + `visual_attributes` for Stage 3 |
| Output | `edit_plan`, `edit_text`, optional `region_replace` |

Default strategy is **`llm_first`**: the LLM proposes substitutes from scene context; the knowledge graph validates and grounds them before realization.

## LLM providers

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_key
LLM_MODEL=llama-3.3-70b-versatile
```

Override strategy:

```env
REASONING_POLICY_REASONING_STRATEGY=llm_first   # or kg_first
```

All policy keys: `src/reasoning/config/reasoning.yaml` and [docs/REASONING.md](../../docs/REASONING.md#configuration).

## CLI

```bash
python src/reasoning/main.py \
  --input data/output/my_run/json/japan_stage1_perception.json \
  --target India \
  --kg data/knowledge_base/countries_graph.json \
  --output data/output/my_run/json/japan_stage2_reasoning.json
```

Via full pipeline:

```bash
python src/main.py --img data/input/samples/japan.jpg --target India --no-cache --run-name my_run
```

## Input / output

- **Input:** Stage-1 JSON (`objects`, `text`, `scene`, `image_type`, bboxes).
- **Output:** Adapted scene graph plus `edit_plan` (`transformations`, `preservations`, `target_culture`), optional `edit_text` and `region_replace`.

`original_object` in transformations always matches the **perception label** (not a KB node name).

## Modules

| File | Role |
|------|------|
| `engine.py` | Core policy, `llm_first` / `kg_first`, plan generation |
| `knowledge_loader.py` | KB candidates, embeddings, avoid lists |
| `llm_client.py` | Groq/OpenAI JSON reasoning calls |
| `policy_config.py` | YAML + `REASONING_POLICY_*` env |
| `config/reasoning.yaml` | Policy and prompt templates |
| `schemas.py` | Pydantic models |
