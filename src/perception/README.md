# Perception Package (Stage 1)

Stage-1 converts an input image into a structured scene JSON consumed by Stage-2 (reasoning) and Stage-3 (realization).

## Purpose in the full pipeline

Perception is responsible for:

- identifying image modality (`photo`, `infographic`, `document`, etc.)
- detecting objects and geometry (labels, confidence, bounding boxes)
- extracting OCR text with style hints (font size, colors, weight)
- producing infographic-specific signals for chart/table-like layouts.

Downstream stages depend on this output for grounded planning and edit execution. If Stage-1 misses or mislabels core entities, Stage-2/Stage-3 quality drops quickly.

## Output contract (what Stage-2/Stage-3 rely on)

The Stage-1 JSON includes:

- `image_type`
- `scene`
- `objects` (with `class_name`/`label`, confidence, bbox; optional semantic fields)
- `faces`
- `text.regions`
- `text.extracted` (OCR text + style metadata)
- `text.typography`
- `text.object_links`
- `quality_summary`
- `infographic_analysis`

Most important for later stages:

- **Object grounding:** object labels and bboxes are used for cultural substitution and inpainting targets.
- **Text transcreation:** OCR regions and style metadata are used for localized text replacement.
- **Infographic fallback:** dense text/icon signals help Stage-2 plan `region_replace` actions when object detection is sparse.

## Run Stage-1 only

```bash
python -m perception data/input/samples/Japan.jpg --output data/output/json/Japan_stage1_perception.json
```

## Run via full pipeline

```bash
python src/main.py \
  --img data/input/samples/Japan.jpg \
  --target India \
  --kg data/knowledge_base/countries_graph.json \
  --output-dir data/output \
  --run-name run_check
```

## Key files

- `main.py`: stage orchestration for detection, OCR, understanding, and JSON build.
- `detectors/object_detector.py`: object detection and base class labels.
- `detectors/text_detector.py`: text region detection for OCR.
- `understanding/icon_semantic_analyzer.py`: icon-like semantic typing and clustering.
- `segmentation/sam_segmenter.py`: SAM segmentation support from object boxes.
- `ocr/ocr_engine.py`: OCR extraction and style estimation.
- `ocr/text_postprocess.py`: OCR cleanup and typography aggregation.
- `utils/infographic.py`: infographic confidence calibration and analytics.
- `builders/scene_json_builder.py`: final Stage-1 JSON assembly.

## Model mapping

| Task             | Model used          |
| ---------------- | ------------------- |
| Detect objects   | YOLOv8x             |
| Scene captioning | BLIP                |
| Semantic cues    | CLIP                |
| OCR              | PaddleOCR           |
| Segmentation     | SAM (box-prompted)  |

## Operational notes

- First run may be slow due to model downloads.
- Debug images are enabled by default and written to `data/output/debug/`.
- Warmup can be toggled in `config/settings.yaml` (`enable_model_warmup`) to reduce first-image latency spikes.
- Additional toggles in `config/settings.yaml` include `enable_face_detection` and `enable_typography_summary`.
- OCR backend is PaddleOCR (Tesseract/EasyOCR are not used in this module).

## Docker caching reminder

Use the repo `docker-compose.yml` so cache directories are persisted (`./cache` -> `/app/cache`) and `HF_HOME`, `TORCH_HOME`, and `HOME` point to that mounted path. Without this, model/OCR caches are lost across container restarts.
