# Perception Package (Stage 1)

Stage-1 extracts structured scene data used by Stage-2 and Stage-3.

## What changed recently

- Image type classification now includes explicit `infographic` category.
- Added infographic-focused analysis:
  - icon cluster semantics
  - text density signals
  - semantic focus hints.
- Added OCR text-region confidence calibration.
- Added OCR style extraction improvements:
  - color/background estimation
  - font size/weight estimation
  - broader font identification coverage using discovered system fonts.
- Added icon semantic analyzer:
  - CLIP-based semantic typing for object crops
  - cluster IDs for icon-like objects.
- Added scene-level `quality_summary` to report object/text confidence aggregates, SAM availability, and segmentation readiness.
- Added text-to-object linking (`text.object_links`) to support downstream OCR-aware edits.

## Main outputs

Stage-1 JSON now includes:

- `image_type`
- `scene`
- `objects` (with optional `semantic_type`, `semantic_score`, `icon_cluster_id`)
- `faces` (detected face regions)
- `text.regions` (calibrated confidence)
- `text.extracted` (OCR text + style metadata)
- `text.typography` (font-family/weight/size summary)
- `text.object_links` (text-to-object association using bbox IoU)
- `quality_summary` (scene-level quality and SAM readiness metrics)
- `infographic_analysis`.

## Run

```bash
python -m perception data/input/samples/Japan.jpg --output data/output/json/Japan_stage1_perception.json
```

Run Stage-1 as part of full pipeline:

```bash
python src/main.py \
  --img data/input/samples/Japan.jpg \
  --target India \
  --kg data/knowledge_base/countries_graph.json \
  --output-dir data/output \
  --run-name run_check
```

## Key files

- `main.py`: orchestrates detectors, OCR, understanding, and builders.
- `detectors/face_detector.py`: face detection (OpenCV Haar cascade).
- `understanding/icon_semantic_analyzer.py`: dedicated icon semantic typing and clustering.
- `segmentation/sam_segmenter.py`: SAM-based object segmentation from detected boxes.
- `ocr/ocr_engine.py`: OCR extraction + style estimation + font identification.
- `ocr/text_postprocess.py`: text cleanup and typography summary.
- `utils/infographic.py`: confidence calibration and infographic analytics.
- `builders/scene_json_builder.py`: final Stage-1 JSON assembly.

## Current pipeline model mapping

| Task             | Best Model | Current in pipeline | Status |
| ---------------- | ---------- | ------------------- | ------ |
| Detect objects   | YOLOv8x    | YOLOv8x             | Using |
| Describe image   | BLIP       | BLIP                | Using |
| Understand image | CLIP       | CLIP                | Using |
| Read text        | OCR        | PaddleOCR           | Using |
| Segment objects  | SAM        | SAM (box-prompted)  | Using |

## Notes

- First run can be slow due to model downloads (YOLO, CLIP, BLIP, PaddleOCR).
- Duplicate Stage-1 log lines can happen due to logger handler setup; this does not affect results.
- Optional startup warmup runs tiny inference passes for YOLO, BLIP, OCR, SAM (and face detector when enabled) to reduce first-image latency spikes.
- Feature toggles are available in `config/settings.yaml`: `enable_face_detection`, `enable_typography_summary`, `enable_model_warmup`.
- Debug images are enabled by default (`output.save_debug_images: true`) and written under `data/output/debug/`.

## Not used in this module

- Tesseract OCR / EasyOCR are not used; OCR backend is PaddleOCR.
- Alternate object detectors like YOLOv9/YOLOv10 are not present in this module.
