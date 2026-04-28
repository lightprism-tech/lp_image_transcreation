"""
Main pipeline for Stage-1 Perception
Orchestrates the complete workflow from image loading to JSON output
"""

import argparse
from pathlib import Path

from perception.config import settings
from perception.utils.image_loader import load_image
from perception.utils.logger import setup_logger
from perception.utils.drawing_utils import DebugVisualizer
from perception.detectors.object_detector import ObjectDetector
from perception.detectors.text_detector import TextDetector
from perception.detectors.image_type_classifier import ImageTypeClassifier
from perception.detectors.face_detector import FaceDetector
from perception.segmentation.sam_segmenter import SAMSegmenter
from perception.understanding.object_captioner import ObjectCaptioner
from perception.understanding.attribute_extractor import AttributeExtractor
from perception.understanding.scene_summarizer import SceneSummarizer
from perception.understanding.blip_model_manager import BLIPModelManager
from perception.understanding.icon_semantic_analyzer import IconSemanticAnalyzer
from perception.ocr.ocr_engine import OCREngine
from perception.ocr.text_postprocess import TextPostProcessor
from perception.builders.scene_json_builder import SceneJSONBuilder
from perception.utils.infographic import calibrate_text_region_confidence, compute_infographic_analysis


def _bbox_iou(box_a: list, box_b: list) -> float:
    """Compute IoU for two [x1, y1, x2, y2] boxes."""
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    denom = (area_a + area_b - inter_area)
    return float(inter_area / denom) if denom > 0 else 0.0


def _build_object_text_links(objects: list, extracted_text: list, min_iou: float = 0.05) -> list:
    """Link OCR regions to most-overlapping detected object."""
    links = []
    for t_idx, text_item in enumerate(extracted_text or []):
        t_bbox = text_item.get("bbox", [])
        best_obj_idx = -1
        best_iou = 0.0
        for o_idx, obj in enumerate(objects or []):
            o_bbox = obj.get("bbox", [])
            iou = _bbox_iou(t_bbox, o_bbox)
            if iou > best_iou:
                best_iou = iou
                best_obj_idx = o_idx
        links.append(
            {
                "text_index": t_idx,
                "object_index": best_obj_idx if best_iou >= min_iou else -1,
                "overlap_iou": round(best_iou, 4),
            }
        )
    return links


def _build_quality_summary(
    objects: list,
    faces: list,
    text_regions: list,
    extracted_text: list,
    object_text_links: list,
    sam_status: dict,
) -> dict:
    """Build scene-level quality and readiness summary."""
    object_scores = [float(o.get("confidence", 0.0)) for o in (objects or [])]
    text_scores = [float(t.get("confidence", 0.0)) for t in (text_regions or [])]
    ocr_scores = [float(t.get("confidence", 0.0)) for t in (extracted_text or [])]
    linked_text_count = sum(1 for l in (object_text_links or []) if int(l.get("object_index", -1)) >= 0)
    segmented_count = sum(
        1
        for o in (objects or [])
        if bool((o.get("segmentation") or {}).get("enabled", False))
    )
    return {
        "object_count": len(objects or []),
        "face_count": len(faces or []),
        "text_region_count": len(text_regions or []),
        "ocr_text_count": len(extracted_text or []),
        "linked_text_count": linked_text_count,
        "object_avg_confidence": round(sum(object_scores) / len(object_scores), 4) if object_scores else 0.0,
        "text_region_avg_confidence": round(sum(text_scores) / len(text_scores), 4) if text_scores else 0.0,
        "ocr_avg_confidence": round(sum(ocr_scores) / len(ocr_scores), 4) if ocr_scores else 0.0,
        "sam_enabled": bool((sam_status or {}).get("enabled", False)),
        "sam_available": bool((sam_status or {}).get("available", False)),
        "sam_reason": str((sam_status or {}).get("reason", "")),
        "sam_segmented_object_count": segmented_count,
    }


def _run_model_warmup(
    object_detector: ObjectDetector,
    blip_manager: BLIPModelManager,
    ocr_engine: OCREngine,
    sam_segmenter: SAMSegmenter,
    face_detector: FaceDetector = None,
) -> None:
    """Run small warmup passes to reduce first-inference latency spikes."""
    object_detector.warmup()
    blip_manager.warmup()
    ocr_engine.warmup()
    sam_segmenter.warmup()
    if face_detector is not None:
        face_detector.warmup()


def main(image_path: str, output_path: str = None):
    """
    Run the complete Stage-1 Perception pipeline

    Workflow:
    1. Load image
    2. Detection: objects, text, image type
    3. Understanding: captions, attributes, scene description
    4. OCR: text extraction
    5. Build final JSON output
    """
    logger = setup_logger()
    logger.info(f"Starting Stage-1 Perception pipeline for: {image_path}")

    # Step 1: Load image
    logger.info("Step 1: Loading image...")
    image = load_image(image_path)

    # Step 2: Context-first analysis for model-driven detection
    logger.info("Step 2: Building image context...")
    text_detector = TextDetector()
    image_classifier = ImageTypeClassifier()
    face_detector = FaceDetector() if settings.ENABLE_FACE_DETECTION else None
    ocr_engine = OCREngine()
    sam_segmenter = SAMSegmenter()
    blip_manager = BLIPModelManager()
    scene_summarizer = SceneSummarizer()

    text_boxes = text_detector.detect(image)
    image_type = image_classifier.classify(image)
    faces = face_detector.detect(image) if face_detector is not None else []
    logger.info("Step 2.5: Extracting text for OCR-first context...")
    text_postprocessor = TextPostProcessor()
    extracted_text = ocr_engine.extract(image, text_boxes)
    text_boxes = calibrate_text_region_confidence(text_boxes, extracted_text)
    typography = (
        text_postprocessor.summarize_styles(extracted_text)
        if settings.ENABLE_TYPOGRAPHY_SUMMARY
        else {}
    )
    scene_description = scene_summarizer.summarize(
        image,
        image_type=image_type,
        extracted_text=extracted_text,
    )

    object_detector = ObjectDetector(
        context={
            "image_type": image_type,
            "scene": scene_description,
            "extracted_text": extracted_text,
        }
    )

    if settings.ENABLE_MODEL_WARMUP:
        logger.info("Step 2.6: Running model warmup...")
        _run_model_warmup(
            object_detector=object_detector,
            blip_manager=blip_manager,
            ocr_engine=ocr_engine,
            sam_segmenter=sam_segmenter,
            face_detector=face_detector,
        )

    detector_bundle = object_detector.detect_with_debug(image)
    bounding_boxes = detector_bundle.get("final", [])
    detector_views = detector_bundle.get("debug_views", {})
    sam_status = sam_segmenter.get_status()
    logger.info(
        "SAM status: enabled=%s available=%s reason=%s model_type=%s checkpoint=%s",
        sam_status.get("enabled"),
        sam_status.get("available"),
        sam_status.get("reason"),
        sam_status.get("model_type"),
        sam_status.get("checkpoint_path"),
    )
    segmentations = sam_segmenter.segment(image, bounding_boxes)
    for idx, segmentation in enumerate(segmentations):
        if 0 <= idx < len(bounding_boxes):
            bounding_boxes[idx]["segmentation"] = segmentation

    logger.info(f"  - Detected {len(bounding_boxes)} objects")
    logger.info(f"  - Detected {len(faces)} faces")
    logger.info(f"  - Detected {len(text_boxes)} text regions")
    logger.info(f"  - Image type: {image_type}")

    # Step 3: Region understanding
    logger.info("Step 3: Understanding scene...")
    object_captioner = ObjectCaptioner()
    attribute_extractor = AttributeExtractor()
    icon_analyzer = IconSemanticAnalyzer(model_name=settings.CLIP_MODEL_NAME)

    object_captions = object_captioner.caption(image, bounding_boxes)
    object_attributes = attribute_extractor.extract(image, bounding_boxes, object_captions)
    icon_semantics = icon_analyzer.analyze(image, bounding_boxes, image_type)
    for entry in icon_semantics.get("objects", []):
        idx = entry.get("object_index")
        if isinstance(idx, int) and 0 <= idx < len(bounding_boxes):
            bounding_boxes[idx]["semantic_type"] = entry.get("semantic_type")
            bounding_boxes[idx]["semantic_score"] = entry.get("semantic_score")
            bounding_boxes[idx]["icon_cluster_id"] = entry.get("icon_cluster_id", -1)

    object_text_links = _build_object_text_links(bounding_boxes, extracted_text)
    quality_summary = _build_quality_summary(
        objects=bounding_boxes,
        faces=faces,
        text_regions=text_boxes,
        extracted_text=extracted_text,
        object_text_links=object_text_links,
        sam_status=sam_status,
    )
    infographic_analysis = compute_infographic_analysis(image_type, bounding_boxes, extracted_text)
    infographic_analysis["icon_cluster_count"] = max(
        int(infographic_analysis.get("icon_cluster_count", 0) or 0),
        int(icon_semantics.get("cluster_count", 0) or 0),
    )

    # Step 4.5: Save debug visualization with bounding boxes
    if settings.SAVE_DEBUG_IMAGES:
        logger.info("Step 4.5: Saving debug visualization...")
        visualizer = DebugVisualizer()
        image_name = Path(image_path).stem
        visualizer.visualize_pipeline_results(
            image=image,
            objects=bounding_boxes,
            text_regions=text_boxes,
            image_name=image_name,
            detector_views=detector_views,
        )
        logger.info(f"  - Debug images saved to: {settings.DEBUG_IMAGES_DIR}")

    # Step 5: Build final JSON
    logger.info("Step 5: Building structured JSON...")
    json_builder = SceneJSONBuilder()
    scene_json = json_builder.build(
        image_path=image_path,
        image_type=image_type,
        bounding_boxes=bounding_boxes,
        text_boxes=text_boxes,
        object_captions=object_captions,
        object_attributes=object_attributes,
        scene_description=scene_description,
        extracted_text=extracted_text,
        faces=faces,
        typography=typography,
        object_text_links=object_text_links,
        quality_summary=quality_summary,
        infographic_analysis=infographic_analysis,
        image_shape=image.shape,
    )

    # Save output
    if output_path:
        json_builder.save(scene_json, output_path)
        logger.info(f"Pipeline complete! Output saved to: {output_path}")
    else:
        logger.info("Pipeline complete! No output path specified - returning scene_json only")

    return scene_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage-1 Perception Pipeline")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    args = parser.parse_args()
    main(args.image_path, args.output)
