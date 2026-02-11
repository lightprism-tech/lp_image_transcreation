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
from perception.understanding.object_captioner import ObjectCaptioner
from perception.understanding.attribute_extractor import AttributeExtractor
from perception.understanding.scene_summarizer import SceneSummarizer
from perception.ocr.ocr_engine import OCREngine
from perception.builders.scene_json_builder import SceneJSONBuilder



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
    
    # Step 2: Detection
    logger.info("Step 2: Running detectors...")
    object_detector = ObjectDetector()
    text_detector = TextDetector()
    image_classifier = ImageTypeClassifier()
    
    bounding_boxes = object_detector.detect(image)
    text_boxes = text_detector.detect(image)
    image_type = image_classifier.classify(image)
    
    logger.info(f"  - Detected {len(bounding_boxes)} objects")
    logger.info(f"  - Detected {len(text_boxes)} text regions")
    logger.info(f"  - Image type: {image_type}")
    
    # Step 3: Understanding
    logger.info("Step 3: Understanding scene...")
    object_captioner = ObjectCaptioner()
    attribute_extractor = AttributeExtractor()
    scene_summarizer = SceneSummarizer()
    
    object_captions = object_captioner.caption(image, bounding_boxes)
    object_attributes = attribute_extractor.extract(image, bounding_boxes, object_captions)  # Pass captions
    scene_description = scene_summarizer.summarize(image)
    
    # Step 4: OCR
    logger.info("Step 4: Extracting text...")
    ocr_engine = OCREngine()
    extracted_text = ocr_engine.extract(image, text_boxes)
    
    # Step 4.5: Save debug visualization with bounding boxes
    if settings.SAVE_DEBUG_IMAGES:
        logger.info("Step 4.5: Saving debug visualization...")
        visualizer = DebugVisualizer()
        image_name = Path(image_path).stem
        visualizer.visualize_pipeline_results(image, bounding_boxes, text_boxes, image_name)
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
        extracted_text=extracted_text
    )
    
    # Save output
    # Note: output/json directory is not used per project requirements
    # User must specify output_path or handle the returned scene_json
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
