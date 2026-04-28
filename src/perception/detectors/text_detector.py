"""
Text Region Detector using PaddleOCR
Detects regions containing text in images
"""

import logging

import numpy as np
import cv2
from paddleocr import PaddleOCR

from perception.config import settings

logger = logging.getLogger(__name__)


class TextDetector:
    """Detects text regions in images using PaddleOCR detection"""
    
    def __init__(self):
        """Initialize PaddleOCR text detector"""
        self.threshold = settings.TEXT_DETECTION_THRESHOLD
        self.retry_upscale_factor = float(getattr(settings, "TEXT_DETECT_RETRY_UPSCALE_FACTOR", 1.5))
        self.max_retries = int(getattr(settings, "TEXT_DETECT_MAX_RETRIES", 1))
        self.ocr = None
        self._load_model()
    
    def _load_model(self):
        """Load PaddleOCR detection model"""
        try:
            # Keep angle classifier enabled to avoid orientation-warning noise.
            self.ocr = PaddleOCR(
                lang="en",
                use_gpu=bool(settings.OCR_GPU),
                use_angle_cls=bool(getattr(settings, "OCR_USE_ANGLE_CLS", True)),
                show_log=False,
            )
            logger.info(
                "PaddleOCR text detector loaded (gpu=%s, use_angle_cls=%s)",
                bool(settings.OCR_GPU),
                bool(getattr(settings, "OCR_USE_ANGLE_CLS", True)),
            )
        except Exception as e:
            logger.error("Failed to load PaddleOCR: %s", e)
            raise
    
    def detect(self, image: np.ndarray) -> list:
        """
        Detect text regions in image using PaddleOCR
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            List of text regions, each containing:
            {
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'polygon': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            }
        """
        if self.ocr is None:
            raise RuntimeError("PaddleOCR not loaded")
        
        # Run detection only (parameters handled by PaddleOCR internally)
        text_regions = self._extract_regions_from_result(self.ocr.ocr(image))
        if text_regions:
            return text_regions

        # Dynamic retry for low-text or low-contrast images.
        retries = max(0, self.max_retries)
        if retries <= 0:
            return []
        upscaled = image
        for _ in range(retries):
            h, w = upscaled.shape[:2]
            new_w = int(round(w * self.retry_upscale_factor))
            new_h = int(round(h * self.retry_upscale_factor))
            if new_w <= w or new_h <= h:
                break
            upscaled = cv2.resize(upscaled, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            retry_regions = self._extract_regions_from_result(self.ocr.ocr(upscaled))
            if retry_regions:
                sx = float(w) / float(new_w)
                sy = float(h) / float(new_h)
                return [self._rescale_region(r, sx, sy) for r in retry_regions]

        return []

    @staticmethod
    def _rescale_region(region: dict, sx: float, sy: float) -> dict:
        scaled = dict(region)
        bbox = scaled.get("bbox", [])
        if isinstance(bbox, list) and len(bbox) >= 4:
            scaled["bbox"] = [
                float(bbox[0]) * sx,
                float(bbox[1]) * sy,
                float(bbox[2]) * sx,
                float(bbox[3]) * sy,
            ]
        polygon = scaled.get("polygon", [])
        if isinstance(polygon, list):
            scaled["polygon"] = [[float(p[0]) * sx, float(p[1]) * sy] for p in polygon if isinstance(p, (list, tuple)) and len(p) >= 2]
        return scaled

    @staticmethod
    def _extract_regions_from_result(result) -> list:
        text_regions = []
        if result and result[0]:
            for detection in result[0]:
                # PaddleOCR returns: [polygon, (text, confidence)]
                # For detection-only mode, it may return just polygon or polygon with text
                # Handle both formats
                if isinstance(detection, (list, tuple)):
                    if len(detection) == 2 and isinstance(detection[1], tuple):
                        # Format: [polygon, (text, confidence)]
                        polygon = detection[0]
                    else:
                        # Format: just polygon (list of points)
                        polygon = detection
                else:
                    continue
                
                # Convert polygon to bbox [x1, y1, x2, y2]
                # Polygon is a list of [x, y] points
                x_coords = [float(p[0]) for p in polygon]
                y_coords = [float(p[1]) for p in polygon]
                bbox = [
                    min(x_coords),
                    min(y_coords),
                    max(x_coords),
                    max(y_coords)
                ]
                
                text_regions.append({
                    'bbox': bbox,
                    'confidence': 1.0,  # PaddleOCR doesn't return detection confidence separately
                    'polygon': polygon
                })

        return text_regions
