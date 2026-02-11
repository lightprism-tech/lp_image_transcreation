"""
Text Region Detector using PaddleOCR
Detects regions containing text in images
"""

import numpy as np
from paddleocr import PaddleOCR
from perception.config import settings


class TextDetector:
    """Detects text regions in images using PaddleOCR detection"""
    
    def __init__(self):
        """Initialize PaddleOCR text detector"""
        self.threshold = settings.TEXT_DETECTION_THRESHOLD
        self.ocr = None
        self._load_model()
    
    def _load_model(self):
        """Load PaddleOCR detection model"""
        try:
            # Initialize PaddleOCR with minimal parameters (defaults to CPU)
            self.ocr = PaddleOCR(lang='en')
            print(" PaddleOCR text detector loaded (CPU mode)")
        except Exception as e:
            print(f" Failed to load PaddleOCR: {e}")
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
        result = self.ocr.ocr(image)
        
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
