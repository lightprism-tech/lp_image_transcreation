"""
OCR Engine using PaddleOCR
Extracts text from images with detection and recognition
"""

import numpy as np
from paddleocr import PaddleOCR
from perception.config import settings


class OCREngine:
    """Performs OCR using PaddleOCR (detection + recognition)"""
    
    def __init__(self, languages=None, use_gpu=None):
        """
        Initialize PaddleOCR engine
        
        Args:
            languages: List of language codes (default: from config)
            use_gpu: Whether to use GPU (default: from config)
        """
        self.languages = languages or settings.OCR_LANGUAGES
        self.use_gpu = use_gpu if use_gpu is not None else settings.OCR_GPU
        self.ocr = None
        self._load_reader()
    
    def _load_reader(self):
        """Load PaddleOCR reader (with detection + recognition)"""
        try:
            lang = self.languages[0] if isinstance(self.languages, list) else self.languages
            
            # Initialize PaddleOCR with minimal parameters (defaults to CPU)
            self.ocr = PaddleOCR(lang=lang)
            print(f"✓ PaddleOCR loaded (lang={lang}, CPU mode)")
        except Exception as e:
            print(f"✗ Failed to load PaddleOCR: {e}")
            raise
    
    def extract(self, image: np.ndarray, text_boxes: list = None) -> list:
        """
        Extract text from image using PaddleOCR
        
        Args:
            image: Input image as numpy array (RGB)
            text_boxes: Optional list of text regions (not used with PaddleOCR full pipeline)
            
        Returns:
            List of extracted text with locations:
            [
                {
                    'text': str,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'polygon': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                },
                ...
            ]
        """
        if self.ocr is None:
            raise RuntimeError("PaddleOCR not loaded")
        
        extracted_text = []
        
        # Run full OCR pipeline (parameters handled internally)
        result = self.ocr.ocr(image)
        
        if result and result[0]:
            for line in result[0]:
                # PaddleOCR returns: [polygon, (text, confidence)]
                polygon = line[0]
                text = line[1][0]
                confidence = line[1][1]
                
                # Convert polygon to bbox [x1, y1, x2, y2]
                x_coords = [p[0] for p in polygon]
                y_coords = [p[1] for p in polygon]
                bbox = [
                    min(x_coords),
                    min(y_coords),
                    max(x_coords),
                    max(y_coords)
                ]
                
                extracted_text.append({
                    'text': text,
                    'bbox': bbox,
                    'confidence': confidence,
                    'polygon': polygon
                })
        
        return extracted_text
