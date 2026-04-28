"""
OCR Engine using PaddleOCR
Extracts text from images with detection and recognition
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

from perception.config import settings

logger = logging.getLogger(__name__)

_BASE_FONT_CANDIDATES = [
    "arial.ttf",
    "arialbd.ttf",
    "calibri.ttf",
    "calibrib.ttf",
    "segoeui.ttf",
    "segoeuib.ttf",
    "times.ttf",
    "timesbd.ttf",
    "verdana.ttf",
    "verdanab.ttf",
]
_FONT_CACHE = None


def _discover_font_candidates(max_fonts: int = 300) -> list:
    global _FONT_CACHE
    if _FONT_CACHE is not None:
        return _FONT_CACHE
    candidates = list(_BASE_FONT_CANDIDATES)
    roots = [
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts"),
        Path("/Library/Fonts"),
        Path.home() / ".fonts",
        Path("fonts"),
    ]
    seen = set(f.lower() for f in candidates)
    for root in roots:
        if not root.exists():
            continue
        for ext in ("*.ttf", "*.otf", "*.ttc"):
            for p in root.rglob(ext):
                name = str(p)
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(name)
                if len(candidates) >= max_fonts:
                    _FONT_CACHE = candidates
                    return _FONT_CACHE
    _FONT_CACHE = candidates
    return _FONT_CACHE


def _identify_font_family(patch: np.ndarray, text: str, font_size: int, font_weight: str) -> str:
    """
    Identify best matching font by rendering OCR text over candidate fonts
    and selecting the one with minimum pixel reconstruction error.
    """
    text = (text or "").strip()
    if not text:
        return "arial.ttf"
    h, w = patch.shape[:2]
    gray = patch.mean(axis=2).astype(np.float32)
    target = gray / 255.0
    best_font = "arial.ttf"
    best_score = float("inf")
    ordered_fonts = _discover_font_candidates()
    if font_weight == "bold":
        ordered_fonts = [f for f in ordered_fonts if "bd" in f.lower() or f.lower().endswith("b.ttf")] + ordered_fonts
    for font_name in ordered_fonts:
        try:
            font = ImageFont.truetype(font_name, max(10, int(font_size)))
        except Exception:
            continue
        canvas = Image.new("L", (w, h), color=255)
        draw = ImageDraw.Draw(canvas)
        draw.text((2, 2), text, fill=0, font=font)
        rendered = np.array(canvas).astype(np.float32) / 255.0
        score = float(np.mean((target - rendered) ** 2))
        if score < best_score:
            best_score = score
            best_font = font_name
    return best_font


def _extract_region_style(image: np.ndarray, bbox: list, text: str) -> dict:
    """Estimate text-region style from OCR bbox for downstream text rendering."""
    try:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        if x2 <= x1 or y2 <= y1:
            return {}
        patch = image[y1:y2, x1:x2]
        if patch.size == 0:
            return {}

        gray = patch.mean(axis=2)
        fg_mask = gray < np.percentile(gray, 40)
        bg_mask = gray >= np.percentile(gray, 70)
        text_color = patch[fg_mask].mean(axis=0) if fg_mask.any() else np.array([20, 20, 20], dtype=np.float32)
        bg_color = patch[bg_mask].mean(axis=0) if bg_mask.any() else np.array([245, 245, 245], dtype=np.float32)

        text_len = max(1, len((text or "").strip()))
        est_font_size = max(10, int((y2 - y1) * 0.75))
        stroke_density = float(fg_mask.mean())
        font_weight = "bold" if stroke_density > 0.45 else "normal"
        font_family = _identify_font_family(patch, text, est_font_size, font_weight)

        return {
            "font_family": font_family,
            "font_weight": font_weight,
            "font_size": est_font_size,
            "text_color": [int(c) for c in text_color[:3]],
            "background_color": [int(c) for c in bg_color[:3]],
            "estimated_chars": text_len,
        }
    except Exception:
        return {}


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
        self.use_angle_cls = bool(getattr(settings, "OCR_USE_ANGLE_CLS", True))
        self.ocr = None
        self.available = False
        self.status_reason = "not_initialized"
        self._load_reader()
    
    def _load_reader(self):
        """Load PaddleOCR reader (with detection + recognition)"""
        try:
            lang = self.languages[0] if isinstance(self.languages, list) else self.languages
            
            self.ocr = PaddleOCR(
                lang=lang,
                use_gpu=bool(self.use_gpu),
                use_angle_cls=self.use_angle_cls,
                show_log=False,
            )
            self.available = True
            self.status_reason = "ready"
            logger.info(
                "PaddleOCR loaded (lang=%s, gpu=%s, use_angle_cls=%s)",
                lang,
                bool(self.use_gpu),
                self.use_angle_cls,
            )
        except Exception as e:
            self.available = False
            self.status_reason = "model_init_failed"
            logger.error("PaddleOCR init failed [model_init_failed]: %s", e)
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
            raise RuntimeError("PaddleOCR not loaded [model_init_failed]")
        
        extracted_text = []
        
        # Run full OCR pipeline (parameters handled internally)
        try:
            result = self.ocr.ocr(image)
        except Exception as e:
            self.status_reason = "inference_failed"
            logger.warning("PaddleOCR inference failed [inference_failed]: %s", e)
            return []
        
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
                    'polygon': polygon,
                    'style': _extract_region_style(image, bbox, text),
                })
        
        return extracted_text

    def warmup(self) -> bool:
        """Run tiny OCR pass to reduce first-request latency."""
        if self.ocr is None:
            self.status_reason = "model_init_failed"
            return False
        try:
            tiny = np.zeros((32, 32, 3), dtype=np.uint8)
            _ = self.ocr.ocr(tiny)
            logger.info("PaddleOCR warmup complete")
            return True
        except Exception as e:
            self.status_reason = "inference_failed"
            logger.warning("PaddleOCR warmup failed [inference_failed]: %s", e)
            return False
