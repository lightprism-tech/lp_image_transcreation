"""OCR modules for text extraction and post-processing."""
from perception.ocr.text_postprocess import TextPostProcessor

try:
    from perception.ocr.ocr_engine import OCREngine
except Exception:  # pragma: no cover - optional heavy dependency may be absent in lightweight test env
    OCREngine = None

__all__ = ["OCREngine", "TextPostProcessor"]
