"""
Object Detector using YOLOv8x
Detects objects and returns bounding boxes
"""

import logging

import numpy as np
from ultralytics import YOLO

from perception.config import settings

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Detects objects in images using YOLOv8x"""
    
    def __init__(self, model_path=None):
        """
        Initialize YOLOv8x object detector
        
        Args:
            model_path: Path to YOLOv8x model weights
        """
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self.threshold = settings.OBJECT_DETECTION_THRESHOLD
        self.fallback_actionable_classes = {
            str(name).lower() for name in getattr(settings, "FALLBACK_ACTIONABLE_CLASSES", [])
        }
        self.model = None
        self.available = False
        self.status_reason = "not_initialized"
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8x model"""
        try:
            self.model = YOLO(str(self.model_path))
            self.available = True
            self.status_reason = "ready"
            logger.info("YOLOv8x model loaded from %s", self.model_path)
        except Exception as e:
            self.available = False
            self.status_reason = "model_init_failed"
            logger.error("YOLO init failed [model_init_failed]: %s", e)
            logger.info("Download with: yolo download model=yolov8x.pt")
            raise
    
    def detect(self, image: np.ndarray) -> list:
        """
        Detect objects in image using YOLOv8x
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            List of detections, each containing:
            {
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'class_id': int,
                'class_name': str
            }
        """
        if self.model is None:
            raise RuntimeError("YOLOv8x model not loaded [model_init_failed]")
        
        detections = self._run_inference(image, threshold=self.threshold)
        if detections:
            return detections

        # Adaptive fallback: if nothing is detected, relax threshold slightly
        # and keep only culturally actionable classes.
        fallback_threshold = max(0.2, float(self.threshold) * 0.6)
        fallback = self._run_inference(image, threshold=fallback_threshold)
        filtered = self._filter_actionable_fallback_detections(fallback)
        if filtered:
            logger.info(
                "Object detector fallback recovered %d object(s) at lower threshold %.2f",
                len(filtered),
                fallback_threshold,
            )
        return filtered

    def warmup(self) -> bool:
        """Run a tiny warmup inference to reduce first-request latency."""
        if self.model is None:
            self.status_reason = "model_init_failed"
            return False
        try:
            tiny = np.zeros((32, 32, 3), dtype=np.uint8)
            _ = self.model(tiny, verbose=False)
            logger.info("YOLO warmup complete")
            return True
        except Exception as e:
            self.status_reason = "inference_failed"
            logger.warning("YOLO warmup failed [inference_failed]: %s", e)
            return False

    def _run_inference(self, image: np.ndarray, threshold: float) -> list:
        """Run YOLO inference and return threshold-filtered detections."""
        results = self.model(image, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < threshold:
                    continue
                detections.append({
                    "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    "confidence": confidence,
                    "class_id": int(box.cls[0]),
                    "class_name": self.model.names[int(box.cls[0])],
                })
        return detections

    def _filter_actionable_fallback_detections(self, detections: list) -> list:
        """
        Keep top fallback detections likely useful for cultural transcreation.
        """
        if not detections:
            return []
        if not self.fallback_actionable_classes:
            return []

        # Prefer high confidence + actionable classes; limit noise to top 3.
        ranked = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
        kept = []
        for det in ranked:
            class_name = str(det.get("class_name", "")).lower()
            if class_name not in self.fallback_actionable_classes:
                continue
            kept.append(det)
            if len(kept) >= 3:
                break
        return kept
