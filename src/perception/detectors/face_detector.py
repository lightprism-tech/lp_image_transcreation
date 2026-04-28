"""
Face detector using OpenCV Haar Cascade.
Provides lightweight face-region detection for richer scene description.
"""

import logging

import cv2
import numpy as np
from perception.config import settings

logger = logging.getLogger(__name__)


class FaceDetector:
    """Detects human faces in images using OpenCV Haar cascades."""

    def __init__(self, scale_factor: float = None, min_neighbors: int = None):
        self.scale_factor = float(
            scale_factor if scale_factor is not None else getattr(settings, "FACE_DETECT_SCALE_FACTOR", 1.1)
        )
        self.min_neighbors = int(
            min_neighbors if min_neighbors is not None else getattr(settings, "FACE_DETECT_MIN_NEIGHBORS", 5)
        )
        self.min_size = tuple(getattr(settings, "FACE_DETECT_MIN_SIZE", (24, 24)))
        self.cascade = None
        self.available = False
        self.status_reason = "not_initialized"
        self._load_model()

    def _load_model(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            logger.warning("Face detector init failed [model_init_failed]: %s", cascade_path)
            self.cascade = None
            self.available = False
            self.status_reason = "model_init_failed"
            return
        self.available = True
        self.status_reason = "ready"
        logger.info("Face detector loaded from %s", cascade_path)

    def detect(self, image: np.ndarray) -> list:
        """
        Detect faces in an RGB image.

        Returns:
            [
                {"bbox": [x1, y1, x2, y2], "confidence": 1.0},
                ...
            ]
        """
        if self.cascade is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(int(self.min_size[0]), int(self.min_size[1])),
        )
        results = []
        for (x, y, w, h) in faces:
            results.append(
                {
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                    "confidence": 1.0,
                }
            )
        return results

    def warmup(self) -> bool:
        """Run tiny face detection pass for startup warmup."""
        if self.cascade is None:
            self.status_reason = "model_init_failed"
            return False
        try:
            tiny = np.zeros((32, 32, 3), dtype=np.uint8)
            _ = self.detect(tiny)
            logger.info("Face detector warmup complete")
            return True
        except Exception as e:
            self.status_reason = "inference_failed"
            logger.warning("Face detector warmup failed [inference_failed]: %s", e)
            return False
