"""
SAM-based object segmentation.
Uses detected object boxes as prompts and returns lightweight polygon masks.
"""

import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from perception.config import settings

logger = logging.getLogger(__name__)


class SAMSegmenter:
    """Segments objects with Segment Anything Model (SAM)."""

    def __init__(self, model_type: str = None, checkpoint_path: Path = None):
        self.enabled = bool(getattr(settings, "ENABLE_SAM_SEGMENTATION", True))
        self.model_type = model_type or settings.SAM_MODEL_TYPE
        self.checkpoint_path = Path(checkpoint_path or settings.SAM_CHECKPOINT_PATH)
        self.predictor = None
        self.available = False
        self.status_reason = "not_initialized"
        self._load_model()

    def _load_model(self) -> None:
        if not self.enabled:
            logger.info("SAM segmentation disabled by configuration")
            self.status_reason = "model_disabled"
            return

        if not self.checkpoint_path.exists():
            logger.warning("SAM checkpoint not found at %s; segmentation will be skipped", self.checkpoint_path)
            self.status_reason = "checkpoint_not_found"
            return

        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry

            model = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint_path))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device=device)
            self.predictor = SamPredictor(model)
            self.available = True
            self.status_reason = "ready"
            logger.info("SAM loaded (%s) from %s on %s", self.model_type, self.checkpoint_path, device)
        except Exception as e:
            logger.warning("Failed to initialize SAM segmenter: %s", e)
            self.predictor = None
            self.available = False
            self.status_reason = "model_init_failed"

    def get_status(self) -> Dict:
        """Return SAM runtime status for preflight diagnostics."""
        return {
            "enabled": self.enabled,
            "available": self.available,
            "reason": self.status_reason,
            "model_type": self.model_type,
            "checkpoint_path": str(self.checkpoint_path),
        }

    def segment(self, image: np.ndarray, objects: List[Dict]) -> List[Dict]:
        """
        Segment each detected object using its bounding box as SAM prompt.

        Returns one segmentation dict per input object.
        """
        if not objects:
            return []

        if not self.available or self.predictor is None:
            reason = self.status_reason if self.status_reason != "not_initialized" else "model_init_failed"
            return [self._empty_result(reason) for _ in objects]

        try:
            self.predictor.set_image(image.astype(np.uint8))
        except Exception as e:
            logger.warning("SAM failed to set image: %s", e)
            return [self._empty_result("inference_failed") for _ in objects]

        h, w = image.shape[:2]
        results: List[Dict] = []
        for obj in objects:
            bbox = obj.get("bbox", [])
            if len(bbox) < 4:
                results.append(self._empty_result("invalid_bbox"))
                continue

            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w - 1), x2), min(float(h - 1), y2)
            if x2 <= x1 or y2 <= y1:
                results.append(self._empty_result("invalid_bbox"))
                continue

            box = np.array([x1, y1, x2, y2], dtype=np.float32)
            try:
                masks, scores, _ = self.predictor.predict(box=box, multimask_output=False)
                if masks is None or len(masks) == 0:
                    results.append(self._empty_result("inference_failed"))
                    continue

                mask = masks[0].astype(np.uint8)
                score = float(scores[0]) if scores is not None and len(scores) > 0 else 0.0
                area_px = int(mask.sum())
                area_ratio = float(area_px / max(1, h * w))
                polygon = self._mask_to_polygon(mask)
                results.append(
                    {
                        "enabled": True,
                        "source": "sam",
                        "score": score,
                        "area_px": area_px,
                        "area_ratio": area_ratio,
                        "polygon": polygon,
                    }
                )
            except Exception as e:
                logger.warning("SAM segmentation failed for bbox %s: %s", bbox, e)
                results.append(self._empty_result("inference_failed"))

        return results

    @staticmethod
    def _mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
        """Extract a simplified polygon from binary mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        largest = max(contours, key=cv2.contourArea)
        epsilon = max(1.0, 0.01 * cv2.arcLength(largest, True))
        approx = cv2.approxPolyDP(largest, epsilon, True)
        return [[float(pt[0][0]), float(pt[0][1])] for pt in approx]

    @staticmethod
    def _empty_result(reason: str) -> Dict:
        return {
            "enabled": False,
            "source": "sam",
            "reason": reason,
            "score": 0.0,
            "area_px": 0,
            "area_ratio": 0.0,
            "polygon": [],
        }

    def warmup(self) -> bool:
        """Run tiny SAM warmup when model is available."""
        if not self.available or self.predictor is None:
            return False
        try:
            tiny = np.zeros((32, 32, 3), dtype=np.uint8)
            self.predictor.set_image(tiny)
            box = np.array([2.0, 2.0, 28.0, 28.0], dtype=np.float32)
            _ = self.predictor.predict(box=box, multimask_output=False)
            logger.info("SAM warmup complete")
            return True
        except Exception as e:
            self.status_reason = "inference_failed"
            logger.warning("SAM warmup failed [inference_failed]: %s", e)
            return False
