"""
Object detector with optional YOLO + DETR + ViT hybrid inference.
"""

import logging
import re
from collections import Counter

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoProcessor, DetrForObjectDetection
from ultralytics import YOLO

from perception.config import settings

logger = logging.getLogger(__name__)
try:
    from transformers import Owlv2ForObjectDetection as ViTDetectorModel
except Exception:  # pragma: no cover - fallback for older transformers builds
    from transformers import OwlViTForObjectDetection as ViTDetectorModel


class ObjectDetector:
    """Detects objects using YOLO, optionally fused with DETR and ViT."""
    
    def __init__(self, model_path=None, context: dict | None = None):
        """
        Initialize YOLOv8x object detector
        
        Args:
            model_path: Path to YOLOv8x model weights
        """
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self.threshold = settings.OBJECT_DETECTION_THRESHOLD
        self.iou_threshold = float(getattr(settings, "YOLO_IOU_THRESHOLD", 0.45))
        self.image_size = int(getattr(settings, "YOLO_IMAGE_SIZE", 1280))
        self.max_det = int(getattr(settings, "YOLO_MAX_DET", 300))
        self.supplemental_enabled = bool(getattr(settings, "YOLO_SUPPLEMENTAL_ENABLED", True))
        self.supplemental_min_threshold = float(
            getattr(settings, "YOLO_SUPPLEMENTAL_MIN_THRESHOLD", 0.2)
        )
        self.supplemental_threshold_ratio = float(
            getattr(settings, "YOLO_SUPPLEMENTAL_THRESHOLD_RATIO", 0.6)
        )
        self.supplemental_max_candidates = int(
            getattr(settings, "YOLO_SUPPLEMENTAL_MAX_CANDIDATES", 5)
        )
        self.duplicate_iou = float(getattr(settings, "YOLO_DUPLICATE_IOU", 0.55))
        self.hybrid_duplicate_iou = float(getattr(settings, "HYBRID_DUPLICATE_IOU", self.duplicate_iou))
        self.fallback_actionable_classes = {
            str(name).lower() for name in getattr(settings, "FALLBACK_ACTIONABLE_CLASSES", [])
        }
        self.enable_detr = bool(getattr(settings, "ENABLE_DETR", False))
        self.detr_model_name = str(getattr(settings, "DETR_MODEL_NAME", "facebook/detr-resnet-50"))
        self.detr_threshold = float(getattr(settings, "DETR_CONFIDENCE_THRESHOLD", self.threshold))
        self.enable_vit = bool(getattr(settings, "ENABLE_VIT_DETECTOR", False))
        self.vit_model_name = str(
            getattr(settings, "VIT_DETECTOR_MODEL_NAME", "google/owlv2-large-patch14-ensemble")
        )
        self.vit_threshold = float(getattr(settings, "VIT_CONFIDENCE_THRESHOLD", 0.3))
        self.open_vocab_cfg = getattr(settings, "OPEN_VOCABULARY_DETECTOR", {})
        self.vit_labels = self._build_open_vocabulary_prompts(context or {})
        self.hybrid_mode = str(getattr(settings, "DETECTOR_HYBRID_MODE", "yolo_only")).lower()
        self.model = None
        self.detr_model = None
        self.detr_processor = None
        self.detr_available = False
        self.vit_model = None
        self.vit_processor = None
        self.vit_available = False
        self.available = False
        self.status_reason = "not_initialized"
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model and optional DETR/ViT models."""
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
        self._load_detr_model()
        self._load_vit_model()

    def _load_detr_model(self):
        """Load DETR model when enabled."""
        if not self.enable_detr:
            logger.info("DETR disabled; running YOLO only")
            return
        try:
            self.detr_processor = AutoImageProcessor.from_pretrained(self.detr_model_name)
            self.detr_model = DetrForObjectDetection.from_pretrained(self.detr_model_name)
            self.detr_model.eval()
            self.detr_available = True
            logger.info("DETR model loaded: %s", self.detr_model_name)
        except Exception as e:
            self.detr_available = False
            logger.warning("DETR init failed; falling back to YOLO only: %s", e)

    def _load_vit_model(self):
        """Load ViT detector model when enabled."""
        if not self.enable_vit:
            logger.info("ViT detector disabled; skipping ViT model load")
            return
        if not self.vit_labels:
            logger.warning("ViT detector enabled but no labels configured; skipping ViT model load")
            return
        try:
            self.vit_processor = AutoProcessor.from_pretrained(self.vit_model_name)
            self.vit_model = ViTDetectorModel.from_pretrained(self.vit_model_name)
            self.vit_model.eval()
            self.vit_available = True
            logger.info("ViT detector model loaded: %s (prompts=%d)", self.vit_model_name, len(self.vit_labels))
        except Exception as e:
            self.vit_available = False
            logger.warning("ViT detector init failed; falling back to YOLO/DETR: %s", e)
    
    def detect(self, image: np.ndarray) -> list:
        """Detect objects and return the final detector output list."""
        return self.detect_with_debug(image).get("final", [])

    def detect_with_debug(self, image: np.ndarray) -> dict:
        """
        Detect objects and return backend-specific views for debug visualization.
        
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

        yolo_detections = detections
        if self.supplemental_enabled:
            # Supplemental low-threshold pass for culturally actionable classes.
            fallback_threshold = max(
                self.supplemental_min_threshold,
                float(self.threshold) * self.supplemental_threshold_ratio,
            )
            fallback = self._run_inference(image, threshold=fallback_threshold)
            filtered = self._filter_actionable_fallback_detections(fallback)
            merged = self._merge_with_supplemental_detections(detections, filtered)
            if len(merged) > len(detections):
                logger.info(
                    "Object detector supplemental pass added %d actionable object(s) at lower threshold %.2f",
                    len(merged) - len(detections),
                    fallback_threshold,
                )
            yolo_detections = merged
        debug_views = {
            "yolo": list(yolo_detections),
            "detr": [],
            "vit": [],
        }
        if not self._should_run_hybrid():
            debug_views["fused"] = list(yolo_detections)
            return {"final": yolo_detections, "debug_views": debug_views}

        fused = list(yolo_detections)
        if self._should_run_detr_hybrid():
            detr_detections = self._run_detr_inference(image)
            fused = self._merge_with_hybrid_detections(fused, detr_detections)
            debug_views["detr"] = list(detr_detections)
        if self._should_run_vit_hybrid():
            vit_detections = self._run_vit_inference(image)
            fused = self._merge_with_hybrid_detections(fused, vit_detections)
            debug_views["vit"] = list(vit_detections)
        debug_views["fused"] = list(fused)
        if len(fused) > len(yolo_detections):
            logger.info(
                "Hybrid detection added %d object(s) (YOLO=%d, DETR=%d, ViT=%d, fused=%d)",
                len(fused) - len(yolo_detections),
                len(yolo_detections),
                len(debug_views["detr"]),
                len(debug_views["vit"]),
                len(fused),
            )
        return {"final": fused, "debug_views": debug_views}

    def warmup(self) -> bool:
        """Run a tiny warmup inference to reduce first-request latency."""
        if self.model is None:
            self.status_reason = "model_init_failed"
            return False
        try:
            tiny = np.zeros((32, 32, 3), dtype=np.uint8)
            _ = self.model(tiny, verbose=False, imgsz=self.image_size)
            if self._should_run_detr_hybrid():
                _ = self._run_detr_inference(tiny)
            if self._should_run_vit_hybrid():
                _ = self._run_vit_inference(tiny)
            logger.info("YOLO warmup complete")
            return True
        except Exception as e:
            self.status_reason = "inference_failed"
            logger.warning("YOLO warmup failed [inference_failed]: %s", e)
            return False

    def _run_inference(self, image: np.ndarray, threshold: float) -> list:
        """Run YOLO inference and return threshold-filtered detections."""
        results = self.model(
            image,
            verbose=False,
            imgsz=self.image_size,
            iou=self.iou_threshold,
            max_det=self.max_det,
        )
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

    def _run_detr_inference(self, image: np.ndarray) -> list:
        """Run DETR inference and return threshold-filtered detections."""
        if self.detr_model is None or self.detr_processor is None:
            return []
        with torch.no_grad():
            inputs = self.detr_processor(images=image, return_tensors="pt")
            outputs = self.detr_model(**inputs)
            target_sizes = torch.tensor([(image.shape[0], image.shape[1])])
            processed = self.detr_processor.post_process_object_detection(
                outputs=outputs,
                threshold=self.detr_threshold,
                target_sizes=target_sizes,
            )
        parsed = []
        if not processed:
            return parsed
        result = processed[0]
        labels = result.get("labels", [])
        scores = result.get("scores", [])
        boxes = result.get("boxes", [])
        for label, score, box in zip(labels, scores, boxes):
            class_id = int(label.item()) if hasattr(label, "item") else int(label)
            confidence = float(score.item()) if hasattr(score, "item") else float(score)
            bbox = [float(v) for v in box.tolist()]
            class_name = str(self.detr_model.config.id2label.get(class_id, f"class_{class_id}"))
            parsed.append(
                {
                    "bbox": bbox,
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name,
                    "detector_backend": "detr",
                }
            )
        return parsed

    def _run_vit_inference(self, image: np.ndarray) -> list:
        """Run ViT detector inference and return threshold-filtered detections."""
        if self.vit_model is None or self.vit_processor is None or not self.vit_labels:
            return []
        with torch.no_grad():
            inputs = self.vit_processor(
                text=self.vit_labels,
                images=image,
                return_tensors="pt",
            )
            outputs = self.vit_model(**inputs)
            target_sizes = torch.tensor([(image.shape[0], image.shape[1])])
            processed = self.vit_processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.vit_threshold,
            )
        parsed = []
        if not processed:
            return parsed
        result = processed[0]
        labels = result.get("labels", [])
        scores = result.get("scores", [])
        boxes = result.get("boxes", [])
        for label, score, box in zip(labels, scores, boxes):
            class_id = int(label.item()) if hasattr(label, "item") else int(label)
            if class_id < 0 or class_id >= len(self.vit_labels):
                continue
            confidence = float(score.item()) if hasattr(score, "item") else float(score)
            bbox = [float(v) for v in box.tolist()]
            class_name = self.vit_labels[class_id]
            parsed.append(
                {
                    "bbox": bbox,
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name,
                    "detector_backend": "vit",
                    "open_vocabulary_prompt": class_name,
                }
            )
        return parsed

    def _build_open_vocabulary_prompts(self, context: dict) -> list:
        """Build OWL-ViT/GroundingDINO-style prompts from VLM/OCR context."""
        configured_labels = [
            str(label).strip()
            for label in getattr(settings, "VIT_DETECTOR_LABELS", [])
            if str(label).strip()
        ]
        if not bool(self.open_vocab_cfg.get("use_context_prompts", True)):
            return configured_labels

        terms = self._extract_context_terms(context)
        if not terms:
            terms = [
                str(term).strip()
                for term in self.open_vocab_cfg.get("fallback_terms", [])
                if str(term).strip()
            ]
        terms.extend(configured_labels)

        templates = [
            str(template).strip()
            for template in self.open_vocab_cfg.get("prompt_templates", ["{term}"])
            if str(template).strip()
        ]
        if not templates:
            templates = ["{term}"]

        prompts = []
        seen_terms = set()
        seen_prompts = set()
        for term in terms:
            normalized_term = str(term).strip().lower()
            if not normalized_term or normalized_term in seen_terms:
                continue
            seen_terms.add(normalized_term)
            for template in templates:
                prompt = template.format(term=str(term).strip()).strip()
                if prompt and prompt.lower() not in seen_prompts:
                    prompts.append(prompt)
                    seen_prompts.add(prompt.lower())
        return prompts

    def _extract_context_terms(self, context: dict) -> list:
        cfg = self.open_vocab_cfg
        max_terms = int(cfg.get("max_context_terms", 12))
        stopwords = {str(word).lower() for word in cfg.get("term_stopwords", [])}
        text_parts = []

        image_type = context.get("image_type") or {}
        if image_type.get("type"):
            text_parts.append(str(image_type.get("type")))

        scene = context.get("scene") or {}
        for key in ("description", "setting", "mood", "activity"):
            value = scene.get(key)
            if value:
                text_parts.append(str(value))
        visual_context = scene.get("visual_context") or {}
        if isinstance(visual_context, dict):
            for value in (visual_context.get("generated_fields") or {}).values():
                if value:
                    text_parts.append(str(value))
            if visual_context.get("prompt_context"):
                text_parts.append(str(visual_context.get("prompt_context")))

        for item in context.get("extracted_text") or []:
            value = item.get("text") if isinstance(item, dict) else item
            if value:
                text_parts.append(str(value))

        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", " ".join(text_parts).lower())
        ranked = Counter(token for token in tokens if token not in stopwords)
        return [term for term, _ in ranked.most_common(max_terms)]

    def _filter_actionable_fallback_detections(self, detections: list) -> list:
        """
        Keep top fallback detections likely useful for cultural transcreation.
        """
        if not detections:
            return []
        if not self.fallback_actionable_classes:
            return []

        # Prefer high confidence + actionable classes; limit noise.
        ranked = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
        kept = []
        for det in ranked:
            class_name = str(det.get("class_name", "")).lower()
            if class_name not in self.fallback_actionable_classes:
                continue
            det["detector_pass"] = "supplemental_low_threshold"
            kept.append(det)
            if len(kept) >= self.supplemental_max_candidates:
                break
        return kept

    def _merge_with_supplemental_detections(self, base: list, supplemental: list) -> list:
        """
        Merge supplemental detections into base detections while removing near-duplicates.
        Duplicate rule: same class with IoU >= 0.55.
        """
        if not supplemental:
            return base or []
        merged = list(base or [])
        for cand in supplemental:
            if self._is_duplicate_detection(cand, merged):
                continue
            merged.append(cand)
        return merged

    def _is_duplicate_detection(self, candidate: dict, existing: list) -> bool:
        cand_cls = str(candidate.get("class_name", "")).lower()
        cand_bbox = candidate.get("bbox", [])
        for det in existing or []:
            det_cls = str(det.get("class_name", "")).lower()
            if cand_cls != det_cls:
                continue
            det_bbox = det.get("bbox", [])
            if ObjectDetector._bbox_iou(cand_bbox, det_bbox) >= self.duplicate_iou:
                return True
        return False

    def _merge_with_hybrid_detections(self, yolo_detections: list, detr_detections: list) -> list:
        """Merge DETR detections into YOLO results with duplicate suppression."""
        if not detr_detections:
            return yolo_detections or []
        merged = list(yolo_detections or [])
        for det in merged:
            det.setdefault("detector_backend", "yolo")
        for cand in detr_detections:
            if self._is_duplicate_hybrid(cand, merged):
                continue
            merged.append(cand)
        return merged

    def _is_duplicate_hybrid(self, candidate: dict, existing: list) -> bool:
        """Hybrid duplicate rule: same class and high IoU overlap."""
        cand_cls = str(candidate.get("class_name", "")).lower()
        cand_bbox = candidate.get("bbox", [])
        for det in existing or []:
            det_cls = str(det.get("class_name", "")).lower()
            if det_cls != cand_cls:
                continue
            det_bbox = det.get("bbox", [])
            if ObjectDetector._bbox_iou(cand_bbox, det_bbox) >= self.hybrid_duplicate_iou:
                return True
        return False

    def _should_run_hybrid(self) -> bool:
        """True when any secondary detector is enabled and selected in hybrid mode."""
        return self._should_run_detr_hybrid() or self._should_run_vit_hybrid()

    def _should_run_detr_hybrid(self) -> bool:
        if not self.enable_detr or not self.detr_available:
            return False
        return self.hybrid_mode in {"yolo_detr", "hybrid", "yolo_plus_detr", "yolo_detr_vit", "yolo_all"}

    def _should_run_vit_hybrid(self) -> bool:
        if not self.enable_vit or not self.vit_available:
            return False
        return self.hybrid_mode in {"yolo_vit", "hybrid", "yolo_plus_vit", "yolo_detr_vit", "yolo_all"}

    @staticmethod
    def _bbox_iou(box_a: list, box_b: list) -> float:
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
        denom = area_a + area_b - inter_area
        return float(inter_area / denom) if denom > 0 else 0.0
