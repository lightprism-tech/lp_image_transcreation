"""
Infographic icon semantics analyzer.
Uses lightweight region segmentation + CLIP typing for icon-like objects.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _is_infographic_like(image_type: Dict[str, Any]) -> bool:
    label = (image_type or {}).get("type", "")
    return label in {"poster", "infographic", "document", "social_media"}


def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) * 0.5, (float(bbox[1]) + float(bbox[3])) * 0.5)


def _cluster_by_center_distance(bboxes: List[List[float]], max_distance: float = 90.0) -> List[int]:
    centers = [_bbox_center(b) for b in bboxes]
    clusters = [-1] * len(centers)
    cid = 0
    for i, c in enumerate(centers):
        if clusters[i] != -1:
            continue
        clusters[i] = cid
        changed = True
        while changed:
            changed = False
            for j, cj in enumerate(centers):
                if clusters[j] != -1:
                    continue
                if any(
                    ((cj[0] - centers[k][0]) ** 2 + (cj[1] - centers[k][1]) ** 2) ** 0.5 <= max_distance
                    for k, ccid in enumerate(clusters)
                    if ccid == cid
                ):
                    clusters[j] = cid
                    changed = True
        cid += 1
    return clusters


class IconSemanticAnalyzer:
    """
    Dedicated icon classifier/segmenter for infographic-like images.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._device = None
        self._load_clip()

    def _load_clip(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = CLIPModel.from_pretrained(self.model_name).to(self._device)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            logger.info("IconSemanticAnalyzer CLIP loaded: %s", self.model_name)
        except Exception as e:
            logger.warning("IconSemanticAnalyzer CLIP unavailable, using fallback semantics: %s", e)
            self._model = None
            self._processor = None
            self._device = None

    def _classify_crop_semantics(self, crop: np.ndarray) -> Tuple[str, float]:
        prompts = [
            "an infographic icon",
            "a pictogram symbol",
            "a chart element",
            "a decorative shape",
            "a photo object",
        ]
        labels = ["icon", "symbol", "chart_element", "decorative", "photo_object"]
        if self._model is None or self._processor is None:
            h, w = crop.shape[:2]
            area = h * w
            if area < 9000:
                return "icon", 0.55
            return "photo_object", 0.45
        try:
            import torch
            from PIL import Image

            image = Image.fromarray(crop.astype(np.uint8))
            inputs = self._processor(text=prompts, images=image, return_tensors="pt", padding=True).to(self._device)
            with torch.no_grad():
                logits = self._model(**inputs).logits_per_image
                probs = logits.softmax(dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            return labels[idx], float(probs[idx])
        except Exception:
            return "icon", 0.50

    def analyze(self, image: np.ndarray, objects: List[Dict[str, Any]], image_type: Dict[str, Any]) -> Dict[str, Any]:
        if not _is_infographic_like(image_type):
            return {"enabled": False, "objects": [], "cluster_count": 0}
        icon_like_indices = []
        icon_like_bboxes = []
        object_semantics = []
        h, w = image.shape[:2]
        for idx, obj in enumerate(objects or []):
            bbox = obj.get("bbox") or []
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
            x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            semantic_type, score = self._classify_crop_semantics(crop)
            is_icon_like = semantic_type in {"icon", "symbol", "chart_element"} and score >= 0.35
            if is_icon_like:
                icon_like_indices.append(idx)
                icon_like_bboxes.append([x1, y1, x2, y2])
            object_semantics.append(
                {"object_index": idx, "semantic_type": semantic_type, "semantic_score": score, "icon_like": is_icon_like}
            )

        clusters = _cluster_by_center_distance(icon_like_bboxes) if icon_like_bboxes else []
        cluster_map: Dict[int, int] = {}
        for local_i, obj_i in enumerate(icon_like_indices):
            cluster_map[obj_i] = clusters[local_i]
        for entry in object_semantics:
            entry["icon_cluster_id"] = cluster_map.get(entry["object_index"], -1)
        cluster_count = (max(clusters) + 1) if clusters else 0
        return {
            "enabled": True,
            "objects": object_semantics,
            "cluster_count": cluster_count,
        }
