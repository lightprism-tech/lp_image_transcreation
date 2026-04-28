"""
Scene JSON Builder
Combines all pipeline outputs into final structured JSON
"""

import json
import re
from pathlib import Path
from datetime import datetime

from perception.config import settings


class SceneJSONBuilder:
    """Builds final structured JSON from all pipeline components"""
    
    def __init__(self):
        """Initialize JSON builder"""
        self.semantic_cfg = getattr(settings, "SEMANTIC_REGION_CONFIG", {})
    
    def build(
        self,
        image_path: str,
        image_type: dict,
        bounding_boxes: list,
        text_boxes: list,
        object_captions: list,
        object_attributes: list,
        scene_description: dict,
        extracted_text: list,
        faces: list = None,
        typography: dict = None,
        object_text_links: list = None,
        quality_summary: dict = None,
        infographic_analysis: dict = None,
        image_shape: tuple | None = None,
    ) -> dict:
        """
        Build final scene JSON
        
        Args:
            image_path: Path to input image
            image_type: Image classification result
            bounding_boxes: Detected objects
            text_boxes: Detected text regions
            object_captions: Object descriptions
            object_attributes: Object attributes
            scene_description: Scene summary
            extracted_text: OCR results
            
        Returns:
            Complete scene JSON structure
        """
        objects = self._build_objects(
            bounding_boxes,
            object_captions,
            object_attributes,
            extracted_text,
        )
        text_regions = self._build_text_regions(extracted_text, text_boxes)
        visual_regions = self._build_visual_regions(objects)
        layout = self._build_layout(image_type, visual_regions, text_regions, image_shape)

        scene_json = {
            'metadata': {
                'image_path': str(image_path),
                'image_name': Path(image_path).name,
                'timestamp': datetime.now().isoformat(),
                'pipeline_version': '1.0'
            },
            'image_type': image_type,
            'scene': scene_description,
            'visual_regions': visual_regions,
            'text_regions': text_regions,
            'layout': layout,
            'objects': objects,
            'faces': faces or [],
            'text': {
                'regions': text_boxes,
                'extracted': extracted_text,
                'full_text': ' '.join([t.get('text', '') for t in (extracted_text or [])]),
                'typography': typography or {},
                'object_links': object_text_links or [],
            },
            'quality_summary': self._augment_quality_summary(quality_summary or {}, visual_regions),
            'infographic_analysis': infographic_analysis or {'enabled': False},
        }
        
        return scene_json
    
    def _build_objects(
        self,
        bounding_boxes: list,
        captions: list,
        attributes: list,
        extracted_text: list | None = None,
    ) -> list:
        """Combine object information into unified structure"""
        objects = []
        
        for i, bbox_info in enumerate(bounding_boxes):
            caption = captions[i].get('caption', '') if i < len(captions) else ''
            original_class_name = bbox_info.get('class_name', '')
            derived_name = self._derive_name_from_caption(caption)
            fused_confidence = self._fuse_confidence(bbox_info, caption, extracted_text or [])
            class_name = derived_name if derived_name else original_class_name
            quality_flags = self._build_quality_flags(
                class_name=class_name,
                detector_label=original_class_name,
                caption=caption,
                fused_confidence=fused_confidence,
                caption_derived=bool(derived_name),
            )
            if "uncertain_label" in quality_flags:
                class_name = self._unknown_label()
            obj = {
                'id': i,
                'bbox': bbox_info.get('bbox', []),
                'class_name': class_name,
                'label': class_name,
                'original_class_name': original_class_name,
                'detector_label': original_class_name,
                'detector_backend': bbox_info.get('detector_backend', 'unknown'),
                'confidence': bbox_info.get('confidence', 0.0),
                'fused_confidence': fused_confidence,
                'quality_flags': quality_flags,
                'semantic_type': bbox_info.get('semantic_type'),
                'semantic_score': bbox_info.get('semantic_score'),
                'icon_cluster_id': bbox_info.get('icon_cluster_id', -1),
                'segmentation': bbox_info.get('segmentation', {'enabled': False, 'source': 'sam'}),
                'caption': caption,
                'caption_candidates': captions[i].get('caption_candidates', []) if i < len(captions) else [],
                'caption_source': captions[i].get('source', 'blip') if i < len(captions) else '',
                'attributes': attributes[i].get('attributes', {}) if i < len(attributes) else {}
            }
            objects.append(obj)
        
        return objects

    def _build_visual_regions(self, objects: list) -> list:
        regions = []
        for obj in objects or []:
            region_type = obj.get("semantic_type") or obj.get("class_name") or self._unknown_label()
            if "uncertain_label" in (obj.get("quality_flags") or []):
                region_type = self._unknown_label()
            regions.append(
                {
                    "id": obj.get("id"),
                    "type": region_type,
                    "description": obj.get("caption") or obj.get("label") or self._unknown_label(),
                    "semantic_role": self._semantic_role_for(obj),
                    "bbox": obj.get("bbox", []),
                    "confidence": obj.get("fused_confidence", obj.get("confidence", 0.0)),
                    "quality_flags": obj.get("quality_flags", []),
                    "source": {
                        "detector_backend": obj.get("detector_backend", "unknown"),
                        "detector_label": obj.get("detector_label", ""),
                        "caption_first": bool(obj.get("caption")),
                    },
                }
            )
        return regions

    def _build_text_regions(self, extracted_text: list, text_boxes: list) -> list:
        text_items = extracted_text or []
        heights = []
        for item in text_items:
            bbox = item.get("bbox") or []
            if len(bbox) >= 4:
                heights.append(max(0.0, float(bbox[3]) - float(bbox[1])))
        avg_height = sum(heights) / len(heights) if heights else 0.0

        regions = []
        for idx, item in enumerate(text_items):
            text = str(item.get("text", "")).strip()
            bbox = item.get("bbox", [])
            regions.append(
                {
                    "id": idx,
                    "role": self._infer_text_role(text, bbox, avg_height),
                    "text": text,
                    "bbox": bbox,
                    "confidence": item.get("confidence", 0.0),
                    "style": item.get("style", {}),
                }
            )

        if regions:
            return regions

        for idx, item in enumerate(text_boxes or []):
            regions.append(
                {
                    "id": idx,
                    "role": "text_block",
                    "text": "",
                    "bbox": item.get("bbox", []),
                    "confidence": item.get("confidence", 0.0),
                    "style": {},
                    "quality_flags": ["ocr_text_missing"],
                }
            )
        return regions

    def _build_layout(
        self,
        image_type: dict,
        visual_regions: list,
        text_regions: list,
        image_shape: tuple | None,
    ) -> dict:
        role_counts = {}
        for region in text_regions or []:
            role = region.get("role", "text_block")
            role_counts[role] = role_counts.get(role, 0) + 1

        visual_types = {}
        for region in visual_regions or []:
            region_type = region.get("type", self._unknown_label())
            visual_types[region_type] = visual_types.get(region_type, 0) + 1

        structure_parts = []
        if role_counts.get("title"):
            structure_parts.append("title")
        if visual_regions:
            structure_parts.append("visual region grid" if len(visual_regions) > 2 else "visual regions")
        if role_counts.get("heading"):
            structure_parts.append("headings")
        if role_counts.get("body") or role_counts.get("text_block"):
            structure_parts.append("description blocks")

        return {
            "structure": " + ".join(structure_parts) if structure_parts else "unstructured visual scene",
            "image_type": (image_type or {}).get("type", "unknown"),
            "text_role_counts": role_counts,
            "visual_region_type_counts": visual_types,
            "image_shape": list(image_shape) if image_shape else [],
        }

    def _derive_name_from_caption(self, caption: str) -> str:
        text = str(caption or "").strip().lower()
        if not text:
            return ""
        text = re.split(r"\b(?:with|on|inside|beside|next to|against|near)\b", text, maxsplit=1)[0]
        stopwords = {
            str(word).lower()
            for word in self.semantic_cfg.get("caption_name_stopwords", [])
        }
        words = [
            word
            for word in re.findall(r"[a-z][a-z0-9_-]*", text)
            if word not in stopwords
        ]
        return "_".join(words[:3])

    def _fuse_confidence(self, bbox_info: dict, caption: str, extracted_text: list) -> float:
        detector_score = float(bbox_info.get("confidence", 0.0) or 0.0)
        caption_score = self._caption_consistency_score(bbox_info.get("class_name", ""), caption)
        layout_score = self._ocr_layout_score(bbox_info.get("bbox", []), extracted_text)
        fused = (0.5 * detector_score) + (0.3 * caption_score) + (0.2 * layout_score)
        return round(max(0.0, min(1.0, fused)), 4)

    def _caption_consistency_score(self, detector_label: str, caption: str) -> float:
        if not caption:
            return float(self.semantic_cfg.get("caption_consistency_missing", 0.4))
        label_tokens = set(re.findall(r"[a-z][a-z0-9_-]*", str(detector_label).lower()))
        caption_tokens = set(re.findall(r"[a-z][a-z0-9_-]*", str(caption).lower()))
        if label_tokens and label_tokens.issubset(caption_tokens):
            return float(self.semantic_cfg.get("caption_consistency_match", 1.0))
        if label_tokens and label_tokens.intersection(caption_tokens):
            return float(self.semantic_cfg.get("caption_consistency_partial", 0.7))
        return float(self.semantic_cfg.get("caption_consistency_missing", 0.4))

    def _ocr_layout_score(self, bbox: list, extracted_text: list) -> float:
        if len(bbox) < 4 or not extracted_text:
            return 0.0
        threshold = float(self.semantic_cfg.get("ocr_overlap_weight_threshold", 0.05))
        best_iou = 0.0
        for item in extracted_text:
            text_bbox = item.get("bbox") or []
            if len(text_bbox) < 4:
                continue
            best_iou = max(best_iou, self._bbox_iou(bbox, text_bbox))
        return 1.0 if best_iou >= threshold else best_iou

    def _build_quality_flags(
        self,
        class_name: str,
        detector_label: str,
        caption: str,
        fused_confidence: float,
        caption_derived: bool,
    ) -> list:
        flags = []
        min_fused = float(self.semantic_cfg.get("min_fused_confidence", 0.45))
        if not class_name:
            flags.append("missing_caption_label")
        if fused_confidence < min_fused and not caption_derived:
            flags.append("uncertain_label")
        if not detector_label:
            flags.append("missing_detector_label")
        if caption_derived and detector_label:
            detector_tokens = set(re.findall(r"[a-z][a-z0-9_-]*", str(detector_label).lower()))
            caption_tokens = set(re.findall(r"[a-z][a-z0-9_-]*", str(caption).lower()))
            if detector_tokens and not detector_tokens.intersection(caption_tokens):
                flags.append("detector_caption_mismatch")
        return flags

    def _infer_text_role(self, text: str, bbox: list, avg_height: float) -> str:
        cfg = self.semantic_cfg.get("text_roles", {})
        word_count = len(str(text or "").split())
        height = 0.0
        if len(bbox) >= 4:
            height = max(0.0, float(bbox[3]) - float(bbox[1]))
        if avg_height > 0:
            relative_height = height / avg_height
            if relative_height >= float(cfg.get("title_min_relative_height", 1.35)):
                return "title"
            if relative_height >= float(cfg.get("heading_min_relative_height", 1.1)):
                return "heading"
        if word_count <= int(cfg.get("short_text_max_words", 5)):
            return "label"
        return "body"

    def _semantic_role_for(self, obj: dict) -> str:
        semantic_type = obj.get("semantic_type")
        if semantic_type:
            return f"{semantic_type} visual element"
        if obj.get("caption"):
            return "caption-described visual element"
        return "detected visual region"

    def _augment_quality_summary(self, quality_summary: dict, visual_regions: list) -> dict:
        out = dict(quality_summary or {})
        uncertain_count = sum(
            1
            for region in visual_regions or []
            if "uncertain_label" in (region.get("quality_flags") or [])
        )
        out["visual_region_count"] = len(visual_regions or [])
        out["uncertain_visual_region_count"] = uncertain_count
        return out

    def _unknown_label(self) -> str:
        return str(self.semantic_cfg.get("unknown_label", "unknown_visual_region"))

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
        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        if inter_area <= 0.0:
            return 0.0
        area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
        denom = area_a + area_b - inter_area
        return float(inter_area / denom) if denom > 0 else 0.0
    
    def save(self, scene_json: dict, output_path: str):
        """
        Save scene JSON to file
        
        Args:
            scene_json: Scene data
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scene_json, f, indent=2, ensure_ascii=False)
