"""
Scene JSON Builder
Combines all pipeline outputs into final structured JSON
"""

import json
from pathlib import Path
from datetime import datetime


class SceneJSONBuilder:
    """Builds final structured JSON from all pipeline components"""
    
    def __init__(self):
        """Initialize JSON builder"""
        pass
    
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
        scene_json = {
            'metadata': {
                'image_path': str(image_path),
                'image_name': Path(image_path).name,
                'timestamp': datetime.now().isoformat(),
                'pipeline_version': '1.0'
            },
            'image_type': image_type,
            'scene': scene_description,
            'objects': self._build_objects(
                bounding_boxes,
                object_captions,
                object_attributes
            ),
            'faces': faces or [],
            'text': {
                'regions': text_boxes,
                'extracted': extracted_text,
                'full_text': ' '.join([t.get('text', '') for t in extracted_text]),
                'typography': typography or {},
                'object_links': object_text_links or [],
            },
            'quality_summary': quality_summary or {},
            'infographic_analysis': infographic_analysis or {'enabled': False},
        }
        
        return scene_json
    
    def _build_objects(
        self,
        bounding_boxes: list,
        captions: list,
        attributes: list
    ) -> list:
        """Combine object information into unified structure"""
        objects = []
        
        for i, bbox_info in enumerate(bounding_boxes):
            class_name = bbox_info.get('class_name', 'unknown')
            obj = {
                'id': i,
                'bbox': bbox_info.get('bbox', []),
                'class_name': class_name,
                'label': class_name,
                'original_class_name': class_name,
                'confidence': bbox_info.get('confidence', 0.0),
                'semantic_type': bbox_info.get('semantic_type'),
                'semantic_score': bbox_info.get('semantic_score'),
                'icon_cluster_id': bbox_info.get('icon_cluster_id', -1),
                'segmentation': bbox_info.get('segmentation', {'enabled': False, 'source': 'sam'}),
                'caption': captions[i].get('caption', '') if i < len(captions) else '',
                'attributes': attributes[i].get('attributes', {}) if i < len(attributes) else {}
            }
            objects.append(obj)
        
        return objects
    
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
