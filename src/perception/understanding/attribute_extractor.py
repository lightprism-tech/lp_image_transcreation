"""
Attribute Extractor
Extracts attributes like emotion, color, clothing, etc. from objects and captions
"""

import numpy as np
import re


class AttributeExtractor:
    """Extracts detailed attributes from detected objects and their captions"""
    
    def __init__(self):
        """Initialize attribute extraction"""
        self.color_keywords = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple', 'pink', 'brown', 'gray', 'grey']
        self.emotion_keywords = {
            'happy': ['happy', 'smiling', 'joyful', 'cheerful', 'laughing'],
            'sad': ['sad', 'crying', 'upset', 'unhappy', 'frowning'],
            'angry': ['angry', 'mad', 'furious'],
            'neutral': ['neutral', 'calm', 'serious']
        }
        self.clothing_keywords = ['shirt', 'pants', 'dress', 'jacket', 'hat', 'shoes', 'tie', 'suit', 'skirt']
    
    def extract(self, image: np.ndarray, bounding_boxes: list, captions: list = None) -> list:
        """
        Extract attributes for each detected object
        
        Args:
            image: Input image as numpy array (RGB)
            bounding_boxes: List of detected objects with bboxes
            captions: Optional list of captions from object_captioner
            
        Returns:
            List of attributes for each object:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'attributes': {
                        'colors': ['red', 'blue'],
                        'emotion': 'happy',
                        'clothing': ['shirt', 'pants'],
                        'pose': 'standing',
                        'age': 'adult',
                        'gender': 'unknown'
                    }
                },
                ...
            ]
        """
        attributes_list = []
        
        for i, bbox_info in enumerate(bounding_boxes):
            # Get caption if available
            caption = ''
            if captions and i < len(captions):
                caption = captions[i].get('caption', '')
            
            # Extract from image crop
            bbox = bbox_info.get('bbox', [])
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cropped = image[y1:y2, x1:x2]
            
            attributes_list.append({
                'bbox': bbox,
                'attributes': self._extract_attributes(cropped, caption, bbox_info)
            })
        
        return attributes_list
    
    def _extract_attributes(self, image_crop: np.ndarray, caption: str, bbox_info: dict) -> dict:
        """Extract all attributes for a single object"""
        caption_lower = caption.lower() if caption else ''
        class_name = bbox_info.get('class_name', '').lower()
        
        # Determine if this is a person object
        is_person = class_name == 'person'
        
        # Extract colors from caption (applicable to all objects)
        colors = [color for color in self.color_keywords if color in caption_lower]
        
        # Initialize person-specific attributes
        emotion = 'neutral'
        clothing = []
        pose = 'unknown'
        age = 'unknown'
        gender = 'unknown'
        
        # Only extract person-specific attributes for person objects
        if is_person:
            # Extract emotion
            for emotion_type, keywords in self.emotion_keywords.items():
                if any(keyword in caption_lower for keyword in keywords):
                    emotion = emotion_type
                    break
            
            # Extract clothing
            clothing = [item for item in self.clothing_keywords if item in caption_lower]
            
            # Extract pose (simple keyword matching)
            pose_keywords = ['standing', 'sitting', 'walking', 'running', 'lying']
            for p in pose_keywords:
                if p in caption_lower:
                    pose = p
                    break
            
            # Extract age/gender (simple keyword matching)
            # Use more precise matching to avoid false positives from context
            # e.g., "a man carrying a bag" should not make the bag male
            
            # For age detection, look for direct subject references
            if 'child' in caption_lower or 'kid' in caption_lower or 'baby' in caption_lower or 'boy' in caption_lower or 'girl' in caption_lower:
                age = 'child'
            elif 'elderly' in caption_lower or 'old person' in caption_lower or 'senior' in caption_lower:
                age = 'elderly'
            elif 'adult' in caption_lower or 'man' in caption_lower or 'woman' in caption_lower:
                age = 'adult'
            
            # For gender detection, prioritize direct subject references
            # Check for woman/female first as it's more specific
            if 'woman' in caption_lower or 'female' in caption_lower or 'girl' in caption_lower or 'lady' in caption_lower:
                gender = 'female'
            elif 'man' in caption_lower or 'male' in caption_lower or 'boy' in caption_lower:
                gender = 'male'
        
        return {
            'colors': colors,
            'emotion': emotion,
            'clothing': clothing,
            'pose': pose,
            'age': age,
            'gender': gender
        }
