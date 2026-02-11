"""
Object Captioner using BLIP
Generates captions/descriptions for detected objects
"""

import numpy as np
import torch
from PIL import Image
from perception.understanding.blip_model_manager import BLIPModelManager
from perception.config import settings


class ObjectCaptioner:
    """Generates captions for detected objects using BLIP"""
    
    def __init__(self, model_name=None):
        """Initialize BLIP captioning model (uses shared model manager)"""
        self.model_name = model_name or settings.BLIP2_MODEL_NAME
        
        # Use shared model manager instead of loading separate model
        self.model_manager = BLIPModelManager()
        self.model = self.model_manager.get_model()
        self.processor = self.model_manager.get_processor()
        self.device = self.model_manager.get_device()
        
        print(f"✓ ObjectCaptioner initialized with shared BLIP model")
    
    def caption(self, image: np.ndarray, bounding_boxes: list) -> list:
        """
        Generate captions for each detected object
        
        Args:
            image: Input image as numpy array (RGB)
            bounding_boxes: List of detected objects with bboxes
            
        Returns:
            List of captions for each object:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'caption': str,
                    'confidence': float
                },
                ...
            ]
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("BLIP-2 model not loaded")
        
        captions = []
        for bbox_info in bounding_boxes:
            bbox = bbox_info.get('bbox', [])
            
            # Crop the object from the image
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                captions.append({
                    'bbox': bbox,
                    'caption': '',
                    'confidence': 0.0
                })
                continue
            
            # Generate caption for cropped object
            caption_text = self._generate_caption(cropped)
            
            captions.append({
                'bbox': bbox,
                'caption': caption_text,
                'confidence': 1.0  # BLIP-2 doesn't return confidence
            })
        
        return captions
    
    def _generate_caption(self, image_crop: np.ndarray) -> str:
        """Generate caption for a single image crop using BLIP-2"""
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image_crop.astype('uint8'))
        
        # Prepare inputs
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        # Generate caption
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=50)
        
        # Decode caption
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        return caption
