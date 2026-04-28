"""
Object Captioner using BLIP
Generates captions/descriptions for detected objects
"""

import logging
import re

import numpy as np
import torch
from PIL import Image

from perception.config import settings
from perception.understanding.blip_model_manager import BLIPModelManager

logger = logging.getLogger(__name__)


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
        self.prompt_config = getattr(settings, "PERCEPTION_PROMPTS", {}).get("object_captioner", {})
        self.prompts = [
            str(prompt)
            for prompt in self.prompt_config.get("prompts", [""])
        ]
        self.max_new_tokens = int(self.prompt_config.get("max_new_tokens", 50))
        
        logger.info("ObjectCaptioner initialized with shared BLIP model")
    
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
            
            caption_candidates = self._generate_caption_candidates(cropped)
            caption_text = self._select_caption(caption_candidates)
            
            captions.append({
                'bbox': bbox,
                'caption': caption_text,
                'caption_candidates': caption_candidates,
                'confidence': 1.0 if caption_text else 0.0,
                'source': 'blip'
            })
        
        return captions

    def _generate_caption_candidates(self, image_crop: np.ndarray) -> list:
        """Generate multiple BLIP descriptions for a crop using configured prompts."""
        candidates = []
        seen = set()
        for prompt in self.prompts or [""]:
            caption = self._generate_caption(image_crop, prompt=prompt)
            normalized = caption.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(
                {
                    "prompt": prompt,
                    "caption": caption,
                    "source": "blip",
                }
            )
        return candidates
    
    def _generate_caption(self, image_crop: np.ndarray, prompt: str = "") -> str:
        """Generate caption for a single image crop using BLIP-2"""
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image_crop.astype('uint8'))
        
        # Prepare inputs
        if prompt:
            inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        # Generate caption
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        
        # Decode caption
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        return self._strip_prompt_echo(caption, prompt)

    def _select_caption(self, caption_candidates: list) -> str:
        """Choose the most descriptive non-empty BLIP caption."""
        if not caption_candidates:
            return ""
        usable = [
            item for item in caption_candidates
            if self._is_usable_caption(str(item.get("caption", "")), str(item.get("prompt", "")))
        ]
        if not usable:
            return ""
        ranked = sorted(
            usable,
            key=self._caption_score,
            reverse=True,
        )
        return str(ranked[0].get("caption", "")).strip()

    def _caption_score(self, item: dict) -> tuple:
        caption = str(item.get("caption", "")).strip()
        prompt = str(item.get("prompt", "")).strip()
        tokens = re.findall(r"[a-z][a-z0-9_-]*", caption.lower())
        unprompted_bonus = 3 if not prompt else 0
        return (unprompted_bonus + min(len(tokens), 12), len(caption))

    def _is_usable_caption(self, caption: str, prompt: str = "") -> bool:
        normalized = re.sub(r"\W+", " ", caption).strip().lower()
        if not normalized:
            return False
        if normalized.endswith("?") or normalized.startswith(("what object", "describe this", "list the")):
            return False
        if prompt:
            prompt_normalized = re.sub(r"\W+", " ", prompt).strip().lower()
            if normalized == prompt_normalized or prompt_normalized in normalized:
                return False
        return True

    @staticmethod
    def _strip_prompt_echo(generated_text: str, prompt: str) -> str:
        if not prompt:
            return generated_text.strip()
        generated = generated_text.strip()
        prompt_clean = prompt.strip()
        if generated.lower().startswith(prompt_clean.lower()):
            generated = generated[len(prompt_clean):].strip(" .,:;-")
        return generated.strip()
