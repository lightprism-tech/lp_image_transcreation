"""
Image Type Classifier using CLIP
Zero-shot classification for ad, poster, UI, etc.
"""

import logging

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from perception.config import settings

logger = logging.getLogger(__name__)


class ImageTypeClassifier:
    """Classifies image type using CLIP zero-shot classification"""
    
    def __init__(self, model_name=None):
        """Initialize CLIP image classifier"""
        self.model_name = model_name or settings.CLIP_MODEL_NAME
        self.threshold = settings.CLASSIFICATION_THRESHOLD
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        prompts_cfg = getattr(settings, "PERCEPTION_PROMPTS", {}).get("image_type_classifier", {})
        self.classes = [str(item).strip() for item in prompts_cfg.get("classes", []) if str(item).strip()]
        self.class_labels = [str(item).strip() for item in prompts_cfg.get("labels", []) if str(item).strip()]
        self.prompt_template = str(prompts_cfg.get("template", "")).strip()
        if not self.classes:
            raise ValueError("Missing prompts.image_type_classifier.classes in perception.yaml")
        if not self.class_labels:
            raise ValueError("Missing prompts.image_type_classifier.labels in perception.yaml")
        if len(self.classes) != len(self.class_labels):
            raise ValueError("prompts.image_type_classifier.classes and labels must have same length")
        if "{class_name}" not in self.prompt_template:
            raise ValueError(
                "prompts.image_type_classifier.template must include '{class_name}' placeholder"
            )
        
        self._load_model()
    
    def _load_model(self):
        """Load CLIP model for classification"""
        try:
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            logger.info("CLIP model loaded: %s on %s", self.model_name, self.device)
        except Exception as e:
            logger.error("Failed to load CLIP model: %s", e)
            raise
    
    def classify(self, image: np.ndarray) -> dict:
        """
        Classify image type using CLIP zero-shot classification
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            Classification result:
            {
                'type': str,
                'confidence': float,
                'all_scores': dict  # All class probabilities
            }
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("CLIP model not loaded")
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image.astype('uint8'))
        
        # Create text prompts for each class from configurable template.
        text_prompts = [self.prompt_template.format(class_name=cls) for cls in self.classes]
        
        # Prepare inputs
        inputs = self.processor(
            text=text_prompts,
            images=pil_image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        # Find best match
        best_idx = np.argmax(probs)
        best_score = float(probs[best_idx])
        best_label = self.class_labels[best_idx]
        quality_flags = []
        if best_score < self.threshold:
            quality_flags.append("low_confidence_image_type")
        
        # Create scores dictionary
        all_scores = {label: float(prob) for label, prob in zip(self.class_labels, probs)}
        
        return {
            'type': best_label,
            'confidence': best_score,
            'all_scores': all_scores,
            'quality_flags': quality_flags,
        }
