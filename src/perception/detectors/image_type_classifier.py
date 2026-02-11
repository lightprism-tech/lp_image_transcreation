"""
Image Type Classifier using CLIP
Zero-shot classification for ad, poster, UI, etc.
"""

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from perception.config import settings


class ImageTypeClassifier:
    """Classifies image type using CLIP zero-shot classification"""
    
    def __init__(self, model_name=None):
        """Initialize CLIP image classifier"""
        self.model_name = model_name or settings.CLIP_MODEL_NAME
        self.threshold = settings.CLASSIFICATION_THRESHOLD
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Image type categories with descriptive prompts
        self.classes = ['advertisement', 'poster', 'user interface', 'product photo', 
                       'social media post', 'document', 'other']
        self.class_labels = ['ad', 'poster', 'ui', 'product', 'social_media', 'document', 'other']
        
        self._load_model()
    
    def _load_model(self):
        """Load CLIP model for classification"""
        try:
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            print(f"✓ CLIP model loaded: {self.model_name} on {self.device}")
        except Exception as e:
            print(f"✗ Failed to load CLIP model: {e}")
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
        
        # Create text prompts for each class
        text_prompts = [f"a photo of a {cls}" for cls in self.classes]
        
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
        
        # Create scores dictionary
        all_scores = {label: float(prob) for label, prob in zip(self.class_labels, probs)}
        
        return {
            'type': self.class_labels[best_idx],
            'confidence': best_score,
            'all_scores': all_scores
        }
