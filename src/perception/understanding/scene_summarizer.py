"""
Scene Summarizer using BLIP
Generates a global description of the entire scene
"""

import numpy as np
import torch
from PIL import Image
from perception.understanding.blip_model_manager import BLIPModelManager
from perception.config import settings


class SceneSummarizer:
    """Generates global scene descriptions using BLIP"""
    
    def __init__(self, model_name=None):
        """Initialize BLIP scene summarization model (uses shared model manager)"""
        self.model_name = model_name or settings.BLIP2_MODEL_NAME
        
        # Use shared model manager instead of loading separate model
        self.model_manager = BLIPModelManager()
        self.model = self.model_manager.get_model()
        self.processor = self.model_manager.get_processor()
        self.device = self.model_manager.get_device()
        
        print(f"✓ SceneSummarizer initialized with shared BLIP model")
    
    def summarize(self, image: np.ndarray) -> dict:
        """
        Generate global scene description
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            Scene summary:
            {
                'description': str,  # Overall scene description
                'setting': str,      # Indoor/outdoor, location type
                'mood': str,         # Overall mood/atmosphere
                'activity': str,     # Main activity happening
                'confidence': float
            }
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("BLIP-2 model not loaded")
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image.astype('uint8'))
        
        # Generate general scene description
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        
        description = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Extract structured information from description
        # (Simplified - could use additional prompting or NLP)
        setting = self._extract_setting(description)
        mood = self._extract_mood(description)
        activity = self._extract_activity(description)
        
        return {
            'description': description,
            'setting': setting,
            'mood': mood,
            'activity': activity,
            'confidence': 1.0
        }
    
    def _extract_setting(self, description: str) -> str:
        """Extract setting from description (simple keyword matching)"""
        description_lower = description.lower()
        if any(word in description_lower for word in ['indoor', 'room', 'office', 'home', 'building']):
            return 'indoor'
        elif any(word in description_lower for word in ['outdoor', 'outside', 'street', 'park', 'sky']):
            return 'outdoor'
        return 'unknown'
    
    def _extract_mood(self, description: str) -> str:
        """Extract mood from description"""
        description_lower = description.lower()
        if any(word in description_lower for word in ['happy', 'joyful', 'bright', 'cheerful']):
            return 'positive'
        elif any(word in description_lower for word in ['sad', 'dark', 'gloomy']):
            return 'negative'
        return 'neutral'
    
    def _extract_activity(self, description: str) -> str:
        """Extract main activity from description"""
        description_lower = description.lower()
        activities = ['walking', 'running', 'sitting', 'standing', 'eating', 'working', 'playing']
        for activity in activities:
            if activity in description_lower:
                return activity
        return 'unknown'
