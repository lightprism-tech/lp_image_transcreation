"""
BLIP Model Manager - Singleton Pattern
Provides shared access to BLIP model and processor to avoid duplicate loading
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from perception.config import settings
from threading import Lock


class BLIPModelManager:
    """
    Singleton class to manage shared BLIP model and processor.
    Ensures only one instance of the model is loaded in memory.
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        """Ensure only one instance exists (thread-safe singleton)"""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super(BLIPModelManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the model manager (only once)"""
        if self._initialized:
            return
        
        self.model_name = settings.BLIP2_MODEL_NAME
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = True
        
        # Load model immediately (eager loading)
        # Alternative: use lazy loading by calling _load_model() only when get_model() is called
        self._load_model()
    
    def _load_model(self):
        """Load BLIP model and processor (called only once)"""
        if self.model is not None and self.processor is not None:
            return  # Already loaded
        
        try:
            print(f"Loading shared BLIP model: {self.model_name}...")
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            print(f"✓ Shared BLIP model loaded on {self.device}")
        except Exception as e:
            print(f"✗ Failed to load BLIP model: {e}")
            raise
    
    def get_model(self):
        """Get the shared BLIP model"""
        if self.model is None:
            self._load_model()
        return self.model
    
    def get_processor(self):
        """Get the shared BLIP processor"""
        if self.processor is None:
            self._load_model()
        return self.processor
    
    def get_device(self):
        """Get the device (cuda/cpu) the model is running on"""
        return self.device
    
    @classmethod
    def reset(cls):
        """Reset the singleton (useful for testing)"""
        with cls._lock:
            if cls._instance is not None:
                if cls._instance.model is not None:
                    del cls._instance.model
                if cls._instance.processor is not None:
                    del cls._instance.processor
                cls._instance = None
