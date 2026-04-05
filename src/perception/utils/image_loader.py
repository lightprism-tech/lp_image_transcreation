"""
Image Loader Utility
Loads and preprocesses images for the pipeline
"""

import cv2
import numpy as np
from pathlib import Path
from perception.config import settings


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (RGB format)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is not supported
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if image_path.suffix.lower() not in settings.IMAGE_FORMATS:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")
    
    # Load image using OpenCV
    image = cv2.imread(str(image_path))
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image


def resize_image(image: np.ndarray, max_size: tuple = None) -> np.ndarray:
    """
    Resize image to maximum dimensions while maintaining aspect ratio
    
    Args:
        image: Input image
        max_size: Maximum (width, height), default from config
        
    Returns:
        Resized image
    """
    if max_size is None:
        max_size = settings.MAX_IMAGE_SIZE
    
    height, width = image.shape[:2]
    max_width, max_height = max_size
    
    # Calculate scaling factor
    scale = min(max_width / width, max_height / height, 1.0)
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for model input
    
    Args:
        image: Input image
        
    Returns:
        Preprocessed image
    """
    # Resize if needed
    image = resize_image(image)
    
    # Additional preprocessing can be added here
    # - Normalization
    # - Color correction
    # - Noise reduction
    
    return image
