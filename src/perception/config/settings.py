"""
Configuration file for Stage-1 Perception Pipeline
Contains model paths, thresholds, and pipeline settings
Supports environment variables via .env file
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# Base paths - adjusted for src/ layout
BASE_DIR = Path(__file__).parent.parent.parent.parent  # Go up to project root
PACKAGE_DIR = Path(__file__).parent.parent  # perception package directory

# Environment
ENV = os.getenv("PERCEPTION_ENV", "development")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Configurable paths (can be overridden via environment variables)
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", BASE_DIR / "cache"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "data" / "output"))

# Schema paths (for validation)
OBJECT_SCHEMA_PATH = PACKAGE_DIR / "schemas" / "object_schema.json"
SCENE_SCHEMA_PATH = PACKAGE_DIR / "schemas" / "scene_schema.json"

# Model paths
YOLO_MODEL_PATH = MODELS_DIR / "yolo" / "yolov8x.pt"  # YOLOv8x for object detection
BLIP2_MODEL_NAME = os.getenv("BLIP_MODEL", "Salesforce/blip-image-captioning-base")
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL", "openai/clip-vit-large-patch14")

# Detection thresholds (configurable via environment)
OBJECT_DETECTION_THRESHOLD = float(os.getenv("OBJECT_THRESHOLD", "0.5"))
TEXT_DETECTION_THRESHOLD = float(os.getenv("TEXT_THRESHOLD", "0.6"))
CLASSIFICATION_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD", "0.7"))

# Image processing settings
MAX_IMAGE_SIZE = (1920, 1080)
IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

# OCR settings (PaddleOCR)
OCR_LANGUAGES = ['en']  # PaddleOCR language
OCR_GPU = os.getenv("OCR_GPU", "false").lower() == "true"

# Output settings
SAVE_DEBUG_IMAGES = os.getenv("SAVE_DEBUG", "true").lower() == "true"
DEBUG_IMAGES_DIR = OUTPUT_DIR / "debug"
# SCENE_JSON_DIR removed - JSON outputs should not be stored in output/json

# Pipeline settings
ENABLE_SCENE_GRAPH = False  # Optional scene graph generation
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "pipeline.log"

# Create output directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
# SCENE_JSON_DIR directory creation removed
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Settings object for easier imports
class Settings:
    """Settings class for easier access to configuration."""
    BASE_DIR = BASE_DIR
    PACKAGE_DIR = PACKAGE_DIR
    ENV = ENV
    DEBUG = DEBUG
    
    MODELS_DIR = MODELS_DIR
    DATA_DIR = DATA_DIR
    CACHE_DIR = CACHE_DIR
    OUTPUT_DIR = OUTPUT_DIR
    
    # Schema paths
    OBJECT_SCHEMA_PATH = OBJECT_SCHEMA_PATH
    SCENE_SCHEMA_PATH = SCENE_SCHEMA_PATH
    
    YOLO_MODEL_PATH = YOLO_MODEL_PATH
    BLIP2_MODEL_NAME = BLIP2_MODEL_NAME
    CLIP_MODEL_NAME = CLIP_MODEL_NAME
    
    OBJECT_DETECTION_THRESHOLD = OBJECT_DETECTION_THRESHOLD
    TEXT_DETECTION_THRESHOLD = TEXT_DETECTION_THRESHOLD
    CLASSIFICATION_THRESHOLD = CLASSIFICATION_THRESHOLD
    
    MAX_IMAGE_SIZE = MAX_IMAGE_SIZE
    IMAGE_FORMATS = IMAGE_FORMATS
    
    OCR_LANGUAGES = OCR_LANGUAGES
    OCR_GPU = OCR_GPU
    
    SAVE_DEBUG_IMAGES = SAVE_DEBUG_IMAGES
    DEBUG_IMAGES_DIR = DEBUG_IMAGES_DIR
    # SCENE_JSON_DIR removed
    
    ENABLE_SCENE_GRAPH = ENABLE_SCENE_GRAPH
    BATCH_SIZE = BATCH_SIZE
    
    LOG_LEVEL = LOG_LEVEL
    LOG_FILE = LOG_FILE

settings = Settings()
