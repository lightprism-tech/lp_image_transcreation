"""
Load configuration from YAML with environment variable overrides.
Supports .env via python-dotenv.
"""

import os
from pathlib import Path
from typing import Any, List, Tuple

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

CONFIG_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = CONFIG_DIR.parent
BASE_DIR = PACKAGE_DIR.parent.parent

_DEFAULT_CONFIG_PATH = CONFIG_DIR / "settings.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return str(val).lower() in ("true", "1", "yes")


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    return float(val)


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    return int(val)


def load_settings(config_path: Path | None = None) -> "Settings":
    config_path = config_path or _DEFAULT_CONFIG_PATH
    data = _load_yaml(config_path)

    env_cfg = data.get("environment", {})
    paths_cfg = data.get("paths", {})
    schemas_cfg = data.get("schemas", {})
    models_cfg = data.get("models", {})
    thresholds_cfg = data.get("thresholds", {})
    image_cfg = data.get("image", {})
    ocr_cfg = data.get("ocr", {})
    output_cfg = data.get("output", {})
    pipeline_cfg = data.get("pipeline", {})
    logging_cfg = data.get("logging", {})

    ENV = os.getenv("PERCEPTION_ENV", env_cfg.get("env", "development"))
    DEBUG = _env_bool("DEBUG", env_cfg.get("debug", False))

    MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / paths_cfg.get("models_dir", "models"))))
    DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / paths_cfg.get("data_dir", "data"))))
    CACHE_DIR = Path(os.getenv("CACHE_DIR", str(BASE_DIR / paths_cfg.get("cache_dir", "cache"))))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / paths_cfg.get("output_dir", "data/output"))))

    OBJECT_SCHEMA_PATH = PACKAGE_DIR / schemas_cfg.get("object_schema", "schemas/object_schema.json")
    SCENE_SCHEMA_PATH = PACKAGE_DIR / schemas_cfg.get("scene_schema", "schemas/scene_schema.json")

    yolo_rel = models_cfg.get("yolo", "yolo/yolov8x.pt")
    if os.getenv("YOLO_MODEL_PATH"):
        YOLO_MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH"))
    else:
        YOLO_MODEL_PATH = MODELS_DIR / yolo_rel

    BLIP2_MODEL_NAME = os.getenv("BLIP_MODEL", models_cfg.get("blip", "Salesforce/blip-image-captioning-base"))
    CLIP_MODEL_NAME = os.getenv("CLIP_MODEL", models_cfg.get("clip", "openai/clip-vit-large-patch14"))
    sam_cfg = models_cfg.get("sam", {})
    SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", sam_cfg.get("model_type", "vit_b"))
    sam_checkpoint_rel = sam_cfg.get("checkpoint", "sam/sam_vit_b_01ec64.pth")
    if os.getenv("SAM_CHECKPOINT_PATH"):
        SAM_CHECKPOINT_PATH = Path(os.getenv("SAM_CHECKPOINT_PATH"))
    else:
        SAM_CHECKPOINT_PATH = MODELS_DIR / sam_checkpoint_rel

    OBJECT_DETECTION_THRESHOLD = _env_float("OBJECT_THRESHOLD", thresholds_cfg.get("object_detection", 0.5))
    TEXT_DETECTION_THRESHOLD = _env_float("TEXT_THRESHOLD", thresholds_cfg.get("text_detection", 0.6))
    CLASSIFICATION_THRESHOLD = _env_float("CLASSIFICATION_THRESHOLD", thresholds_cfg.get("classification", 0.7))

    max_size = image_cfg.get("max_size", [1920, 1080])
    MAX_IMAGE_SIZE: Tuple[int, int] = (int(max_size[0]), int(max_size[1]))
    IMAGE_FORMATS: List[str] = image_cfg.get("formats", [".jpg", ".jpeg", ".png", ".bmp", ".webp"])

    OCR_LANGUAGES: List[str] = ocr_cfg.get("languages", ["en"])
    OCR_GPU = _env_bool("OCR_GPU", ocr_cfg.get("gpu", False))

    SAVE_DEBUG_IMAGES = _env_bool("SAVE_DEBUG", output_cfg.get("save_debug_images", True))
    DEBUG_IMAGES_DIR = OUTPUT_DIR / "debug"

    ENABLE_SCENE_GRAPH = pipeline_cfg.get("enable_scene_graph", False)
    ENABLE_SAM_SEGMENTATION = _env_bool(
        "ENABLE_SAM_SEGMENTATION",
        pipeline_cfg.get("enable_sam_segmentation", True),
    )
    ENABLE_FACE_DETECTION = _env_bool(
        "ENABLE_FACE_DETECTION",
        pipeline_cfg.get("enable_face_detection", True),
    )
    ENABLE_TYPOGRAPHY_SUMMARY = _env_bool(
        "ENABLE_TYPOGRAPHY_SUMMARY",
        pipeline_cfg.get("enable_typography_summary", True),
    )
    ENABLE_MODEL_WARMUP = _env_bool(
        "ENABLE_MODEL_WARMUP",
        pipeline_cfg.get("enable_model_warmup", True),
    )
    BATCH_SIZE = _env_int("BATCH_SIZE", pipeline_cfg.get("batch_size", 1))
    FALLBACK_ACTIONABLE_CLASSES: List[str] = pipeline_cfg.get(
        "fallback_actionable_classes",
        [],
    )

    LOG_LEVEL = os.getenv("LOG_LEVEL", logging_cfg.get("level", "INFO"))
    LOG_FILE = BASE_DIR / logging_cfg.get("file", "pipeline.log")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    return Settings(
        BASE_DIR=BASE_DIR,
        PACKAGE_DIR=PACKAGE_DIR,
        ENV=ENV,
        DEBUG=DEBUG,
        MODELS_DIR=MODELS_DIR,
        DATA_DIR=DATA_DIR,
        CACHE_DIR=CACHE_DIR,
        OUTPUT_DIR=OUTPUT_DIR,
        OBJECT_SCHEMA_PATH=OBJECT_SCHEMA_PATH,
        SCENE_SCHEMA_PATH=SCENE_SCHEMA_PATH,
        YOLO_MODEL_PATH=YOLO_MODEL_PATH,
        BLIP2_MODEL_NAME=BLIP2_MODEL_NAME,
        CLIP_MODEL_NAME=CLIP_MODEL_NAME,
        SAM_MODEL_TYPE=SAM_MODEL_TYPE,
        SAM_CHECKPOINT_PATH=SAM_CHECKPOINT_PATH,
        OBJECT_DETECTION_THRESHOLD=OBJECT_DETECTION_THRESHOLD,
        TEXT_DETECTION_THRESHOLD=TEXT_DETECTION_THRESHOLD,
        CLASSIFICATION_THRESHOLD=CLASSIFICATION_THRESHOLD,
        MAX_IMAGE_SIZE=MAX_IMAGE_SIZE,
        IMAGE_FORMATS=IMAGE_FORMATS,
        OCR_LANGUAGES=OCR_LANGUAGES,
        OCR_GPU=OCR_GPU,
        SAVE_DEBUG_IMAGES=SAVE_DEBUG_IMAGES,
        DEBUG_IMAGES_DIR=DEBUG_IMAGES_DIR,
        ENABLE_SCENE_GRAPH=ENABLE_SCENE_GRAPH,
        ENABLE_SAM_SEGMENTATION=ENABLE_SAM_SEGMENTATION,
        ENABLE_FACE_DETECTION=ENABLE_FACE_DETECTION,
        ENABLE_TYPOGRAPHY_SUMMARY=ENABLE_TYPOGRAPHY_SUMMARY,
        ENABLE_MODEL_WARMUP=ENABLE_MODEL_WARMUP,
        BATCH_SIZE=BATCH_SIZE,
        FALLBACK_ACTIONABLE_CLASSES=FALLBACK_ACTIONABLE_CLASSES,
        LOG_LEVEL=LOG_LEVEL,
        LOG_FILE=LOG_FILE,
    )


class Settings:
    """Immutable settings container loaded from YAML and env."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"Settings are read-only; cannot set {name}")


settings = load_settings()
