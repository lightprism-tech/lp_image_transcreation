"""
Unit tests for configuration module
"""
import pytest
from pathlib import Path


def test_settings_import():
    """Test that settings can be imported."""
    from perception.config import settings
    assert settings is not None


def test_base_dir_exists():
    """Test that BASE_DIR points to project root."""
    from perception.config import settings
    assert settings.BASE_DIR.exists()
    assert settings.BASE_DIR.is_dir()
    # Should contain src directory
    assert (settings.BASE_DIR / "src").exists()


def test_paths_configuration():
    """Test that all configured paths are Path objects."""
    from perception.config import settings
    
    assert isinstance(settings.BASE_DIR, Path)
    assert isinstance(settings.MODELS_DIR, Path)
    assert isinstance(settings.DATA_DIR, Path)
    assert isinstance(settings.CACHE_DIR, Path)
    assert isinstance(settings.OUTPUT_DIR, Path)


def test_model_names():
    """Test that model names are configured."""
    from perception.config import settings
    
    assert settings.BLIP2_MODEL_NAME
    assert settings.CLIP_MODEL_NAME
    assert isinstance(settings.BLIP2_MODEL_NAME, str)
    assert isinstance(settings.CLIP_MODEL_NAME, str)


def test_thresholds():
    """Test that detection thresholds are valid."""
    from perception.config import settings
    
    assert 0.0 <= settings.OBJECT_DETECTION_THRESHOLD <= 1.0
    assert 0.0 <= settings.TEXT_DETECTION_THRESHOLD <= 1.0
    assert 0.0 <= settings.CLASSIFICATION_THRESHOLD <= 1.0


def test_output_directories_created():
    """Test that output directories are created."""
    from perception.config import settings
    
    # Directories are created by config loader
    assert settings.OUTPUT_DIR.exists()
    assert settings.DEBUG_IMAGES_DIR.exists()


def test_settings_class():
    """Test settings object has expected attributes."""
    from perception.config import settings

    assert hasattr(settings, 'BASE_DIR')
    assert hasattr(settings, 'MODELS_DIR')
    assert hasattr(settings, 'YOLO_MODEL_PATH')
    assert hasattr(settings, 'LOG_LEVEL')


def test_sam_and_pipeline_feature_toggles_present():
    """Test SAM settings and new pipeline toggles are available."""
    from perception.config import settings

    assert hasattr(settings, "SAM_MODEL_TYPE")
    assert hasattr(settings, "SAM_CHECKPOINT_PATH")
    assert hasattr(settings, "ENABLE_SAM_SEGMENTATION")
    assert hasattr(settings, "ENABLE_FACE_DETECTION")
    assert hasattr(settings, "ENABLE_TYPOGRAPHY_SUMMARY")
    assert hasattr(settings, "ENABLE_MODEL_WARMUP")
