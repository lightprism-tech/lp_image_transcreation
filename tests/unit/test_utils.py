"""
Unit tests for utility modules
"""
import pytest
import numpy as np
from pathlib import Path


class TestImageLoader:
    """Tests for image_loader module."""
    
    def test_load_image_import(self):
        """Test that load_image can be imported."""
        from perception.utils.image_loader import load_image
        assert callable(load_image)
    
    def test_load_image_with_sample(self):
        """Test loading a sample image if available."""
        from perception.utils.image_loader import load_image
        from perception.config import settings
        
        sample_dir = settings.BASE_DIR / "data" / "input" / "samples"
        if sample_dir.exists():
            images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
            if images:
                img = load_image(str(images[0]))
                assert img is not None
                assert isinstance(img, np.ndarray)
                assert len(img.shape) == 3  # Height, Width, Channels


class TestLogger:
    """Tests for logger module."""
    
    def test_setup_logger_import(self):
        """Test that setup_logger can be imported."""
        from perception.utils.logger import setup_logger
        assert callable(setup_logger)
    
    def test_setup_logger_creates_logger(self):
        """Test that setup_logger returns a logger."""
        from perception.utils.logger import setup_logger
        import logging
        
        logger = setup_logger()
        assert logger is not None
        assert isinstance(logger, logging.Logger)


class TestBBoxUtils:
    """Tests for bbox_utils module."""
    
    def test_bbox_utils_import(self):
        """Test that bbox_utils module can be imported."""
        from perception.utils import bbox_utils
        assert bbox_utils is not None


class TestDrawingUtils:
    """Tests for drawing_utils module."""
    
    def test_debug_visualizer_import(self):
        """Test that DebugVisualizer can be imported."""
        from perception.utils.drawing_utils import DebugVisualizer
        assert DebugVisualizer is not None
    
    def test_debug_visualizer_instantiation(self):
        """Test that DebugVisualizer can be instantiated."""
        from perception.utils.drawing_utils import DebugVisualizer
        visualizer = DebugVisualizer()
        assert visualizer is not None
