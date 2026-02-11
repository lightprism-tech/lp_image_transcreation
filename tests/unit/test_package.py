"""
Unit tests for package-level imports
"""
import pytest


def test_package_import():
    """Test that perception package can be imported."""
    import perception
    assert perception is not None


def test_package_version():
    """Test that package has version."""
    import perception
    assert hasattr(perception, '__version__')
    assert perception.__version__ == "0.1.0"


def test_package_author():
    """Test that package has author."""
    import perception
    assert hasattr(perception, '__author__')


def test_settings_accessible_from_package():
    """Test that settings can be accessed from package root."""
    from perception import settings
    assert settings is not None


def test_utils_modules_importable():
    """Test that utility modules can be imported (no ML dependencies)."""
    # Utils
    from perception.utils import image_loader
    from perception.utils import logger
    from perception.utils import drawing_utils
    from perception.utils import bbox_utils
    
    # Config
    from perception.config import settings
    
    # All imports successful
    assert True


def test_builders_modules_importable():
    """Test that builder modules can be imported (no ML dependencies)."""
    # Builders
    from perception.builders import scene_json_builder
    from perception.builders import scene_graph_builder
    
    # All imports successful
    assert True
