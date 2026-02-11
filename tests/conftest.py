"""Pytest configuration and shared fixtures."""
import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def test_image_path():
    """Provide path to test image."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "images"
    # Return first available test image or None
    for img in fixtures_dir.glob("*.jpg"):
        return str(img)
    for img in fixtures_dir.glob("*.png"):
        return str(img)
        
    # Fallback to sample images in data/input/samples
    project_root = Path(__file__).parent.parent
    sample_dir = project_root / "data" / "input" / "samples"
    if sample_dir.exists():
        for img in sample_dir.glob("*.jpg"):
            return str(img)
            
    return None

@pytest.fixture
def sample_config():
    """Provide test configuration."""
    from perception.config.settings import Settings
    settings = Settings()
    settings.DEBUG = True
    settings.SAVE_DEBUG_IMAGES = False
    return settings

@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide temporary output directory for tests."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir
