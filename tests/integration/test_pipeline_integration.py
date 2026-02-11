"""
Integration test for the full Stage-1 Perception Pipeline.
Ensures all components work together to produce valid output.
"""

import pytest
import os
from pathlib import Path
from perception.core.pipeline import main as pipeline_main

@pytest.mark.integration
def test_pipeline_end_to_end(test_image_path, temp_output_dir):
    """
    Test the full pipeline from image loading to JSON output.
    Uses a real image if available, otherwise skips.
    """
    if not test_image_path or not os.path.exists(test_image_path):
        pytest.skip("No test image found in fixtures or data/input/samples")

    print(f"\nRunning integration test with image: {test_image_path}")

    # Define output path
    output_path = str(temp_output_dir / "integration_test_output.json")

    # Run pipeline
    # Note: We're not mocking anything here to ensure true integration testing
    # This requires models to be loaded, which might be slow
    try:
        result = pipeline_main(test_image_path, output_path=output_path)
    except Exception as e:
        pytest.fail(f"Pipeline execution failed: {str(e)}")

    # Verify Output Structure
    assert isinstance(result, dict), "Pipeline should return a dictionary"
    
    # Check essential top-level keys
    expected_keys = [
        "image_path", 
        "image_type", 
        "objects", 
        "text_regions", 
        "scene_description", 
        "text_content"
    ]
    for key in expected_keys:
        assert key in result, f"Missing key in output: {key}"

    # Verify Image Path (should be absolute or relative as processed)
    assert str(result["image_path"]) == str(test_image_path)

    # Verify Image Type
    assert isinstance(result["image_type"], str), "image_type should be a string"
    assert len(result["image_type"]) > 0, "image_type should not be empty"

    # Verify Objects List
    assert isinstance(result["objects"], list), "objects should be a list"
    
    # If objects were detected, verify their structure
    if len(result["objects"]) > 0:
        obj = result["objects"][0]
        assert "label" in obj, "Object should have a label"
        assert "box_2d" in obj, "Object should have a bounding box"
        assert "attributes" in obj, "Object should have attributes"
        assert "captions" in obj, "Object should have captions"

    # Verify Scene Description
    assert isinstance(result["scene_description"], str), "scene_description should be a string"
    # Note: Description might be empty if model fails or image is blank, but usually it produces something
    
    # Verify File Creation
    assert os.path.exists(output_path), f"Output file should be created at {output_path}"
    
    print(f"\n✅ Integration verification successful. Output generated at {output_path}")
