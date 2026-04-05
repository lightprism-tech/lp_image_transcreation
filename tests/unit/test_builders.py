"""
Unit tests for builder modules
"""
import pytest


class TestSceneJSONBuilder:
    """Tests for SceneJSONBuilder."""
    
    def test_import(self):
        """Test that SceneJSONBuilder can be imported."""
        from perception.builders.scene_json_builder import SceneJSONBuilder
        assert SceneJSONBuilder is not None
    
    def test_instantiation(self):
        """Test that SceneJSONBuilder can be instantiated."""
        from perception.builders.scene_json_builder import SceneJSONBuilder
        builder = SceneJSONBuilder()
        assert builder is not None
        assert hasattr(builder, 'build')
        assert hasattr(builder, 'save')

    def test_build_includes_object_label_and_original_class_name(self):
        from perception.builders.scene_json_builder import SceneJSONBuilder

        builder = SceneJSONBuilder()
        scene_json = builder.build(
            image_path="data/input/samples/test.jpg",
            image_type={"type": "poster", "confidence": 0.8},
            bounding_boxes=[{"bbox": [1, 2, 10, 20], "class_name": "bicycle", "confidence": 0.9}],
            text_boxes=[],
            object_captions=[{"caption": "a bicycle"}],
            object_attributes=[{"attributes": {}}],
            scene_description={"description": "test", "confidence": 1.0},
            extracted_text=[],
            infographic_analysis={"enabled": False},
        )

        assert len(scene_json["objects"]) == 1
        obj = scene_json["objects"][0]
        assert obj["class_name"] == "bicycle"
        assert obj["label"] == "bicycle"
        assert obj["original_class_name"] == "bicycle"

    def test_build_includes_new_text_and_quality_fields(self):
        from perception.builders.scene_json_builder import SceneJSONBuilder

        builder = SceneJSONBuilder()
        scene_json = builder.build(
            image_path="data/input/samples/test.jpg",
            image_type={"type": "poster", "confidence": 0.8},
            bounding_boxes=[{"bbox": [1, 2, 10, 20], "class_name": "bicycle", "confidence": 0.9}],
            text_boxes=[{"bbox": [1, 1, 5, 5], "confidence": 0.7}],
            object_captions=[{"caption": "a bicycle"}],
            object_attributes=[{"attributes": {}}],
            scene_description={"description": "test", "confidence": 1.0},
            extracted_text=[{"text": "Sale", "bbox": [1, 1, 5, 5], "confidence": 0.8}],
            faces=[{"bbox": [2, 2, 4, 4], "confidence": 1.0}],
            typography={"avg_font_size": 14.0, "styled_regions": 1},
            object_text_links=[{"text_index": 0, "object_index": 0, "overlap_iou": 0.2}],
            quality_summary={"object_count": 1, "sam_available": False},
            infographic_analysis={"enabled": False},
        )

        assert "faces" in scene_json
        assert isinstance(scene_json["faces"], list)
        assert scene_json["faces"][0]["confidence"] == 1.0
        assert scene_json["text"]["typography"]["avg_font_size"] == 14.0
        assert scene_json["text"]["object_links"][0]["object_index"] == 0
        assert scene_json["quality_summary"]["object_count"] == 1
        assert "segmentation" in scene_json["objects"][0]


class TestSceneGraphBuilder:
    """Tests for SceneGraphBuilder."""
    
    def test_import(self):
        """Test that SceneGraphBuilder can be imported."""
        from perception.builders.scene_graph_builder import SceneGraphBuilder
        assert SceneGraphBuilder is not None
    
    def test_instantiation(self):
        """Test that SceneGraphBuilder can be instantiated."""
        from perception.builders.scene_graph_builder import SceneGraphBuilder
        builder = SceneGraphBuilder()
        assert builder is not None


class TestBuildersModule:
    """Tests for builders module as a whole."""
    
    def test_module_import(self):
        """Test that builders module can be imported."""
        from perception import builders
        assert builders is not None
    
    def test_all_exports(self):
        """Test that all builder classes are exported."""
        from perception.builders import SceneJSONBuilder, SceneGraphBuilder
        assert SceneJSONBuilder is not None
        assert SceneGraphBuilder is not None

