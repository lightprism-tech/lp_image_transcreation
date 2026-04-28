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

    def test_build_adds_semantic_regions_and_layout(self):
        from perception.builders.scene_json_builder import SceneJSONBuilder

        builder = SceneJSONBuilder()
        scene_json = builder.build(
            image_path="data/input/samples/test.jpg",
            image_type={"type": "infographic", "confidence": 0.9},
            bounding_boxes=[
                {
                    "bbox": [10, 20, 50, 80],
                    "class_name": "bench",
                    "confidence": 0.85,
                    "semantic_type": "icon",
                    "semantic_score": 0.9,
                }
            ],
            text_boxes=[],
            object_captions=[
                {
                    "caption": "red folded paper bird icon",
                    "caption_candidates": [
                        {"prompt": "", "caption": "paper bird"},
                        {"prompt": "Describe this visual region precisely.", "caption": "red folded paper bird icon"},
                    ],
                    "source": "blip",
                }
            ],
            object_attributes=[{"attributes": {}}],
            scene_description={"description": "travel infographic", "confidence": 1.0},
            extracted_text=[
                {"text": "JAPAN", "bbox": [5, 1, 100, 31], "confidence": 0.95},
                {"text": "Travel guide details for visitors", "bbox": [5, 50, 140, 62], "confidence": 0.9},
            ],
            infographic_analysis={"enabled": True},
            image_shape=(100, 200, 3),
        )

        assert scene_json["objects"][0]["class_name"] == "red_folded_paper"
        assert scene_json["objects"][0]["original_class_name"] == "bench"
        assert len(scene_json["objects"][0]["caption_candidates"]) == 2
        assert scene_json["objects"][0]["caption_source"] == "blip"
        assert scene_json["visual_regions"][0]["type"] == "icon"
        assert scene_json["visual_regions"][0]["description"] == "red folded paper bird icon"
        assert scene_json["text_regions"][0]["role"] == "title"
        assert "title" in scene_json["layout"]["structure"]
        assert scene_json["quality_summary"]["visual_region_count"] == 1

    def test_uncertain_visual_regions_use_unknown_label(self):
        from perception.builders.scene_json_builder import SceneJSONBuilder

        builder = SceneJSONBuilder()
        scene_json = builder.build(
            image_path="data/input/samples/test.jpg",
            image_type={"type": "poster", "confidence": 0.8},
            bounding_boxes=[{"bbox": [1, 2, 10, 20], "class_name": "umbrella", "confidence": 0.1}],
            text_boxes=[],
            object_captions=[{"caption": ""}],
            object_attributes=[{"attributes": {}}],
            scene_description={"description": "test", "confidence": 1.0},
            extracted_text=[],
            infographic_analysis={"enabled": False},
        )

        obj = scene_json["objects"][0]
        assert obj["class_name"] == "unknown_visual_region"
        assert "uncertain_label" in obj["quality_flags"]
        assert scene_json["visual_regions"][0]["type"] == "unknown_visual_region"
        assert scene_json["quality_summary"]["uncertain_visual_region_count"] == 1

    def test_caption_label_survives_detector_mismatch(self):
        from perception.builders.scene_json_builder import SceneJSONBuilder

        builder = SceneJSONBuilder()
        scene_json = builder.build(
            image_path="data/input/samples/burger.jpg",
            image_type={"type": "product", "confidence": 0.64},
            bounding_boxes=[{"bbox": [1, 2, 10, 20], "class_name": "cake", "confidence": 0.31}],
            text_boxes=[],
            object_captions=[{"caption": "a close up of a hamburger with lettuce and tomato on it"}],
            object_attributes=[{"attributes": {}}],
            scene_description={"description": "hamburger product image", "confidence": 1.0},
            extracted_text=[],
            infographic_analysis={"enabled": False},
        )

        obj = scene_json["objects"][0]
        assert obj["class_name"] == "hamburger"
        assert "detector_caption_mismatch" in obj["quality_flags"]
        assert "uncertain_label" not in obj["quality_flags"]


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

