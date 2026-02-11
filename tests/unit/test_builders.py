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

