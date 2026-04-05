"""Pytest for reasoning engine: one test per method."""
import pytest
from unittest.mock import MagicMock, patch
from src.reasoning.engine import CulturalReasoningEngine
from src.reasoning.schemas import (
    ReasoningInput,
    TranscreationPlan,
    Transformation,
    Preservation,
)


@pytest.fixture
def mock_loader():
    return MagicMock()


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def engine_with_mocks(mock_loader, mock_llm, tmp_path):
    (tmp_path / "dummy.json").write_text("{}")
    with patch("src.reasoning.engine.KnowledgeLoader", return_value=mock_loader), \
         patch("src.reasoning.engine.LLMClient", return_value=mock_llm):
        engine = CulturalReasoningEngine(str(tmp_path / "dummy.json"))
        engine.kg_loader = mock_loader
        engine.llm_client = mock_llm
        mock_loader.get_cultural_types.return_value = set()
        mock_loader.get_avoid_list.return_value = []
        mock_loader.get_label_to_type.return_value = {}
        mock_loader.get_candidates_from_kb.return_value = []
        mock_loader.get_preferred_substitution.return_value = None
        mock_loader.get_style_priors.return_value = None
        mock_loader.get_sensitivity_notes.return_value = []
        mock_llm.generate_candidates.return_value = []
        return engine


def test_cultural_reasoning_engine_init_creates_loader_and_client(tmp_path):
    (tmp_path / "kg.json").write_text('{"nodes": [], "edges": []}')
    with patch("src.reasoning.engine.LLMClient"):
        engine = CulturalReasoningEngine(str(tmp_path / "kg.json"))
        assert engine.kg_loader is not None
        assert engine.llm_client is not None


def test_analyze_image_returns_transcreation_plan(engine_with_mocks, mock_loader, mock_llm):
    mock_loader.find_node.return_value = None
    mock_loader.get_nodes_by_type_and_culture.return_value = []
    mock_llm.generate_reasoning.return_value = {"action": "preserve", "rationale": "OK"}
    inp = ReasoningInput(scene_graph={"objects": [{"label": "Tree"}]}, target_culture="India")
    plan = engine_with_mocks.analyze_image(inp)
    assert isinstance(plan, TranscreationPlan)
    assert plan.target_culture == "India"


def test_analyze_image_transformation_action(engine_with_mocks, mock_loader, mock_llm):
    mock_node = MagicMock()
    mock_node.type = "FOOD"
    mock_node.id = "F_1"
    mock_loader.find_node.return_value = mock_node
    mock_loader.get_culture_of_node.return_value = "USA"
    mock_loader.get_candidates_from_kb.return_value = ["Sushi"]
    mock_loader.get_nodes_by_type_and_culture.return_value = []
    mock_llm.generate_reasoning.return_value = {
        "action": "transform",
        "target_object": "Sushi",
        "rationale": "Fit",
        "confidence": 0.9,
    }
    inp = ReasoningInput(
        scene_graph={"objects": [{"label": "Burger", "id": 1}]},
        target_culture="Japan",
    )
    plan = engine_with_mocks.analyze_image(inp)
    assert len(plan.transformations) == 1
    assert plan.transformations[0].original_object == "Burger"
    assert plan.transformations[0].target_object == "Sushi"
    assert plan.transformations[0].confidence == 0.9
    assert len(plan.preservations) == 0


def test_analyze_image_preservation_action(engine_with_mocks, mock_loader, mock_llm):
    mock_loader.find_node.return_value = None
    mock_loader.get_nodes_by_type_and_culture.return_value = []
    mock_llm.generate_reasoning.return_value = {"action": "preserve", "rationale": "Universal"}
    inp = ReasoningInput(
        scene_graph={"objects": [{"label": "Tree", "id": 1}]},
        target_culture="India",
    )
    plan = engine_with_mocks.analyze_image(inp)
    assert len(plan.preservations) == 1
    assert plan.preservations[0].original_object == "Tree"
    assert len(plan.transformations) == 0


def test_analyze_image_skips_objects_without_label(engine_with_mocks, mock_llm):
    inp = ReasoningInput(
        scene_graph={"objects": [{"id": 1}, {"label": "Car", "id": 2}]},
        target_culture="Japan",
    )
    mock_llm.generate_reasoning.return_value = {"action": "preserve", "rationale": "OK"}
    plan = engine_with_mocks.analyze_image(inp)
    assert mock_llm.generate_reasoning.call_count == 0
    assert plan.preservations[0].original_object == "Car"


def test_analyze_image_uses_class_name_fallback(engine_with_mocks, mock_loader, mock_llm):
    mock_loader.find_node.return_value = None
    mock_loader.get_nodes_by_type_and_culture.return_value = []
    mock_llm.generate_reasoning.return_value = {"action": "preserve", "rationale": "OK"}
    inp = ReasoningInput(
        scene_graph={"objects": [{"class_name": "Bicycle", "id": 1}]},
        target_culture="India",
    )
    plan = engine_with_mocks.analyze_image(inp)
    assert len(plan.preservations) == 1
    assert plan.preservations[0].original_object == "Bicycle"


def test_construct_prompt_contains_object_and_culture(engine_with_mocks):
    prompt = engine_with_mocks._construct_prompt(
        obj_label="Burger",
        obj_type="FOOD",
        source_culture="USA",
        target_culture="Japan",
        candidates=["Sushi", "Ramen"],
        context="A meal scene",
        avoid_list=["pork"],
    )
    assert "Burger" in prompt
    assert "FOOD" in prompt
    assert "USA" in prompt
    assert "Japan" in prompt
    assert "Sushi" in prompt or "Ramen" in prompt
    assert "pork" in prompt
    assert "transform" in prompt.lower()
    assert "preserve" in prompt.lower()


def test_analyze_image_uses_llm_candidates_when_kb_missing(engine_with_mocks, mock_loader, mock_llm):
    mock_loader.find_node.return_value = None
    mock_loader.get_candidates_from_kb.return_value = []
    mock_loader.get_nodes_by_type_and_culture.return_value = []
    mock_llm.generate_candidates.return_value = ["Samosa", "Biryani"]
    mock_llm.generate_reasoning.return_value = {"action": "preserve", "rationale": "KB-first"}
    inp = ReasoningInput(scene_graph={"objects": [{"label": "Burger"}]}, target_culture="India")
    plan = engine_with_mocks.analyze_image(inp)
    assert len(plan.transformations) == 0
    assert len(plan.preservations) == 1
    assert "No grounded KB candidates" in plan.preservations[0].rationale
    assert mock_llm.generate_candidates.call_count == 0


def test_analyze_image_prefers_kb_candidates_before_llm(engine_with_mocks, mock_loader, mock_llm):
    mock_loader.find_node.return_value = None
    mock_loader.get_candidates_from_kb.return_value = ["Samosa", "Biryani"]
    mock_loader.get_nodes_by_type_and_culture.return_value = []
    mock_llm.generate_reasoning.return_value = {
        "action": "transform",
        "target_object": "Samosa",
        "rationale": "KB grounded",
        "confidence": 0.9,
    }
    inp = ReasoningInput(scene_graph={"objects": [{"label": "Burger"}]}, target_culture="India")
    plan = engine_with_mocks.analyze_image(inp)
    assert len(plan.transformations) == 1
    assert plan.transformations[0].target_object == "Samosa"
    assert mock_llm.generate_candidates.call_count == 0
