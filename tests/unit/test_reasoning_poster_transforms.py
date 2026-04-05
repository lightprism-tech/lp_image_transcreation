from unittest.mock import MagicMock, patch

from src.reasoning.engine import CulturalReasoningEngine
from src.reasoning.schemas import ReasoningInput


def _engine(tmp_path):
    (tmp_path / "dummy.json").write_text("{}")
    with patch("src.reasoning.engine.KnowledgeLoader", return_value=MagicMock()), patch(
        "src.reasoning.engine.LLMClient", return_value=MagicMock()
    ):
        engine = CulturalReasoningEngine(str(tmp_path / "dummy.json"))
        engine.kg_loader.get_cultural_types.return_value = {"FOOD", "CLOTHING", "ART", "SYMBOL", "TEXT", "SPORT"}
        engine.kg_loader.find_node.return_value = None
        engine.kg_loader.get_avoid_list.return_value = []
        engine.kg_loader.get_label_to_type.return_value = {"bicycle": "SPORT"}
        engine.kg_loader.get_candidates_from_kb.return_value = ["cricket"]
        engine.kg_loader.get_nodes_by_type_and_culture.return_value = []
        engine.kg_loader.get_style_priors.return_value = None
        engine.kg_loader.get_sensitivity_notes.return_value = []
        engine.kg_loader.get_preferred_substitution.return_value = "cricket"
        engine.llm_client.generate_candidates.return_value = []
        engine.llm_client.generate_reasoning.return_value = {
            "action": "transform",
            "target_object": "cricket",
            "confidence": 0.9,
            "rationale": "Use a culturally relevant sport.",
        }
        return engine


def test_poster_allows_object_transform_when_candidates_exist(tmp_path):
    engine = _engine(tmp_path)
    inp = ReasoningInput(
        scene_graph={
            "image_type": {"type": "poster"},
            "infographic_analysis": {"enabled": True, "icon_cluster_count": 1},
            "scene": {"description": "A person riding a bicycle."},
            "objects": [{"label": "bicycle", "confidence": 0.9}],
            "text": {"extracted": []},
        },
        target_culture="Nigeria",
    )
    plan = engine.analyze_image(inp)
    assert len(plan.transformations) == 1
    assert plan.transformations[0].original_object.lower() == "bicycle"
    assert plan.transformations[0].target_object.lower() == "cricket"
