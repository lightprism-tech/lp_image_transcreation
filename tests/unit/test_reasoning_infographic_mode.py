from unittest.mock import MagicMock, patch

from src.reasoning.engine import CulturalReasoningEngine
from src.reasoning.schemas import ReasoningInput


def _engine(tmp_path):
    (tmp_path / "dummy.json").write_text("{}")
    with patch("src.reasoning.engine.KnowledgeLoader", return_value=MagicMock()), patch(
        "src.reasoning.engine.LLMClient", return_value=MagicMock()
    ):
        engine = CulturalReasoningEngine(str(tmp_path / "dummy.json"))
        engine.kg_loader.get_cultural_types.return_value = {"FOOD", "CLOTHING", "ART", "SYMBOL", "TEXT"}
        engine.kg_loader.get_avoid_list.return_value = []
        engine.kg_loader.get_label_to_type.return_value = {}
        engine.kg_loader.get_candidates_from_kb.return_value = []
        engine.kg_loader.get_nodes_by_type_and_culture.return_value = []
        engine.kg_loader.get_style_priors.return_value = None
        engine.kg_loader.get_sensitivity_notes.return_value = []
        engine.kg_loader.get_preferred_substitution.return_value = None
        engine.llm_client.generate_candidates.return_value = []
        engine.llm_client.generate_reasoning.return_value = {"action": "preserve", "rationale": "test"}
        return engine


def test_infographic_mode_preserves_coco_style_objects(tmp_path):
    engine = _engine(tmp_path)
    inp = ReasoningInput(
        scene_graph={
            "image_type": {"type": "infographic"},
            "scene": {"description": "A stylized infographic with icons and labels."},
            "objects": [{"label": "bicycle", "confidence": 0.9}],
            "text": {"extracted": []},
        },
        target_culture="India",
    )
    plan = engine.analyze_image(inp)
    assert len(plan.transformations) == 0
    assert len(plan.preservations) == 1
    assert "Infographic mode" in plan.preservations[0].rationale


def test_infographic_mode_preserves_ambiguous_person_detection(tmp_path):
    engine = _engine(tmp_path)
    inp = ReasoningInput(
        scene_graph={
            "image_type": {"type": "poster"},
            "infographic_analysis": {"enabled": True, "icon_cluster_count": 3},
            "scene": {"description": "Stylized chart with tiny avatars."},
            "objects": [{"label": "person", "confidence": 0.41}],
            "text": {"extracted": []},
        },
        target_culture="Japan",
    )
    plan = engine.analyze_image(inp)
    assert len(plan.transformations) == 0
    assert len(plan.preservations) == 1
    assert "ambiguous person detection preserved" in plan.preservations[0].rationale


def test_infographic_mode_allows_cultural_icon_transform(tmp_path):
    engine = _engine(tmp_path)
    engine.kg_loader.find_node.return_value = None
    engine.kg_loader.get_cultural_types.return_value = {"FOOD", "CLOTHING", "ART", "SYMBOL", "TEXT", "SPORT"}
    engine.kg_loader.get_label_to_type.return_value = {"bicycle": "SPORT"}
    engine.kg_loader.get_candidates_from_kb.return_value = ["cricket"]
    engine.kg_loader.get_preferred_substitution.return_value = "cricket"
    engine.llm_client.generate_reasoning.return_value = {
        "action": "transform",
        "target_object": "cricket",
        "confidence": 0.9,
        "rationale": "Use a culturally relevant sport icon.",
    }
    inp = ReasoningInput(
        scene_graph={
            "image_type": {"type": "infographic"},
            "infographic_analysis": {"enabled": True, "icon_cluster_count": 1},
            "scene": {"description": "An icon-based sports infographic."},
            "objects": [
                {
                    "label": "bicycle",
                    "confidence": 0.8,
                    "semantic_type": "icon",
                    "icon_cluster_id": 0,
                }
            ],
            "text": {"extracted": []},
        },
        target_culture="India",
    )
    plan = engine.analyze_image(inp)
    assert len(plan.transformations) == 1
    assert plan.transformations[0].target_object.lower() == "cricket"
