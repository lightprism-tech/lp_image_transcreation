"""Pytest for reasoning engine: one test per method."""
import pytest
from unittest.mock import MagicMock, patch
from src.reasoning.engine import (
    CulturalReasoningEngine,
    _normalize_reasoning_result,
    _extract_food_terms_from_text,
    _build_scene_override_region,
    _recover_grounded_label,
    _prioritize_unused_candidates,
    _infer_cultural_type_from_kb_signals,
    _infer_type_from_label_cues,
    _resolve_obj_type_for_localized_edit,
    _dedupe_transformations_by_original,
    _ground_llm_target_to_kb,
    _reasoning_strategy,
)
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
        mock_loader.get_kb_entry.return_value = None
        mock_loader.get_all_labels.return_value = []
        mock_loader.rank_candidates_by_embedding.side_effect = lambda _q, cands: list(cands)
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
        reasoning_strategy="kg_first",
    )
    assert "Burger" in prompt
    assert "FOOD" in prompt
    assert "USA" in prompt
    assert "Japan" in prompt
    assert "Sushi" in prompt or "Ramen" in prompt
    assert "pork" in prompt
    assert "transform" in prompt.lower()
    assert "preserve" in prompt.lower()
    assert "KG-first planning rubric" in prompt
    assert "Never invent a new target_object" in prompt


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
    assert "preserved" in plan.preservations[0].rationale.lower()
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


def test_normalize_reasoning_result_grounds_non_candidate_target():
    result = _normalize_reasoning_result(
        reasoning_result={
            "action": "transform",
            "target_object": "Traditional Japanese Sushi platter",
            "rationale": "culture fit",
            "confidence": 0.7,
        },
        candidates=["Sushi", "Ramen"],
        original_label="Burger",
    )
    assert result["action"] == "transform"
    assert result["target_object"] == "Sushi"


def test_normalize_reasoning_result_preserve_sets_original_target():
    result = _normalize_reasoning_result(
        reasoning_result={"action": "preserve", "target_object": "Sushi", "confidence": "oops"},
        candidates=["Sushi"],
        original_label="Burger",
    )
    assert result["action"] == "preserve"
    assert result["target_object"] == "Burger"
    assert result["confidence"] == 0.0


def test_extract_food_terms_from_text_uses_dynamic_mapping():
    terms = _extract_food_terms_from_text(
        "Weekly count of pizzas and burgers sold",
        label_to_type={"pizza": "FOOD", "burger": "FOOD", "bicycle": "SPORT"},
        kb_entry=None,
    )
    assert "pizza" in terms
    assert "burger" in terms
    assert "bicycle" not in terms


def test_build_region_replacements_uses_kg_candidates_and_llm_choice(engine_with_mocks, mock_loader, mock_llm):
    mock_loader.get_label_to_type.return_value = {"pizza": "FOOD"}
    mock_loader.get_kb_entry.return_value = None
    mock_loader.get_preferred_substitution.return_value = None
    mock_loader.get_candidates_from_kb.return_value = ["samosa", "kachori"]
    # LLM chooses best grounded target candidate.
    mock_llm.generate_reasoning.return_value = {"target_food": "kachori"}
    inp = ReasoningInput(
        scene_graph={
            "image_type": {"type": "infographic"},
            "objects": [],
            "scene": {"description": "worksheet"},
            "text": {
                "full_text": "Number of pizzas sold Monday Tuesday",
                "extracted": [
                    {"text": "Monday", "bbox": [10, 10, 80, 40]},
                    {"text": "Tuesday", "bbox": [10, 50, 80, 80]},
                    {"text": "Number of pizzas sold", "bbox": [100, 5, 300, 35]},
                ],
            },
        },
        target_culture="India",
    )
    region = engine_with_mocks.build_region_replacements(inp)
    assert len(region) == 2
    assert all(r["new"] == "kachori icon" for r in region)


def test_analyze_image_strict_mode_skips_full_scene_override_for_photo_like_input(mock_loader, mock_llm, tmp_path):
    (tmp_path / "dummy.json").write_text("{}")
    with patch("src.reasoning.engine.KnowledgeLoader", return_value=mock_loader), \
         patch("src.reasoning.engine.LLMClient", return_value=mock_llm):
        engine = CulturalReasoningEngine(str(tmp_path / "dummy.json"), strict_mode=True)
        engine.kg_loader = mock_loader
        engine.llm_client = mock_llm
        mock_loader.get_cultural_types.return_value = set()
        mock_loader.get_avoid_list.return_value = []
        mock_loader.get_label_to_type.return_value = {}
        mock_loader.get_candidates_from_kb.return_value = []
        mock_loader.get_preferred_substitution.return_value = None
        mock_loader.get_style_priors.return_value = None
        mock_loader.get_sensitivity_notes.return_value = []
        mock_loader.get_kb_entry.return_value = None
        mock_llm.generate_candidates.return_value = []
        mock_llm.generate_reasoning.return_value = {"action": "preserve", "rationale": "No grounded candidates"}

        inp = ReasoningInput(
            scene_graph={
                "image_type": {"type": "other"},
                "scene": {"description": "a young tiger cub in the wild"},
                "objects": [],
                "text": {"extracted": [], "full_text": ""},
            },
            target_culture="Japan",
        )
        plan = engine.analyze_image(inp)

    assert plan.region_replace == []
    assert (plan.scene_adaptation or {}).get("override_mode") == "none"
    assert (plan.scene_adaptation or {}).get("override_reason") == "disabled_for_photo_like_image_type"


def test_strict_infographic_scene_override_adds_scene_transformation(mock_loader, mock_llm, tmp_path):
    (tmp_path / "dummy.json").write_text("{}")
    with patch("src.reasoning.engine.KnowledgeLoader", return_value=mock_loader), \
         patch("src.reasoning.engine.LLMClient", return_value=mock_llm):
        engine = CulturalReasoningEngine(str(tmp_path / "dummy.json"), strict_mode=True)
        engine.kg_loader = mock_loader
        engine.llm_client = mock_llm
        mock_loader.get_scene_candidates.return_value = [
            {"name": "Fort", "elements": ["Fort details"], "lighting": "natural", "style": "vibrant"}
        ]
        mock_loader.get_cultural_types.return_value = set()
        mock_loader.get_avoid_list.return_value = []
        mock_loader.get_label_to_type.return_value = {}
        mock_loader.get_candidates_from_kb.return_value = []
        mock_loader.get_preferred_substitution.return_value = None
        mock_loader.get_style_priors.return_value = None
        mock_loader.get_sensitivity_notes.return_value = []
        mock_loader.get_kb_entry.return_value = None
        mock_llm.generate_candidates.return_value = []
        mock_llm.generate_reasoning.return_value = {"action": "preserve", "rationale": "No grounded candidates"}

        inp = ReasoningInput(
            scene_graph={
                "image_type": {"type": "infographic"},
                "scene": {"description": "Japan infographic elements"},
                "objects": [{"label": "bench", "confidence": 0.8}],
                "text": {"extracted": [], "full_text": ""},
            },
            target_culture="India",
        )
        plan = engine.analyze_image(inp)

    assert plan.transformations
    assert plan.transformations[0].original_type == "SCENE"
    assert plan.transformations[0].target_object == "India cultural environment"
    assert plan.region_replace


def test_scene_override_region_infers_canvas_from_text_and_object_boxes():
    scene_graph = {
        "objects": [{"bbox": [704.1, 950.0, 1714.2, 1643.8]}],
        "text": {
            "regions": [{"bbox": [2706.0, 5196.0, 4421.0, 5271.0]}],
        },
    }

    region = _build_scene_override_region(
        scene_graph,
        "India",
        {"scene": "Fort", "elements": ["Fort details", "India local context"]},
    )

    assert region["bbox"] == [0, 0, 4421, 5271]
    assert region["new"] == "India Fort cultural infographic with Fort details, India local context"


def test_prioritize_unused_candidates_moves_used_targets_to_end():
    candidates = ["Taj Mahal", "Biryani", "Red Fort"]
    ranked = _prioritize_unused_candidates(candidates, {"Taj Mahal"})
    assert ranked[0] in {"Biryani", "Red Fort"}
    assert ranked[-1] == "Taj Mahal"


def test_normalize_reasoning_result_prefers_unused_candidate():
    result = _normalize_reasoning_result(
        reasoning_result={
            "action": "transform",
            "target_object": "Taj Mahal",
            "confidence": 0.9,
            "rationale": "iconic",
        },
        candidates=["Taj Mahal", "Biryani", "Samosa"],
        original_label="plate_sushi",
        used_targets={"Taj Mahal"},
    )
    assert result["action"] == "transform"
    assert result["target_object"] == "Biryani"


def test_reasoning_strategy_defaults_to_llm_first():
    assert _reasoning_strategy() == "llm_first"


def test_ground_llm_target_to_kb_maps_fuzzy_llm_name():
    loader = MagicMock()
    loader.find_node.return_value = None
    loader.rank_candidates_by_embedding.return_value = ["Taj Mahal", "Red Fort"]
    grounded = _ground_llm_target_to_kb(
        "taj mahal monument",
        loader,
        ["Taj Mahal", "Red Fort", "Chapati"],
    )
    assert grounded == "Taj Mahal"


def test_infer_type_from_label_cues_maps_building_to_landmark():
    obj = {
        "label": "illustration_japanese_building",
        "caption": "illustration of a japanese building with a tree",
        "semantic_type": "symbol",
    }
    assert _infer_type_from_label_cues(obj, "illustration_japanese_building") == "LANDMARK"


def test_infer_cultural_type_from_kb_signals_ignores_stopword_food_matches():
    obj = {
        "label": "illustration_japanese_building",
        "caption": "illustration of a japanese building with a tree and a tree in front of it",
        "semantic_type": "symbol",
    }
    type_index = {
        "FOOD": {"a", "and", "in", "on", "burger"},
        "LANDMARK": {"building", "taj", "mahal"},
    }
    inferred = _infer_cultural_type_from_kb_signals(obj, type_index)
    assert inferred == "LANDMARK"


def test_infer_cultural_type_from_kb_signals_uses_dynamic_type_tokens():
    obj = {
        "label": "plate_sushi",
        "caption": "there is a plate of sushi with a bowl of sauce on it",
        "semantic_type": "icon",
    }
    type_index = {
        "FOOD": {"sushi", "biryani", "samosa", "plate"},
        "LANDMARK": {"taj", "mahal", "fort"},
    }
    inferred = _infer_cultural_type_from_kb_signals(obj, type_index)
    assert inferred == "FOOD"


def test_recover_grounded_label_rejects_embedding_without_token_overlap(monkeypatch):
    monkeypatch.setattr(
        "src.reasoning.engine.get_policy_int",
        lambda key, default=0: 2 if key == "grounding_min_embedding_token_overlap" else 1,
    )
    class _Node:
        def __init__(self, label, node_type):
            self.label = label
            self.type = node_type
            self.id = label

    loader = MagicMock()
    loader.get_all_labels.return_value = ["Chapati", "Ninja", "Samurai"]
    loader.find_node.side_effect = lambda label: {
        "chapati": _Node("Chapati", "FOOD"),
        "ninja": _Node("Ninja", "SYMBOL"),
        "samurai": _Node("Samurai", "SYMBOL"),
    }.get(label.lower())
    loader.rank_candidates_by_embedding.return_value = ["Chapati", "Ninja"]

    obj = {
        "label": "red_fan",
        "caption": "a decorative red folding fan on white background",
        "semantic_type": "icon",
        "bbox": [10, 10, 100, 100],
    }
    grounded = _recover_grounded_label(
        obj,
        loader,
        exclude_scope_types=True,
        allowed_types={"FOOD"},
    )
    assert grounded is None


def test_recover_grounded_label_excludes_country_nodes_for_localized_icons():
    class _Node:
        def __init__(self, label, node_type):
            self.label = label
            self.type = node_type
            self.id = label

    loader = MagicMock()
    loader.get_all_labels.return_value = ["Japan", "Sushi", "Taj Mahal"]
    loader.find_node.side_effect = lambda label: {
        "japan": _Node("Japan", "COUNTRY"),
        "sushi": _Node("Sushi", "FOOD"),
        "taj mahal": _Node("Taj Mahal", "LANDMARK"),
    }.get(label.lower())
    loader.rank_candidates_by_embedding.return_value = ["Sushi"]

    obj = {
        "label": "plate_sushi",
        "caption": "plate of sushi on a table",
        "bbox": [10, 10, 100, 100],
    }
    grounded = _recover_grounded_label(obj, loader, exclude_scope_types=True)
    assert grounded == "Sushi"


def test_resolve_obj_type_avoids_country_node_type_for_bbox_icons():
    class _Node:
        def __init__(self, label, node_type):
            self.label = label
            self.type = node_type
            self.id = label

    loader = MagicMock()
    loader.find_node.return_value = _Node("Japan", "COUNTRY")
    loader.get_culture_of_node.return_value = "Japan"
    loader.get_label_to_type.return_value = {}

    obj = {
        "label": "Japan",
        "caption": "plate of sushi with sauce",
        "semantic_type": "icon",
        "bbox": [1, 2, 3, 4],
    }
    obj_type, _ = _resolve_obj_type_for_localized_edit(
        obj=obj,
        source_label="Japan",
        kg_loader=loader,
        type_token_index={"FOOD": {"sushi", "sauce", "plate"}, "LANDMARK": {"fort"}},
    )
    assert obj_type == "FOOD"


def test_dedupe_transformations_by_original():
    items = [
        Transformation(
            original_object="red_origami_bird",
            original_type="SYMBOL",
            target_object="Peacock",
            rationale="a",
            confidence=0.8,
        ),
        Transformation(
            original_object="red_origami_bird",
            original_type="SYMBOL",
            target_object="Taj Mahal",
            rationale="b",
            confidence=0.8,
        ),
    ]
    deduped = _dedupe_transformations_by_original(items)
    assert len(deduped) == 1
    assert deduped[0].target_object == "Peacock"
