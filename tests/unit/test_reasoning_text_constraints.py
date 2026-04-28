from src.reasoning.engine import (
    _build_text_edits_for_document,
    _estimate_max_chars_for_region,
    _validate_rewrite_constraints,
    _is_placeholder_text,
    _rewrite_culture_title_text,
    _pick_best_rewrite_candidate,
)


def test_estimate_max_chars_respects_bbox_and_font():
    max_chars = _estimate_max_chars_for_region(
        "Original Label",
        bbox=[0, 0, 120, 24],
        style={"font_size": 12},
    )
    assert max_chars >= 8
    assert max_chars <= 200


def test_validate_rewrite_constraints_trims_overflow():
    original = "Original Text"
    candidate = "This is a very long rewrite candidate that should be trimmed by layout rules"
    out = _validate_rewrite_constraints(
        original,
        candidate,
        bbox=[0, 0, 60, 20],
        style={"font_size": 12},
    )
    assert len(out) <= len(candidate)
    assert len(out) >= 2


def test_pick_best_candidate_prefers_length_compatible():
    original = "Eco Growth"
    best = _pick_best_rewrite_candidate(
        original,
        ["A", "Sustainable Growth", "This candidate is too long and should be heavily penalized for layout"],
        bbox=[0, 0, 120, 22],
        style={"font_size": 12},
    )
    assert isinstance(best, str)
    assert best != "A"


def test_placeholder_text_is_not_rewritten():
    original = "Duis aute irure dolor in reprehenderit"

    assert _is_placeholder_text(original)
    assert _is_placeholder_text("voluptate velit esse cillum dolore eu")
    assert _is_placeholder_text("XXX")
    assert _validate_rewrite_constraints(
        original,
        "Error in judgment",
        bbox=[0, 0, 200, 30],
        style={"font_size": 12},
    ) == original


def test_repeated_text_rewrite_is_consistent():
    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def generate_reasoning(self, prompt):
            self.calls += 1
            return {"candidates": [f"Localized title {self.calls}"]}

    llm = FakeLLM()
    scene_graph = {
        "image_type": {"type": "infographic"},
        "scene": {"description": "Travel infographic"},
        "text": {
            "full_text": "GLOBAL GUIDE GLOBAL GUIDE",
            "extracted": [
                {"text": "GLOBAL GUIDE", "bbox": [0, 0, 200, 40], "style": {"font_size": 12}},
                {"text": "GLOBAL GUIDE", "bbox": [0, 50, 200, 90], "style": {"font_size": 12}},
            ],
        },
    }

    edits = _build_text_edits_for_document(scene_graph, "India", llm_client=llm)

    assert len(edits) == 2
    assert edits[0]["translated"] == edits[1]["translated"]
    assert llm.calls == 1


def test_country_title_is_rewritten_from_scene_context():
    assert _rewrite_culture_title_text("JAPAN", "India", "japan infographic elements") == "INDIA"
    assert _rewrite_culture_title_text("INFOGRAPHICS ELEMENTS", "India", "japan infographic elements") is None
