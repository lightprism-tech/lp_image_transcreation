from src.reasoning.engine import (
    _estimate_max_chars_for_region,
    _validate_rewrite_constraints,
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
