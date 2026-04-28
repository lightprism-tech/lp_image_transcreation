from src.realization.engine import RealizationEngine


def test_font_candidates_prioritize_matching_family_and_weight():
    engine = RealizationEngine()
    candidates = engine._font_candidates_for_family("Calibri", "bold")
    assert candidates
    assert candidates[0].lower().startswith("calibri")
    assert any("calibrib.ttf" == c.lower() for c in candidates)


def test_font_candidates_include_defaults_when_family_missing():
    engine = RealizationEngine()
    candidates = engine._font_candidates_for_family(None, "normal")
    assert "arial.ttf" in [c.lower() for c in candidates]
    assert "calibri.ttf" in [c.lower() for c in candidates]
