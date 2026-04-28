import logging
from pathlib import Path
from unittest.mock import MagicMock

from src.main import (
    _edit_plan_has_actions,
    _generate_with_strict_quality,
    _log_stage2_actionability,
    _normalize_stage2_objects,
    _score_below_threshold,
)


def test_normalize_stage2_objects_populates_required_fields():
    scene_graph = {
        "objects": [
            {
                "id": 1,
                "label": "banana",
            }
        ]
    }

    _normalize_stage2_objects(scene_graph)

    assert isinstance(scene_graph["objects"], list)
    assert scene_graph["objects"][0]["class_name"] == "banana"
    assert scene_graph["objects"][0]["original_class_name"] == "banana"
    assert scene_graph["objects"][0]["bbox"] == []


def test_normalize_stage2_objects_uses_visual_regions_when_objects_missing():
    scene_graph = {
        "visual_regions": [
            {
                "id": 3,
                "type": "icon",
                "description": "red folded paper bird icon",
                "bbox": [1, 2, 3, 4],
                "confidence": 0.82,
                "quality_flags": [],
            }
        ]
    }

    _normalize_stage2_objects(scene_graph)

    assert scene_graph["objects"][0]["class_name"] == "icon"
    assert scene_graph["objects"][0]["caption"] == "red folded paper bird icon"
    assert scene_graph["objects"][0]["bbox"] == [1, 2, 3, 4]


def test_log_stage2_actionability_warns_when_no_actions(caplog):
    scene_graph = {
        "image_type": {"type": "infographic"},
        "objects": [],
    }
    edit_plan = {"transformations": [], "edit_text": []}

    with caplog.at_level(logging.WARNING):
        _log_stage2_actionability(scene_graph, edit_plan)

    assert "Stage-2 produced no actionable edits" in caplog.text
    assert "image_type=infographic" in caplog.text
    assert "detected_objects=0" in caplog.text


def test_log_stage2_actionability_no_warning_when_transform_exists(caplog):
    scene_graph = {"image_type": {"type": "poster"}, "objects": []}
    edit_plan = {
        "transformations": [{"original_object": "x", "target_object": "y"}],
        "edit_text": [],
    }

    with caplog.at_level(logging.WARNING):
        _log_stage2_actionability(scene_graph, edit_plan)

    assert "Stage-2 produced no actionable edits" not in caplog.text


def test_score_below_threshold_reports_low_scores():
    failures = _score_below_threshold(
        {"cultural_score": 0.59, "object_presence_score": 0.61},
        {"min_cultural_score": 0.7, "min_object_presence_score": 0.7},
    )

    assert len(failures) == 2
    assert "cultural_score" in failures[0]
    assert "object_presence_score" in failures[1]


def test_edit_plan_has_actions_ignores_preserve_only_plan():
    plan = MagicMock()
    plan.replace = []
    plan.edit_text = []
    plan.adjust_style = None
    plan.preserve = ["unknown_visual_region"]

    assert not _edit_plan_has_actions(plan)


def test_generate_with_strict_quality_retries_then_selects_best_effort(tmp_path):
    output1 = tmp_path / "generated1.png"
    output2 = tmp_path / "generated2.png"
    output1.write_bytes(b"fake1")
    output2.write_bytes(b"fake2")
    engine = MagicMock()
    engine.generate.side_effect = [str(output1), str(output2)]
    engine.passes_composite_validation.return_value = True
    engine.get_run_metrics.side_effect = [
        {"cultural_score": 0.4, "object_presence_score": 0.4},
        {"cultural_score": 0.4, "object_presence_score": 0.4},
        {"cultural_score": 0.6, "object_presence_score": 0.6},
        {"cultural_score": 0.6, "object_presence_score": 0.6},
    ]

    selected = _generate_with_strict_quality(
        realization_engine=engine,
        edit_plan=MagicMock(),
        image_path=Path("input.png"),
        target_culture="India",
        target_objects=["India cultural environment"],
        validation_cfg={
            "strict_quality_gate": True,
            "max_quality_attempts": 2,
            "return_best_attempt_on_quality_failure": True,
            "min_cultural_score": 0.7,
            "min_object_presence_score": 0.7,
        },
    )

    assert selected == str(output2)
    assert engine.generate.call_count == 2
    assert engine._run_metrics["selected_best_effort"] is True
