import logging

from src.main import _log_stage2_actionability, _normalize_stage2_objects


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
