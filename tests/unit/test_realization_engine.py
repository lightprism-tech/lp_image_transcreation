"""Pytest for realization engine: one test per method."""
import logging
from src.realization.engine import RealizationEngine
from src.realization.models import (
    EditPlan,
    ReplaceAction,
    EditTextAction,
    AdjustStyleAction,
)


def test_realization_engine_init_default_config():
    engine = RealizationEngine()
    assert engine.config == {}


def test_realization_engine_init_with_config():
    engine = RealizationEngine(config={"key": "value"})
    assert engine.config == {"key": "value"}


def test_realization_engine_init_with_none_config():
    engine = RealizationEngine(config=None)
    assert engine.config == {}


def test_realization_engine_generate_returns_mock_path():
    engine = RealizationEngine()
    plan = EditPlan(preserve=[], replace=[], edit_text=[])
    path = engine.generate(plan, "input.jpg")
    assert path == ""


def test_realization_engine_generate_calls_adjust_style_when_present(caplog):
    caplog.set_level(logging.INFO)
    engine = RealizationEngine()
    plan = EditPlan(
        preserve=[],
        replace=[],
        edit_text=[],
        adjust_style=AdjustStyleAction(palette="pastel", motifs=["m"], texture="t"),
    )
    engine.generate(plan, "input.jpg")
    logs = caplog.text
    assert "Adjusting style" in logs
    assert "pastel" in logs
    assert "m" in logs
    assert "t" in logs


def test_realization_engine_generate_calls_replace_for_each_action(caplog):
    caplog.set_level(logging.INFO)
    engine = RealizationEngine()
    plan = EditPlan(
        preserve=[],
        replace=[
            ReplaceAction(object_id=1, original="cat", new="dog"),
            ReplaceAction(object_id=2, original="burger", new="onigiri"),
        ],
        edit_text=[],
    )
    engine.generate(plan, "input.jpg")
    logs = caplog.text
    assert "Replacing object 1" in logs
    assert "cat" in logs
    assert "dog" in logs
    assert "Replacing object 2" in logs
    assert "burger" in logs
    assert "onigiri" in logs


def test_realization_engine_generate_calls_edit_text_for_each_action(caplog):
    caplog.set_level(logging.INFO)
    engine = RealizationEngine()
    plan = EditPlan(
        preserve=[],
        replace=[],
        edit_text=[
            EditTextAction(bbox=[0, 0, 10, 10], original="Hi", translated="Hello"),
        ],
    )
    engine.generate(plan, "input.jpg")
    logs = caplog.text
    assert "Editing text" in logs
    assert "Hi" in logs
    assert "Hello" in logs


def test_realization_engine_generate_calls_check_preservation(caplog):
    caplog.set_level(logging.INFO)
    engine = RealizationEngine()
    plan = EditPlan(preserve=["layout", "pose"], replace=[], edit_text=[])
    engine.generate(plan, "input.jpg")
    logs = caplog.text
    assert "Ensuring preservation" in logs
    assert "layout" in logs
    assert "pose" in logs


def test_realization_engine_generate_logs_starting_message(caplog):
    caplog.set_level(logging.INFO)
    engine = RealizationEngine()
    plan = EditPlan(preserve=[], replace=[], edit_text=[])
    engine.generate(plan, "some_image.png")
    assert "Starting realization for image: some_image.png" in caplog.text


def test_realization_engine_generate_logs_complete(caplog):
    caplog.set_level(logging.INFO)
    engine = RealizationEngine()
    plan = EditPlan(preserve=[], replace=[], edit_text=[])
    engine.generate(plan, "x.jpg")
    assert "Realization complete" in caplog.text
