"""Pytest for realization schema: one test per function."""
import pytest
from src.realization.schema import (
    get_edit_plan_schema,
    adapt_plan_to_edit_format,
    validate_edit_plan,
)
from src.realization.models import EditPlan


def test_get_edit_plan_schema_returns_dict():
    schema = get_edit_plan_schema()
    assert isinstance(schema, dict)


def test_get_edit_plan_schema_has_properties():
    schema = get_edit_plan_schema()
    assert "properties" in schema


def test_get_edit_plan_schema_has_replace():
    schema = get_edit_plan_schema()
    assert "replace" in schema["properties"]


def test_get_edit_plan_schema_has_edit_text():
    schema = get_edit_plan_schema()
    assert "edit_text" in schema["properties"]


def test_get_edit_plan_schema_has_preserve():
    schema = get_edit_plan_schema()
    assert "preserve" in schema["properties"]


def test_adapt_plan_to_edit_format_passthrough_when_already_edit_plan():
    data = {"preserve": ["a"], "replace": []}
    out = adapt_plan_to_edit_format(data)
    assert out is data
    assert out["preserve"] == ["a"]


def test_adapt_plan_to_edit_format_converts_preservations():
    data = {"preservations": [{"original_object": "person"}, {"original_object": "bicycle"}]}
    out = adapt_plan_to_edit_format(data)
    assert out["preserve"] == ["person", "bicycle"]


def test_adapt_plan_to_edit_format_converts_transformations():
    data = {
        "transformations": [
            {"original_object": "burger", "target_object": "onigiri"},
            {"original_object": "cat", "target_object": "dog"},
        ]
    }
    out = adapt_plan_to_edit_format(data)
    assert len(out["replace"]) == 2
    assert out["replace"][0]["object_id"] == 0
    assert out["replace"][0]["original"] == "burger"
    assert out["replace"][0]["new"] == "onigiri"
    assert out["replace"][1]["object_id"] == 1
    assert out["replace"][1]["original"] == "cat"
    assert out["replace"][1]["new"] == "dog"


def test_adapt_plan_to_edit_format_skips_non_dict_items():
    data = {"preservations": [{"original_object": "a"}, "invalid", None]}
    out = adapt_plan_to_edit_format(data)
    assert out["preserve"] == ["a"]


def test_adapt_plan_to_edit_format_defaults_edit_text_and_adjust_style():
    data = {"preservations": [], "transformations": []}
    out = adapt_plan_to_edit_format(data)
    assert out["edit_text"] == []
    assert out["adjust_style"] is None


def test_validate_edit_plan_returns_edit_plan():
    data = {"preserve": [], "replace": [], "edit_text": []}
    plan = validate_edit_plan(data)
    assert isinstance(plan, EditPlan)


def test_validate_edit_plan_parses_full_plan():
    data = {
        "preserve": ["layout"],
        "replace": [{"object_id": 1, "original": "x", "new": "y"}],
        "edit_text": [{"bbox": [0, 0, 10, 10], "original": "Hi", "translated": "Hello"}],
        "adjust_style": {"palette": "pastel", "motifs": [], "texture": None},
    }
    plan = validate_edit_plan(data)
    assert plan.preserve == ["layout"]
    assert len(plan.replace) == 1
    assert plan.replace[0].original == "x"
    assert len(plan.edit_text) == 1
    assert plan.edit_text[0].translated == "Hello"
    assert plan.adjust_style.palette == "pastel"
