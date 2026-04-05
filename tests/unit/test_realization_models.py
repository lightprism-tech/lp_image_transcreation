"""Pytest for realization models: one test per model/field."""
import pytest
from pydantic import ValidationError
from src.realization.models import (
    PreserveConstraint,
    ReplaceAction,
    EditTextAction,
    AdjustStyleAction,
    EditPlan,
)


def test_preserve_constraint_requires_aspect():
    c = PreserveConstraint(aspect="layout")
    assert c.aspect == "layout"


def test_preserve_constraint_raises_without_aspect():
    with pytest.raises(ValidationError):
        PreserveConstraint()


def test_replace_action_required_fields():
    a = ReplaceAction(object_id=1, original="burger", new="onigiri")
    assert a.object_id == 1
    assert a.original == "burger"
    assert a.new == "onigiri"
    assert a.constraints is None


def test_replace_action_optional_constraints():
    a = ReplaceAction(object_id=2, original="cat", new="dog", constraints={"size": "medium"})
    assert a.constraints == {"size": "medium"}


def test_edit_text_action_required_fields():
    a = EditTextAction(bbox=[10, 20, 100, 50], original="Hello", translated="Hola")
    assert a.bbox == [10, 20, 100, 50]
    assert a.original == "Hello"
    assert a.translated == "Hola"


def test_adjust_style_action_defaults():
    a = AdjustStyleAction()
    assert a.palette is None
    assert a.motifs == []
    assert a.texture is None


def test_adjust_style_action_all_fields():
    a = AdjustStyleAction(palette="pastel", motifs=["cherry"], texture="watercolor")
    assert a.palette == "pastel"
    assert a.motifs == ["cherry"]
    assert a.texture == "watercolor"


def test_edit_plan_defaults():
    plan = EditPlan()
    assert plan.preserve == []
    assert plan.replace == []
    assert plan.edit_text == []
    assert plan.adjust_style is None


def test_edit_plan_full():
    plan = EditPlan(
        preserve=["layout"],
        replace=[ReplaceAction(object_id=1, original="a", new="b")],
        edit_text=[EditTextAction(bbox=[0, 0, 1, 1], original="x", translated="y")],
        adjust_style=AdjustStyleAction(palette="warm"),
    )
    assert plan.preserve == ["layout"]
    assert len(plan.replace) == 1
    assert len(plan.edit_text) == 1
    assert plan.adjust_style.palette == "warm"


def test_edit_plan_model_dump_roundtrip():
    data = {"preserve": ["p"], "replace": [], "edit_text": []}
    plan = EditPlan(**data)
    dumped = plan.model_dump()
    assert dumped["preserve"] == ["p"]
