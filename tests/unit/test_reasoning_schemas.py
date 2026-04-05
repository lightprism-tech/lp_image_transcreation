"""Pytest for reasoning schemas: one test per model."""
import pytest
from pydantic import ValidationError
from src.reasoning.schemas import (
    CulturalNode,
    Transformation,
    Preservation,
    TranscreationPlan,
    ReasoningInput,
)


def test_cultural_node_required_fields():
    n = CulturalNode(id="n1", label="Sushi", type="FOOD")
    assert n.id == "n1"
    assert n.label == "Sushi"
    assert n.type == "FOOD"
    assert n.culture is None


def test_cultural_node_optional_culture():
    n = CulturalNode(id="n1", label="Sushi", type="FOOD", culture="Japan")
    assert n.culture == "Japan"


def test_transformation_required_fields():
    t = Transformation(
        original_object="burger",
        original_type="FOOD",
        target_object="onigiri",
        rationale="Fit for Japan",
        confidence=0.9,
    )
    assert t.original_object == "burger"
    assert t.target_object == "onigiri"
    assert t.confidence == 0.9


def test_preservation_required_fields():
    p = Preservation(original_object="tree", rationale="Universal")
    assert p.original_object == "tree"
    assert p.rationale == "Universal"


def test_transcreation_plan_required_fields():
    plan = TranscreationPlan(
        target_culture="India",
        transformations=[],
        preservations=[],
        avoidance_adherence=[],
    )
    assert plan.target_culture == "India"
    assert plan.transformations == []
    assert plan.preservations == []
    assert plan.avoidance_adherence == []


def test_transcreation_plan_with_items():
    plan = TranscreationPlan(
        target_culture="Japan",
        transformations=[
            Transformation(
                original_object="a",
                original_type="FOOD",
                target_object="b",
                rationale="r",
                confidence=0.5,
            )
        ],
        preservations=[Preservation(original_object="x", rationale="keep")],
        avoidance_adherence=["avoid X"],
    )
    assert len(plan.transformations) == 1
    assert plan.transformations[0].target_object == "b"
    assert len(plan.preservations) == 1
    assert plan.preservations[0].original_object == "x"
    assert plan.avoidance_adherence == ["avoid X"]


def test_reasoning_input_required_fields():
    r = ReasoningInput(scene_graph={"objects": []}, target_culture="India")
    assert r.scene_graph == {"objects": []}
    assert r.target_culture == "India"
    assert r.avoid_list == []


def test_reasoning_input_optional_avoid_list():
    r = ReasoningInput(
        scene_graph={},
        target_culture="Japan",
        avoid_list=["item1", "item2"],
    )
    assert r.avoid_list == ["item1", "item2"]


def test_reasoning_input_model_dump():
    r = ReasoningInput(scene_graph={"k": "v"}, target_culture="X")
    d = r.model_dump()
    assert d["target_culture"] == "X"
    assert d["scene_graph"] == {"k": "v"}
