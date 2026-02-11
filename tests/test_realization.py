import pytest
import json
from src.realization.models import EditPlan, ReplaceAction, EditTextAction, AdjustStyleAction
from src.realization.schema import validate_edit_plan, get_edit_plan_schema
from src.realization.engine import RealizationEngine

def test_edit_plan_schema_generation():
    schema = get_edit_plan_schema()
    assert "properties" in schema
    assert "replace" in schema["properties"]
    assert "edit_text" in schema["properties"]

def test_valid_edit_plan_parsing():
    data = {
        "preserve": ["layout", "pose"],
        "replace": [
            {
                "object_id": 1,
                "original": "burger",
                "new": "onigiri",
                "constraints": {"size": "medium"}
            }
        ],
        "edit_text": [
            {
                "bbox": [10, 10, 100, 50],
                "original": "Welcome",
                "translated": "Yokoso"
            }
        ],
        "adjust_style": {
            "palette": "pastel",
            "motifs": ["cherry_blossom"]
        }
    }
    plan = validate_edit_plan(data)
    assert isinstance(plan, EditPlan)
    assert plan.preserve == ["layout", "pose"]
    assert len(plan.replace) == 1
    assert plan.replace[0].new == "onigiri"
    assert plan.adjust_style.palette == "pastel"

def test_realization_engine_flow(capsys):
    data = {
        "preserve": ["lighting"],
        "replace": [
            {"object_id": 2, "original": "cat", "new": "dog"}
        ],
        "edit_text": [],
        "adjust_style": {"texture": "watercolor"}
    }
    plan = validate_edit_plan(data)
    engine = RealizationEngine()
    
    output_path = engine.generate(plan, "dummy_input.jpg")
    
    assert output_path == "output/generated_image_mock.png"
    
    captured = capsys.readouterr()
    assert "Starting realization for image: dummy_input.jpg" in captured.out
    assert "Adjusting style" in captured.out
    assert "Replacing object 2" in captured.out
    assert "Ensuring preservation of: ['lighting']" in captured.out
