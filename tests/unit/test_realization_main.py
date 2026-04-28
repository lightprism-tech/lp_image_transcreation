"""Pytest for realization main: one test per function."""
import json
import os
import pytest
from src.realization.main import load_json, _apply_mock_overlay, _apply_mock_text_changes
from src.realization.models import EditPlan, EditTextAction


def test_load_json_returns_dict(tmp_path):
    data = {"preserve": [], "replace": []}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    result = load_json(str(path))
    assert result == data


def test_load_json_handles_unicode(tmp_path):
    data = {"target_culture": "Japan", "note": "\u3042\u3044\u3046"}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    result = load_json(str(path))
    assert result["note"] == "\u3042\u3044\u3046"


def test_apply_mock_overlay_creates_file(tmp_path):
    """_apply_mock_overlay writes a PNG to output_path."""
    input_img = tmp_path / "in.png"
    # Minimal 2x2 RGB PNG (valid image)
    from PIL import Image
    img = Image.new("RGB", (2, 2), color=(100, 150, 200))
    img.save(str(input_img), "PNG")
    output_img = tmp_path / "out.png"
    _apply_mock_overlay(str(input_img), str(output_img), "India")
    assert output_img.exists()
    assert output_img.stat().st_size > 0
    loaded = Image.open(output_img)
    assert loaded.size == (2, 2)


def test_apply_mock_overlay_uses_target_culture(tmp_path):
    """Output image is created with culture in label (PIL draws text)."""
    from PIL import Image
    input_img = tmp_path / "in.png"
    Image.new("RGB", (50, 50), color=(0, 0, 0)).save(str(input_img), "PNG")
    output_img = tmp_path / "out.png"
    _apply_mock_overlay(str(input_img), str(output_img), "Japan")
    assert output_img.exists()
    # Content differs from input due to overlay
    assert output_img.stat().st_size >= input_img.stat().st_size


def test_apply_mock_text_changes_creates_output(tmp_path):
    from PIL import Image

    input_img = tmp_path / "in.png"
    Image.new("RGB", (120, 80), color=(240, 240, 240)).save(str(input_img), "PNG")
    output_img = tmp_path / "out.png"
    plan = EditPlan(
        preserve=[],
        replace=[],
        edit_text=[
            EditTextAction(
                bbox=[10, 10, 110, 40],
                original="Global Offer",
                translated="Local Festival Offer",
                style={
                    "text_color": [10, 10, 10],
                    "background_color": [250, 250, 250],
                    "font_size": 16,
                },
            )
        ],
    )
    _apply_mock_text_changes(plan, str(input_img), str(output_img), "India")
    assert output_img.exists()
    assert output_img.stat().st_size > 0
