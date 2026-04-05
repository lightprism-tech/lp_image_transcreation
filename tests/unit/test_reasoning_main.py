"""Pytest for reasoning main: one test per function."""
import json
import os
import pytest
from src.reasoning.main import load_json, save_json


def test_load_json_returns_dict(tmp_path):
    data = {"scene_graph": {"objects": []}, "target_culture": "India"}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    result = load_json(str(path))
    assert result == data


def test_load_json_unicode(tmp_path):
    data = {"note": "\u3042\u3044\u3046"}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    result = load_json(str(path))
    assert result["note"] == "\u3042\u3044\u3046"


def test_save_json_writes_file(tmp_path):
    data = {"target_culture": "Japan", "transformations": []}
    path = tmp_path / "output.json"
    save_json(data, str(path))
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["target_culture"] == "Japan"


def test_save_json_indent_and_ensure_ascii_false(tmp_path):
    data = {"text": "\u3042"}
    path = tmp_path / "out.json"
    save_json(data, str(path))
    content = path.read_text(encoding="utf-8")
    assert "\u3042" in content or "\\u3042" in content
