"""Pytest for realization engine: one test per method."""
import logging
import numpy as np
from PIL import Image
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


def test_text_quality_gate_rejects_low_color_delta():
    engine = RealizationEngine(config={"text_quality_gate": {"enabled": True, "min_color_delta": 60}})
    src = np.full((20, 20, 3), 240, dtype=np.uint8)
    out = src.copy()
    failed, metrics = engine._fails_text_quality_gate(
        source_arr=src,
        output_arr=out,
        bbox=[0, 0, 20, 20],
        fg_rgb=(230, 230, 230),
        bg_rgb=(240, 240, 240),
    )
    assert failed is True
    assert metrics["color_delta"] < 60


def test_text_quality_gate_allows_full_bbox_redraw_when_configured():
    engine = RealizationEngine.__new__(RealizationEngine)
    engine._text_quality_config = {"enabled": True, "max_bbox_occupancy": 1.0}
    src = np.full((20, 20, 3), 240, dtype=np.uint8)
    out = np.full((20, 20, 3), 20, dtype=np.uint8)
    failed, metrics = engine._fails_text_quality_gate(
        source_arr=src,
        output_arr=out,
        bbox=[0, 0, 20, 20],
        fg_rgb=(10, 10, 10),
        bg_rgb=(245, 245, 245),
    )
    assert failed is False
    assert metrics["occupancy_ratio"] == 1.0


def test_text_edit_skips_local_quality_gate_by_default(tmp_path):
    src_path = tmp_path / "src.png"
    Image.fromarray(np.full((80, 240, 3), 245, dtype=np.uint8)).save(src_path)
    engine = RealizationEngine.__new__(RealizationEngine)
    engine._text_quality_config = {"enabled": True, "max_bbox_occupancy": 1.0}
    engine._quality_gate_config = {"enabled": True}
    engine._fails_local_quality_gate = lambda *args, **kwargs: True

    out_path = engine._edit_text(
        str(src_path),
        EditTextAction(
            bbox=[10, 10, 220, 60],
            original="JAPAN",
            translated="INDIA",
            style={
                "font_size": 24,
                "text_color": [10, 10, 10],
                "background_color": [245, 245, 245],
            },
        ),
    )

    assert out_path


def test_pick_high_contrast_text_color_prefers_dark_on_bright_bg():
    engine = RealizationEngine()
    picked = engine._pick_high_contrast_text_color((245, 245, 245))
    assert picked == (10, 10, 10)


def test_local_quality_gate_text_mode_skips_ssim_clip_by_default(tmp_path):
    src_path = tmp_path / "src.png"
    out_path = tmp_path / "out.png"
    arr = np.full((40, 120, 3), 245, dtype=np.uint8)
    arr[:, 20:100, :] = 230
    Image.fromarray(arr).save(src_path)
    arr2 = arr.copy()
    arr2[10:30, 25:95, :] = 60
    Image.fromarray(arr2).save(out_path)

    engine = RealizationEngine(config={"quality_gate": {"enabled": True, "use_ssim": True, "use_clip_local": True}})
    called = {"ssim": False, "clip": False}

    def _mark_ssim(*args, **kwargs):
        called["ssim"] = True
        return False

    def _mark_clip(*args, **kwargs):
        called["clip"] = True
        return False

    engine._fails_ssim_gate = _mark_ssim
    engine._fails_clip_local_gate = _mark_clip
    engine._fails_local_quality_gate(str(src_path), str(out_path), [20, 5, 100, 35], edit_kind="text")
    assert called["ssim"] is False
    assert called["clip"] is False
