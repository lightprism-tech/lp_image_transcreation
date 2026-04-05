import os

from PIL import Image, ImageDraw

from src.realization.engine import RealizationEngine


def _make_image(path: str, color=(128, 128, 128)):
    img = Image.new("RGB", (120, 120), color=color)
    img.save(path)


def test_quality_gate_rejects_large_distribution_shift(tmp_path):
    src_path = str(tmp_path / "src.png")
    out_path = str(tmp_path / "out.png")
    _make_image(src_path, color=(120, 120, 120))
    _make_image(out_path, color=(120, 120, 120))
    out = Image.open(out_path).convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.rectangle([30, 30, 90, 90], fill=(250, 0, 0))
    out.save(out_path)

    engine = RealizationEngine(config={"quality_gate": {"enabled": True, "max_mean_distance": 20, "max_std_distance": 20}})
    assert engine._fails_local_quality_gate(src_path, out_path, [30, 30, 90, 90]) is True


def test_quality_gate_can_be_disabled(tmp_path):
    src_path = str(tmp_path / "src.png")
    out_path = str(tmp_path / "out.png")
    _make_image(src_path, color=(120, 120, 120))
    _make_image(out_path, color=(200, 50, 50))

    engine = RealizationEngine(config={"quality_gate": {"enabled": False}})
    assert engine._fails_local_quality_gate(src_path, out_path, [10, 10, 80, 80]) is False
    assert os.path.exists(src_path)
