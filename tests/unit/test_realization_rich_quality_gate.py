import numpy as np
import os
from PIL import Image

from src.realization.engine import RealizationEngine


def test_ssim_gate_path_is_enforced_when_enabled():
    engine = RealizationEngine(config={"quality_gate": {"enabled": True, "use_ssim": True, "use_clip_local": False, "max_mean_distance": 999, "max_std_distance": 999}})
    engine._fails_ssim_gate = lambda *args, **kwargs: True
    src = np.full((100, 100, 3), 120, dtype=np.uint8)
    out = src.copy()
    src_path = "tmp_src_ssim.png"
    out_path = "tmp_out_ssim.png"
    Image.fromarray(src).save(src_path)
    Image.fromarray(out).save(out_path)
    try:
        assert engine._fails_local_quality_gate(src_path, out_path, [30, 30, 70, 70]) is True
    finally:
        if os.path.exists(src_path):
            os.remove(src_path)
        if os.path.exists(out_path):
            os.remove(out_path)


def test_clip_gate_can_be_mocked_for_threshold():
    engine = RealizationEngine(config={"quality_gate": {"enabled": True, "use_ssim": False, "use_clip_local": True}})
    engine._fails_clip_local_gate = lambda *args, **kwargs: True
    src = np.full((80, 80, 3), 100, dtype=np.uint8)
    out = src.copy()
    src_path = "tmp_src_qg.png"
    out_path = "tmp_out_qg.png"
    Image.fromarray(src).save(src_path)
    Image.fromarray(out).save(out_path)
    try:
        assert engine._fails_local_quality_gate(src_path, out_path, [20, 20, 50, 50]) is True
    finally:
        if os.path.exists(src_path):
            os.remove(src_path)
        if os.path.exists(out_path):
            os.remove(out_path)
