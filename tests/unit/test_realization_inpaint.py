import base64
import io

import numpy as np
from PIL import Image

from src.realization.inpaint import (
    _bbox_to_alpha_edit_mask_bytes,
    _normalize_gpt_image_size,
    get_inpainter,
)


def test_bbox_to_alpha_edit_mask_bytes_transparent_inside_bbox():
    mask_bytes = _bbox_to_alpha_edit_mask_bytes(
        bbox=[2, 2, 8, 8],
        width=10,
        height=10,
        pad_pct=0.0,
    )
    assert mask_bytes
    mask = Image.open(io.BytesIO(mask_bytes)).convert("RGBA")
    arr = np.array(mask)
    assert arr[3, 3, 3] == 0
    assert arr[0, 0, 3] == 255


def test_get_inpainter_uses_gpt_image_edits_with_plan_bbox(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://unit-test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("AZURE_OPENAI_IMAGE_DEPLOYMENT", "gpt-image-2")
    monkeypatch.setenv("AZURE_OPENAI_IMAGE_API_VERSION", "2025-04-01-preview")

    out = io.BytesIO()
    Image.fromarray(np.full((64, 64, 3), 180, dtype=np.uint8)).save(out, format="PNG")
    encoded = base64.b64encode(out.getvalue()).decode("ascii")
    captured = {}

    class DummyResponse:
        status_code = 200
        headers = {"x-ms-request-id": "req-test"}
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"b64_json": encoded}]}

    def _fake_post(url, headers, data, files, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["data"] = data
        captured["timeout"] = timeout
        captured["files"] = files
        return DummyResponse()

    monkeypatch.setattr("src.realization.inpaint.requests.post", _fake_post)

    source_path = tmp_path / "source.png"
    Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8)).save(source_path)
    backend = get_inpainter(
        {
            "use_inpainting": True,
            "inpaint_model": "gpt-image-2",
            "gpt_image_quality": "medium",
            "gpt_image_output_format": "png",
            "gpt_image_size": "auto",
            "gpt_image_timeout_seconds": 90,
            "inpaint_mask_pad_pct": 0.0,
            "gpt_image_strict_bbox_lock": True,
            "gpt_image_use_edits": True,
        }
    )
    result_path = backend.inpaint(str(source_path), [10, 12, 30, 40], "replace burger with idli")
    assert result_path
    assert "/images/edits?api-version=2025-04-01-preview" in captured["url"]
    assert captured["data"]["quality"] == "medium"
    assert captured["data"]["size"] == "64x64"
    mask_tuple = captured["files"]["mask"]
    mask = Image.open(io.BytesIO(mask_tuple[1])).convert("RGBA")
    mask_arr = np.array(mask)
    assert mask_arr[20, 20, 3] == 0
    assert mask_arr[5, 5, 3] == 255


def test_normalize_gpt_image_size_caps_long_edge():
    width, height = _normalize_gpt_image_size(4496, 5488)
    assert max(width, height) <= 3840
    assert width % 16 == 0
    assert height % 16 == 0
    assert (width * height) <= 8294400
