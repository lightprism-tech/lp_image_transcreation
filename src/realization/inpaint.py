"""
Inpainting backends for object/clothing replacement.
Mask convention: white = region to inpaint, black = keep.
"""
import logging
import os
import tempfile
import base64
import io
import time
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, List
import requests
from dotenv import load_dotenv
from src.realization.prompt_config import get_prompt, get_prompt_list

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Default size for inpainting (SD models often expect 512)
INPAINT_SIZE = 512


def _clamp_bbox(bbox: List[int], width: int, height: int, pad_pct: float = 0.0) -> Optional[List[int]]:
    """Clamp bbox to image bounds with optional proportional padding."""
    if not bbox or len(bbox) < 4:
        return None
    x_min, y_min, x_max, y_max = [int(round(x)) for x in bbox[:4]]
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return None
    if pad_pct > 0:
        pad_x = max(1, int(w * pad_pct))
        pad_y = max(1, int(h * pad_pct))
        x_min -= pad_x
        y_min -= pad_y
        x_max += pad_x
        y_max += pad_y
    x_min = max(0, min(x_min, width))
    y_min = max(0, min(y_min, height))
    x_max = max(0, min(x_max, width))
    y_max = max(0, min(y_max, height))
    if x_max <= x_min or y_max <= y_min:
        return None
    return [x_min, y_min, x_max, y_max]


def _bbox_to_mask_pil(bbox: List[int], width: int, height: int, pad_pct: float = 0.1):
    """Create a PIL Image mask: white inside bbox (with optional padding), black outside."""
    from PIL import Image
    import numpy as np
    x_min, y_min, x_max, y_max = [int(round(x)) for x in bbox[:4]]
    w = x_max - x_min
    h = y_max - y_min
    pad_x = max(1, int(w * pad_pct))
    pad_y = max(1, int(h * pad_pct))
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(width, x_max + pad_x)
    y_max = min(height, y_max + pad_y)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y_min:y_max, x_min:x_max] = 255
    return Image.fromarray(mask)


def _bbox_to_alpha_edit_mask_bytes(bbox: List[int], width: int, height: int, pad_pct: float = 0.0) -> Optional[bytes]:
    """
    Build PNG RGBA mask bytes for Azure image edits.
    Transparent area (alpha=0) is the edit region; opaque outside is preserved.
    """
    from PIL import Image
    import numpy as np

    normalized_bbox = _clamp_bbox(bbox, width, height, pad_pct=pad_pct)
    if normalized_bbox is None:
        return None
    x_min, y_min, x_max, y_max = normalized_bbox
    mask = np.zeros((height, width, 4), dtype=np.uint8)
    mask[:, :, :3] = 255
    mask[:, :, 3] = 255
    mask[y_min:y_max, x_min:x_max, 3] = 0
    image = Image.fromarray(mask)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _parse_size(size_text: str) -> Optional[tuple[int, int]]:
    parts = (size_text or "").lower().split("x")
    if len(parts) != 2:
        return None
    try:
        w = int(parts[0].strip())
        h = int(parts[1].strip())
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return (w, h)


def _normalize_gpt_image_size(
    width: int,
    height: int,
    max_edge: int = 3840,
    max_pixels: int = 8_294_400,
) -> tuple[int, int]:
    """
    Normalize size to gpt-image-2 constraints:
    - both dimensions multiples of 16
    - longest edge <= max_edge
    """
    if width <= 0 or height <= 0:
        return (1024, 1024)
    scale_edge = min(1.0, float(max_edge) / float(max(width, height)))
    pixel_count = float(width * height)
    scale_pixels = min(1.0, (float(max_pixels) / pixel_count) ** 0.5) if pixel_count > 0 else 1.0
    scale = min(scale_edge, scale_pixels)
    new_w = max(16, int(round(width * scale)))
    new_h = max(16, int(round(height * scale)))
    new_w = max(16, new_w - (new_w % 16))
    new_h = max(16, new_h - (new_h % 16))
    new_w = min(max_edge, new_w)
    new_h = min(max_edge, new_h)
    new_w = max(16, new_w - (new_w % 16))
    new_h = max(16, new_h - (new_h % 16))
    while (new_w * new_h) > max_pixels and new_w > 16 and new_h > 16:
        new_w = max(16, new_w - 16)
        new_h = max(16, new_h - 16)
    return (new_w, new_h)


def _build_inpaint_prompt(original_label: str, new_label: str, target_culture: str = "target") -> str:
    """Build a prompt for inpainting (e.g. clothing -> person wearing X)."""
    original_lower = (original_label or "").strip().lower()
    person_tokens = [
        str(t).strip().lower()
        for t in get_prompt_list(
            "inpaint.label_groups.person",
            ["person", "people", "man", "woman", "child"],
        )
    ]
    clothing_tokens = [
        str(t).strip().lower()
        for t in get_prompt_list(
            "inpaint.label_groups.clothing",
            ["cloth", "shirt", "dress"],
        )
    ]

    if original_lower in set(person_tokens):
        template = get_prompt(
            "inpaint.person_template",
            "person wearing {new_label}, same pose and lighting, photorealistic",
        )
        return template.format(new_label=new_label, target_culture=target_culture)
    if any(token and token in original_lower for token in clothing_tokens):
        template = get_prompt(
            "inpaint.clothing_template",
            "person wearing {new_label}, same pose, photorealistic",
        )
        return template.format(new_label=new_label, target_culture=target_culture)
    template = get_prompt(
        "inpaint.generic_template",
        "{new_label}, same scene and lighting, photorealistic",
    )
    return template.format(new_label=new_label, target_culture=target_culture)


class Inpainter(ABC):
    """Abstract inpainting backend."""

    @abstractmethod
    def inpaint(
        self,
        image_path: str,
        bbox: List[int],
        prompt: str,
        negative_prompt: str = "blurry, distorted, low quality",
    ) -> Optional[str]:
        """
        Inpaint the region defined by bbox in the image using the given prompt.
        Returns path to the resulting image, or None on failure.
        """
        pass


class MockInpainter(Inpainter):
    """No-op inpainter; returns None so caller falls back to mock overlay."""

    def inpaint(
        self,
        image_path: str,
        bbox: List[int],
        prompt: str,
        negative_prompt: str = "blurry, distorted, low quality",
    ) -> Optional[str]:
        logger.info("Mock inpainter: would inpaint bbox=%s with prompt=%s", bbox, prompt)
        return None


def _apply_mask_composite(base_img, generated_img, bbox, mask_pad_pct: float):
    """Composite generated content into bbox area with configurable padding."""
    import numpy as np
    w_orig, h_orig = base_img.size
    mask_full = _bbox_to_mask_pil(bbox, w_orig, h_orig, pad_pct=mask_pad_pct)
    base_arr = np.array(base_img)
    gen_arr = np.array(generated_img)
    mask_arr = np.array(mask_full) > 128
    base_arr[mask_arr] = gen_arr[mask_arr]
    return base_arr


def _decode_flux_response_image(response: requests.Response):
    """Decode image from Flux API response (binary, base64 JSON, or URL JSON)."""
    from PIL import Image
    content_type = (response.headers.get("content-type") or "").lower()

    # Direct image bytes response
    if content_type.startswith("image/"):
        return Image.open(io.BytesIO(response.content)).convert("RGB")

    data = response.json()

    # Base64 image in common keys
    b64_candidates = []
    if isinstance(data, dict):
        data_list = data.get("data")
        if isinstance(data_list, list):
            for item in data_list:
                if not isinstance(item, dict):
                    continue
                b64 = item.get("b64_json")
                if isinstance(b64, str):
                    try:
                        raw = base64.b64decode(b64)
                        return Image.open(io.BytesIO(raw)).convert("RGB")
                    except Exception:
                        pass
                url = item.get("url")
                if isinstance(url, str):
                    try:
                        r = requests.get(url, timeout=60)
                        r.raise_for_status()
                        return Image.open(io.BytesIO(r.content)).convert("RGB")
                    except Exception:
                        pass

        for key in ("image", "b64_json", "image_base64", "output"):
            val = data.get(key)
            if isinstance(val, str):
                b64_candidates.append(val)
        images = data.get("images")
        if isinstance(images, list):
            for item in images:
                if isinstance(item, str):
                    b64_candidates.append(item)
                elif isinstance(item, dict):
                    b64 = item.get("b64_json") or item.get("image")
                    if isinstance(b64, str):
                        b64_candidates.append(b64)
        result = data.get("result")
        if isinstance(result, dict):
            b64 = result.get("b64_json") or result.get("image")
            if isinstance(b64, str):
                b64_candidates.append(b64)

        # URL image in common keys
        url_candidates = []
        for key in ("url", "image_url"):
            val = data.get(key)
            if isinstance(val, str):
                url_candidates.append(val)
        if isinstance(result, dict):
            for key in ("url", "image_url"):
                val = result.get(key)
                if isinstance(val, str):
                    url_candidates.append(val)
        if isinstance(images, list):
            for item in images:
                if isinstance(item, dict):
                    for key in ("url", "image_url"):
                        val = item.get(key)
                        if isinstance(val, str):
                            url_candidates.append(val)

        for b64 in b64_candidates:
            try:
                if "," in b64 and "base64" in b64:
                    b64 = b64.split(",", 1)[1]
                raw = base64.b64decode(b64)
                return Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                continue

        for url in url_candidates:
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                return Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception:
                continue

    raise ValueError("Flux API response did not contain a decodable image")


def _get_flux_inpainter(config: dict) -> Optional[Inpainter]:
    """Build a Flux/Azure image-edit backend using source image + mask + prompt."""
    model_name = str(config.get("inpaint_model") or "").strip().lower()
    supported_models = {
        str(m).strip().lower()
        for m in (config.get("flux_supported_models") or ["flux.2-pro", "flux-2-pro", "flux2-pro"])
        if str(m).strip()
    }
    if model_name not in supported_models:
        return None

    endpoint = (
        os.getenv("AZURE_FLUX_EDIT_URL", "").strip()
        or os.getenv("AZURE_FLUX_IMAGE_URL", "").strip()
    )
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    flux_model = str(config.get("flux_model") or os.getenv("AZURE_FLUX_MODEL", "FLUX.2-pro")).strip() or "FLUX.2-pro"
    output_format = str(config.get("flux_output_format") or os.getenv("FLUX_OUTPUT_FORMAT", "png")).strip() or "png"
    flux_width = int(config.get("flux_width", os.getenv("FLUX_WIDTH", 1024)))
    flux_height = int(config.get("flux_height", os.getenv("FLUX_HEIGHT", 1024)))
    if not endpoint or not api_key:
        logger.warning(
            "FLUX.2-pro selected but AZURE_FLUX_IMAGE_URL / AZURE_OPENAI_API_KEY is missing."
        )
        return None

    from PIL import Image
    import numpy as np

    class FluxInpainter(Inpainter):
        def __init__(self):
            self.mask_pad_pct = float(config.get("inpaint_mask_pad_pct", 0.25))
            self.generation_timeout_s = int(config.get("flux_timeout_seconds", os.getenv("FLUX_TIMEOUT_SECONDS", 120)))

        def _request_flux_generated_image(self, prompt: str):
            """
            Call Azure FLUX using the same JSON request shape as scripts/test_api.py,
            which is known to work for this endpoint.
            """
            headers = {"Content-Type": "application/json", "api-key": api_key}
            payload = {
                "prompt": prompt,
                "n": 1,
                "width": flux_width,
                "height": flux_height,
                "output_format": output_format,
                "model": flux_model,
            }
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=self.generation_timeout_s)
            try:
                resp.raise_for_status()
            except requests.HTTPError as http_err:
                body = (resp.text or "").strip()
                preview = (body[:600] + "...") if len(body) > 600 else body
                logger.warning(
                    "FLUX request failed status=%s endpoint=%s payload_model=%s body=%s error=%s",
                    resp.status_code,
                    endpoint,
                    flux_model,
                    preview,
                    http_err,
                )
                raise
            return _decode_flux_response_image(resp)

        def inpaint(
            self,
            image_path: str,
            bbox: List[int],
            prompt: str,
            negative_prompt: str = "blurry, distorted, low quality",
        ) -> Optional[str]:
            try:
                img = Image.open(image_path).convert("RGB")
                w_orig, h_orig = img.size
                gen = self._request_flux_generated_image(prompt)
                gen_resized = gen.resize((w_orig, h_orig))
                result_arr = _apply_mask_composite(img, gen_resized, bbox, self.mask_pad_pct)
                result = Image.fromarray(result_arr)

                fd, path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                result.save(path)
                logger.info("FLUX image-edit result generated for bbox=%s", bbox)
                return path
            except Exception as e:
                logger.warning("FLUX image-edit generation failed: %s", e)
                return None

    return FluxInpainter()


def _get_gpt_image_inpainter(config: dict) -> Optional[Inpainter]:
    """Build an Azure GPT Image backend for bbox-locked image edits."""
    model_name = str(config.get("inpaint_model") or "").strip().lower()
    supported_models = {
        str(m).strip().lower()
        for m in (config.get("gpt_image_supported_models") or ["gpt-image-2"])
        if str(m).strip()
    }
    if model_name not in supported_models:
        return None

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    deployment = (
        os.getenv("AZURE_OPENAI_IMAGE_DEPLOYMENT", "").strip()
        or os.getenv("AZURE_OPENAI_IMAGES_DEPLOYMENT", "").strip()
        or "gpt-image-2"
    )
    api_version = os.getenv("AZURE_OPENAI_IMAGE_API_VERSION", "").strip() or "2025-04-01-preview"
    quality = str(config.get("gpt_image_quality", os.getenv("GPT_IMAGE_QUALITY", "medium"))).strip() or "medium"
    output_format = str(
        config.get("gpt_image_output_format", os.getenv("GPT_IMAGE_OUTPUT_FORMAT", "png"))
    ).strip() or "png"
    timeout_s = int(config.get("gpt_image_timeout_seconds", os.getenv("GPT_IMAGE_TIMEOUT_SECONDS", 180)))
    request_retries = max(1, int(config.get("gpt_image_request_retries", os.getenv("GPT_IMAGE_REQUEST_RETRIES", 3))))
    retry_delay_s = float(config.get("gpt_image_retry_delay_seconds", os.getenv("GPT_IMAGE_RETRY_DELAY_SECONDS", 2)))
    use_edits = bool(config.get("gpt_image_use_edits", True))
    mask_pad_pct = float(config.get("inpaint_mask_pad_pct", 0.0))
    strict_bbox_lock = bool(config.get("gpt_image_strict_bbox_lock", True))
    composite_bbox_only = bool(config.get("gpt_image_composite_bbox_only", True))
    max_edge = int(config.get("gpt_image_max_edge", os.getenv("GPT_IMAGE_MAX_EDGE", 3840)))
    max_pixels = int(config.get("gpt_image_max_pixels", os.getenv("GPT_IMAGE_MAX_PIXELS", 8294400)))

    if not endpoint or not api_key:
        logger.warning(
            "gpt-image backend selected but AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY is missing."
        )
        return None
    if not use_edits:
        logger.warning("gpt-image backend requires edits API for bbox-locked replacement.")
        return None

    from PIL import Image

    class AzureGPTImageInpainter(Inpainter):
        def _build_size(self, width: int, height: int) -> tuple[int, int]:
            configured = str(config.get("gpt_image_size", "")).strip().lower()
            if configured and configured != "auto":
                parsed = _parse_size(configured)
                if parsed:
                    return _normalize_gpt_image_size(
                        parsed[0], parsed[1], max_edge=max_edge, max_pixels=max_pixels
                    )
            return _normalize_gpt_image_size(width, height, max_edge=max_edge, max_pixels=max_pixels)

        def _build_retry_sizes(self, width: int, height: int) -> List[tuple[int, int]]:
            scales = [1.0, 0.85, 0.72, 0.6, 0.5, 0.42, 0.34]
            sizes: List[tuple[int, int]] = []
            for scale in scales:
                candidate = _normalize_gpt_image_size(
                    int(round(width * scale)),
                    int(round(height * scale)),
                    max_edge=max_edge,
                    max_pixels=max_pixels,
                )
                if candidate not in sizes:
                    sizes.append(candidate)
            return sizes

        def _request_edit_image(
            self,
            image_path: str,
            bbox: List[int],
            prompt: str,
            request_width: int,
            request_height: int,
        ):
            with Image.open(image_path).convert("RGB") as source:
                width, height = source.size
                resized = source.resize((request_width, request_height), Image.LANCZOS)
                img_buf = io.BytesIO()
                resized.save(img_buf, format="PNG")
            scaled_bbox = [
                int(round((bbox[0] / width) * request_width)),
                int(round((bbox[1] / height) * request_height)),
                int(round((bbox[2] / width) * request_width)),
                int(round((bbox[3] / height) * request_height)),
            ]
            mask_bytes = _bbox_to_alpha_edit_mask_bytes(
                bbox=scaled_bbox,
                width=request_width,
                height=request_height,
                pad_pct=mask_pad_pct,
            )
            if not mask_bytes:
                raise ValueError("Invalid bbox provided for gpt-image edit")
            edit_url = (
                f"{endpoint}/openai/deployments/{deployment}/images/edits"
                f"?api-version={api_version}"
            )
            headers = {"api-key": api_key}
            data = {
                "prompt": prompt,
                "n": "1",
                "quality": quality,
                "output_format": output_format,
                "size": f"{request_width}x{request_height}",
                "model": deployment,
            }
            files = {
                "image": ("source.png", img_buf.getvalue(), "image/png"),
                "mask": ("mask.png", mask_bytes, "image/png"),
            }
            response = None
            for attempt in range(1, request_retries + 1):
                try:
                    response = requests.post(
                        edit_url,
                        headers=headers,
                        data=data,
                        files=files,
                        timeout=timeout_s,
                    )
                    break
                except requests.exceptions.RequestException as req_err:
                    if attempt >= request_retries:
                        raise
                    logger.warning(
                        "Azure gpt-image edit request failed (attempt %s/%s); retrying: %s",
                        attempt,
                        request_retries,
                        req_err,
                    )
                    time.sleep(retry_delay_s * attempt)
            if response is None:
                raise RuntimeError("Azure gpt-image edit request did not return a response")
            try:
                response.raise_for_status()
            except requests.HTTPError as http_err:
                req_id = response.headers.get("x-ms-request-id", "")
                body = (response.text or "").strip()
                preview = (body[:600] + "...") if len(body) > 600 else body
                logger.warning(
                    "Azure gpt-image edit failed status=%s request_id=%s body=%s error=%s",
                    response.status_code,
                    req_id,
                    preview,
                    http_err,
                )
                raise RuntimeError(
                    f"Azure gpt-image edit failed status={response.status_code}: {preview}"
                ) from http_err
            payload = response.json()
            data_items = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data_items, list) or not data_items:
                raise ValueError("gpt-image edit response missing data array")
            first = data_items[0] if isinstance(data_items[0], dict) else {}
            b64 = first.get("b64_json")
            if not isinstance(b64, str) or not b64:
                raise ValueError("gpt-image edit response missing b64_json")
            raw = base64.b64decode(b64)
            return Image.open(io.BytesIO(raw)).convert("RGB")

        def inpaint(
            self,
            image_path: str,
            bbox: List[int],
            prompt: str,
            negative_prompt: str = "blurry, distorted, low quality",
        ) -> Optional[str]:
            try:
                with Image.open(image_path).convert("RGB") as src:
                    width, height = src.size
                normalized_bbox = _clamp_bbox(bbox, width, height, pad_pct=0.0)
                if normalized_bbox is None:
                    logger.warning("Skipping gpt-image edit because bbox is invalid: %s", bbox)
                    return None
                retry_sizes = self._build_retry_sizes(width, height)
                if not retry_sizes:
                    retry_sizes = [self._build_size(width, height)]
                edited = None
                last_err = ""
                for request_width, request_height in retry_sizes:
                    try:
                        edited = self._request_edit_image(
                            image_path=image_path,
                            bbox=normalized_bbox,
                            prompt=prompt,
                            request_width=request_width,
                            request_height=request_height,
                        )
                        if (request_width, request_height) != retry_sizes[0]:
                            logger.info(
                                "Azure gpt-image edit succeeded after fallback size=%sx%s",
                                request_width,
                                request_height,
                            )
                        break
                    except RuntimeError as req_err:
                        last_err = str(req_err)
                        if "pixel budget" in last_err.lower() or "invalid size" in last_err.lower():
                            logger.warning(
                                "Retrying Azure gpt-image edit with smaller size after error: %s",
                                last_err,
                            )
                            continue
                        raise
                if edited is None:
                    raise RuntimeError(last_err or "Azure gpt-image edit failed at all retry sizes")
                if edited.size != (width, height):
                    edited = edited.resize((width, height))
                if strict_bbox_lock and edited.size != (width, height):
                    raise ValueError(
                        f"gpt-image edit returned mismatched size {edited.size}, expected {(width, height)}"
                    )
                if composite_bbox_only:
                    with Image.open(image_path).convert("RGB") as src:
                        result_arr = _apply_mask_composite(src, edited, normalized_bbox, mask_pad_pct)
                    edited = Image.fromarray(result_arr)
                fd, out_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                edited.save(out_path, format="PNG")
                logger.info("Azure gpt-image edit succeeded for bbox=%s", normalized_bbox)
                return out_path
            except Exception as e:
                logger.warning("Azure gpt-image edit failed: %s", e)
                return None

    return AzureGPTImageInpainter()


def _patch_torch_xpu() -> None:
    """
    Patch torch.xpu for older PyTorch builds (< 2.4) that lack it.
    Newer diffusers checks torch.xpu.is_available() and other attributes at import.
    Uses __getattr__ as catch-all so any attribute access returns a safe default.
    """
    try:
        import torch
        if not hasattr(torch, "xpu"):
            import types

            class _XPUStub:
                """Stub for torch.xpu on older PyTorch that does not ship XPU support."""
                def is_available(self):
                    return False
                def device_count(self):
                    return 0
                def current_device(self):
                    return 0
                def get_device_name(self, *a, **kw):
                    return ""
                def __getattr__(self, name):
                    return lambda *a, **kw: None

            torch.xpu = _XPUStub()
    except Exception:
        pass


def _get_diffusers_inpainter(config: dict) -> Optional[Inpainter]:
    """Try to build a Diffusers-based inpainter. Returns None if unavailable."""
    _patch_torch_xpu()
    try:
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        from PIL import Image
    except ImportError as e:
        logger.warning("diffusers not installed or missing dependency: %s", e)
        return None
    except Exception as e:
        logger.warning("diffusers import failed: %s", e)
        return None

    model = config.get("inpaint_model") or "runwayml/stable-diffusion-inpainting"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    class DiffusersInpainter(Inpainter):
        def __init__(self):
            self.mask_pad_pct = float(config.get("inpaint_mask_pad_pct", 0.25))
            try:
                _patch_torch_xpu()
                logger.info("Loading inpainting model: %s (device=%s)", model, device)
                self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    model,
                    torch_dtype=dtype,
                )
                self.pipe = self.pipe.to(device)
                self.device = device
                logger.info("Inpainting model loaded successfully.")
            except Exception as e:
                logger.warning("Failed to load inpainting model: %s", e)
                self.pipe = None

        def inpaint(
            self,
            image_path: str,
            bbox: List[int],
            prompt: str,
            negative_prompt: str = "blurry, distorted, low quality",
        ) -> Optional[str]:
            if self.pipe is None:
                logger.warning("Inpainting model not loaded; skipping.")
                return None
            try:
                img = Image.open(image_path).convert("RGB")
                w_orig, h_orig = img.size
                mask_pil = _bbox_to_mask_pil(bbox, w_orig, h_orig)
                img_resized = img.resize((INPAINT_SIZE, INPAINT_SIZE), resample=Image.LANCZOS)
                mask_resized = mask_pil.resize((INPAINT_SIZE, INPAINT_SIZE), resample=Image.NEAREST)
                out = self.pipe(
                    prompt=prompt,
                    image=img_resized,
                    mask_image=mask_resized,
                    negative_prompt=negative_prompt,
                    guidance_scale=7.5,
                    num_inference_steps=config.get("inpaint_steps", 25),
                ).images[0]
                out_resized = out.resize((w_orig, h_orig), resample=Image.LANCZOS)
                result_arr = _apply_mask_composite(img, out_resized, bbox, self.mask_pad_pct)
                result = Image.fromarray(result_arr)
                fd, path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                result.save(path)
                return path
            except Exception as e:
                logger.warning("Inpainting failed: %s", e)
                return None

    try:
        inpainter = DiffusersInpainter()
        if inpainter.pipe is None:
            return None
        return inpainter
    except Exception as e:
        logger.warning("DiffusersInpainter construction failed: %s", e)
        return None


def get_inpainter(config: Optional[dict] = None) -> Inpainter:
    """
    Return an inpainter from config. If config has use_inpainting=True,
    use provider selected by inpaint_model.
    """
    config = config or {}
    if not config.get("use_inpainting", False):
        raise RuntimeError(
            "Realization requires real inpainting, but use_inpainting is false. "
            "Enable use_inpainting in realization config."
        )
    # Diffusers backend intentionally disabled.
    # impl = _get_diffusers_inpainter(config)
    # if impl is not None:
    #     logger.info("Using diffusers inpainting backend for model=%s", config.get("inpaint_model"))
    #     return impl
    gpt_image_impl = _get_gpt_image_inpainter(config)
    if gpt_image_impl is not None:
        logger.info("Using Azure gpt-image edit backend for model=%s", config.get("inpaint_model"))
        return gpt_image_impl
    flux_impl = _get_flux_inpainter(config)
    if flux_impl is not None:
        logger.info("Using FLUX image-edit backend for model=%s", config.get("inpaint_model"))
        return flux_impl
    raise RuntimeError(
        "Inpainting is enabled but no valid image-edit backend is available. "
        "Configure Azure gpt-image or FLUX endpoint/API key for real generation."
    )
