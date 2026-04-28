"""
Inpainting backends for object/clothing replacement.
Mask convention: white = region to inpaint, black = keep.
"""
import logging
import os
import tempfile
import base64
import io
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
    prefers FLUX backend; otherwise MockInpainter.
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
    flux_impl = _get_flux_inpainter(config)
    if flux_impl is not None:
        logger.info("Using FLUX image-edit backend for model=%s", config.get("inpaint_model"))
        return flux_impl
    raise RuntimeError(
        "Inpainting is enabled but no valid image-edit backend is available. "
        "Configure FLUX endpoint/API key for real generation."
    )
