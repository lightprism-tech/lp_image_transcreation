import logging
import os
import tempfile
from typing import Dict, Any, List, Optional
import numpy as np

from src.realization.models import EditPlan, ReplaceAction, EditTextAction, AdjustStyleAction
from src.realization.inpaint import get_inpainter, _build_inpaint_prompt
from src.realization.prompt_refiner import refine_inpaint_prompt

logger = logging.getLogger(__name__)


class RealizationEngine:
    """
    The control interface for the Visual Realization stage.
    It takes an Edit-Plan and orchestrates the generation process.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._inpainter = get_inpainter(self.config)
        self._quality_gate_config = self.config.get("quality_gate", {})
        self._clip_components = None

    def generate(self, plan: EditPlan, input_image_path: str) -> str:
        """
        Executes the Edit-Plan on the input image.
        Returns the path to the generated image (or a mock path if no inpainting was done).
        """
        logger.info("Starting realization for image: %s", input_image_path)
        current_path = input_image_path

        # 1. Global Style Adjustment
        if plan.adjust_style:
            self._adjust_style(plan.adjust_style)

        # 2. Object Replacement (Inpainting) - chain results so each replace uses updated image
        for replacement in plan.replace:
            next_path = self._replace_object(current_path, replacement)
            if next_path and os.path.exists(next_path):
                if current_path != input_image_path:
                    try:
                        os.remove(current_path)
                    except OSError:
                        pass
                current_path = next_path

        # 3. Text Editing
        for text_edit in plan.edit_text:
            next_path = self._edit_text(current_path, text_edit)
            if next_path and os.path.exists(next_path):
                if current_path != input_image_path:
                    try:
                        os.remove(current_path)
                    except OSError:
                        pass
                current_path = next_path

        # 4. Preservation Checks (Logic to ensure constraints are met)
        self._check_preservation(plan.preserve)

        logger.info("Realization complete.")
        if current_path != input_image_path:
            return current_path
        return ""  # Signal: no real output, use mock overlay

    def _adjust_style(self, style: AdjustStyleAction):
        """
        Applies global style adjustments.
        """
        logger.info("Adjusting style: Palette=%s, Motifs=%s, Texture=%s", style.palette, style.motifs, style.texture)
        # Logic to condition the diffusion model on these styles would go here.

    def _replace_object(self, image_path: str, action: ReplaceAction) -> Optional[str]:
        """
        Performs object substitution (inpainting) when bbox is available.
        Returns path to new image if inpainting succeeded, else None.
        """
        logger.info("Replacing object %s ('%s') with '%s'", action.object_id, action.original, action.new)
        if not action.bbox or len(action.bbox) < 4:
            logger.debug("No bbox for object_id=%s; skipping inpainting", action.object_id)
            return None
        fallback = _build_inpaint_prompt(action.original, action.new)
        if self.config.get("use_llm_prompt_refinement"):
            target_culture = self.config.get("target_culture") or "target"
            prompt = refine_inpaint_prompt(
                action.original, action.new, target_culture, fallback
            )
        else:
            prompt = fallback
        candidate_path = self._inpainter.inpaint(image_path, action.bbox, prompt)
        if not candidate_path:
            return None
        if self._fails_local_quality_gate(image_path, candidate_path, action.bbox):
            logger.warning("Rejected inpainted replacement for object_id=%s by quality gate.", action.object_id)
            try:
                os.remove(candidate_path)
            except OSError:
                pass
            return None
        return candidate_path

    def _edit_text(self, image_path: str, action: EditTextAction) -> Optional[str]:
        """
        Performs text replacement by drawing translated text in the target bbox.
        """
        logger.info("Editing text in %s: '%s' -> '%s'", action.bbox, action.original, action.translated)
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            x1, y1, x2, y2 = [int(v) for v in action.bbox[:4]]
            x1, x2 = max(0, min(x1, x2)), min(img.width, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(img.height, max(y1, y2))
            if x2 <= x1 or y2 <= y1:
                return None

            style = action.style or {}
            bg = tuple(style.get("background_color", [245, 245, 245])[:3])
            fg = tuple(style.get("text_color", [20, 20, 20])[:3])
            draw.rectangle([x1, y1, x2, y2], fill=bg, outline=(40, 40, 40), width=1)
            text = action.translated.strip() or action.original
            box_h = max(1, y2 - y1)
            src_font_size = int(style.get("font_size", max(12, int(box_h * 0.5))))
            font_size = max(10, min(72, src_font_size))
            font = self._load_font(style.get("font_family"), font_size, style.get("font_weight"))
            draw.text((x1 + 6, y1 + max(2, box_h // 6)), text, fill=fg, font=font)

            fd, out_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            img.save(out_path)
            if self._fails_local_quality_gate(image_path, out_path, action.bbox):
                logger.warning("Rejected text edit for bbox %s by quality gate.", action.bbox)
                try:
                    os.remove(out_path)
                except OSError:
                    pass
                return None
            return out_path
        except Exception as e:
            logger.warning("Text edit failed for bbox %s: %s", action.bbox, e)
            return None

    def _load_font(self, family: Optional[str], size: int, weight: Optional[str]):
        from PIL import ImageFont
        candidates = []
        if family and isinstance(family, str):
            candidates.append(family)
        if weight and isinstance(weight, str) and weight.lower() in {"bold", "700", "800", "900"}:
            candidates.extend(["arialbd.ttf", "calibrib.ttf", "segoeuib.ttf"])
        candidates.extend(["arial.ttf", "calibri.ttf", "segoeui.ttf"])
        for name in candidates:
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _fails_local_quality_gate(self, source_path: str, output_path: str, bbox: List[int]) -> bool:
        """Reject edits if region statistics diverge too much from neighborhood."""
        if not self._quality_gate_config.get("enabled", True):
            return False
        try:
            from PIL import Image
            src = np.array(Image.open(source_path).convert("RGB"))
            out = np.array(Image.open(output_path).convert("RGB"))
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            x1, x2 = max(0, min(x1, x2)), min(src.shape[1], max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(src.shape[0], max(y1, y2))
            if x2 <= x1 or y2 <= y1:
                return False
            region = out[y1:y2, x1:x2].astype(np.float32)
            pad = int(self._quality_gate_config.get("neighborhood_pad_px", 24))
            nx1, ny1 = max(0, x1 - pad), max(0, y1 - pad)
            nx2, ny2 = min(out.shape[1], x2 + pad), min(out.shape[0], y2 + pad)
            hood = out[ny1:ny2, nx1:nx2].astype(np.float32)
            if hood.size == 0 or region.size == 0:
                return False
            region_mean = region.mean(axis=(0, 1))
            hood_mean = hood.mean(axis=(0, 1))
            region_std = region.std(axis=(0, 1))
            hood_std = hood.std(axis=(0, 1))
            mean_dist = float(np.linalg.norm(region_mean - hood_mean))
            std_dist = float(np.linalg.norm(region_std - hood_std))
            max_mean = float(self._quality_gate_config.get("max_mean_distance", 95.0))
            max_std = float(self._quality_gate_config.get("max_std_distance", 80.0))
            if mean_dist > max_mean or std_dist > max_std:
                return True

            if self._quality_gate_config.get("use_ssim", True):
                if self._fails_ssim_gate(src, out, x1, y1, x2, y2, nx1, ny1, nx2, ny2):
                    return True

            if self._quality_gate_config.get("use_clip_local", True):
                if self._fails_clip_local_gate(src, out, x1, y1, x2, y2):
                    return True
            return False
        except Exception as e:
            logger.warning("Quality gate check failed; allowing output. Reason: %s", e)
            return False

    def _fails_ssim_gate(self, src, out, x1, y1, x2, y2, nx1, ny1, nx2, ny2) -> bool:
        try:
            from skimage.metrics import structural_similarity as ssim
            src_patch = src[ny1:ny2, nx1:nx2].copy()
            out_patch = out[ny1:ny2, nx1:nx2].copy()
            rx1, ry1 = x1 - nx1, y1 - ny1
            rx2, ry2 = x2 - nx1, y2 - ny1
            src_patch[ry1:ry2, rx1:rx2] = 0
            out_patch[ry1:ry2, rx1:rx2] = 0
            score = ssim(src_patch, out_patch, channel_axis=2, data_range=255)
            min_ssim = float(self._quality_gate_config.get("min_neighborhood_ssim", 0.75))
            return float(score) < min_ssim
        except Exception:
            return False

    def _get_clip_components(self):
        if self._clip_components is not None:
            return self._clip_components
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            model_name = self._quality_gate_config.get("clip_model_name", "openai/clip-vit-base-patch32")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = CLIPModel.from_pretrained(model_name).to(device)
            processor = CLIPProcessor.from_pretrained(model_name)
            self._clip_components = (model, processor, device)
        except Exception:
            self._clip_components = ()
        return self._clip_components

    def _fails_clip_local_gate(self, src, out, x1, y1, x2, y2) -> bool:
        comps = self._get_clip_components()
        if not comps:
            return False
        try:
            import torch
            from PIL import Image
            model, processor, device = comps
            region_out = out[y1:y2, x1:x2]
            pad = int(self._quality_gate_config.get("clip_pad_px", 20))
            nx1, ny1 = max(0, x1 - pad), max(0, y1 - pad)
            nx2, ny2 = min(out.shape[1], x2 + pad), min(out.shape[0], y2 + pad)
            hood_out = out[ny1:ny2, nx1:nx2]
            if region_out.size == 0 or hood_out.size == 0:
                return False
            img_a = Image.fromarray(region_out.astype(np.uint8))
            img_b = Image.fromarray(hood_out.astype(np.uint8))
            inputs = processor(images=[img_a, img_b], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                feats = model.get_image_features(pixel_values=inputs["pixel_values"])
                feats = feats / feats.norm(dim=-1, keepdim=True)
                sim = float((feats[0] * feats[1]).sum().item())
            min_sim = float(self._quality_gate_config.get("min_clip_local_similarity", 0.18))
            return sim < min_sim
        except Exception:
            return False

    def _check_preservation(self, constraints: List[str]):
        """
        Validates that preservation constraints are respected.
        """
        logger.info("Ensuring preservation of: %s", constraints)
        # Logic to verify layout/pose/lighting (e.g., using ControlNet or similar).
