import logging
import os
import tempfile
from typing import Dict, Any, List, Optional
import numpy as np
import re

from src.realization.models import EditPlan, ReplaceAction, EditTextAction, AdjustStyleAction
from src.realization.inpaint import get_inpainter, _build_inpaint_prompt
from src.realization.prompt_refiner import refine_inpaint_prompt
from src.realization.metrics import cultural_score, object_presence_score
from src.realization.prompt_builder import build_prompt

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
        self._text_quality_config = self.config.get("text_quality_gate", {})
        self._clip_components = None
        self._debug_prompt = bool(self.config.get("debug_prompt", False))
        self._run_metrics = {}
        self._validation_config = self.config.get("validation", {})
        self._max_inpaint_prompt_passes = max(1, int(self.config.get("max_inpaint_prompt_passes", 1)))

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
        final_image_path = current_path if current_path != input_image_path else input_image_path
        target_culture = str(self.config.get("target_culture", "")).strip()
        target_objects = [r.new for r in plan.replace if isinstance(r.new, str) and r.new.strip()]
        self._run_metrics = {
            "cultural_score": round(cultural_score(final_image_path, target_culture), 4),
            "object_presence_score": round(object_presence_score(final_image_path, target_objects), 4),
        }
        logger.info(
            "Realization metrics: cultural_score=%.4f, object_presence_score=%.4f",
            self._run_metrics["cultural_score"],
            self._run_metrics["object_presence_score"],
        )
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
        target_culture = self.config.get("target_culture") or "target"
        constraints = action.constraints if isinstance(action.constraints, dict) else {}
        base_prompt, negative = build_prompt(
            original_label=action.original,
            new_label=action.new,
            target_culture=target_culture,
            constraints=constraints,
        )
        fallback = _build_inpaint_prompt(action.original, action.new, str(target_culture))
        prompt_candidates = [
            base_prompt,
            f"{base_prompt}. Strong cultural fidelity.",
            f"{fallback}. {base_prompt}",
        ][: self._max_inpaint_prompt_passes]
        if self.config.get("use_llm_prompt_refinement"):
            prompt_candidates[0] = refine_inpaint_prompt(
                action.original,
                action.new,
                target_culture,
                prompt_candidates[0],
            )
        for pass_idx, prompt in enumerate(prompt_candidates, start=1):
            if self._debug_prompt:
                logger.info("Stage-3 final prompt for object_id=%s: %s", action.object_id, prompt)
            candidate_path = self._inpainter.inpaint(
                image_path,
                action.bbox,
                prompt,
                negative_prompt=negative,
            )
            if not candidate_path:
                continue
            if self._fails_local_quality_gate(image_path, candidate_path, action.bbox):
                logger.warning(
                    "Rejected inpainted replacement for object_id=%s by quality gate (pass=%d).",
                    action.object_id,
                    pass_idx,
                )
                try:
                    os.remove(candidate_path)
                except OSError:
                    pass
                continue
            return candidate_path
        return None

    def _edit_text(self, image_path: str, action: EditTextAction) -> Optional[str]:
        """
        Performs text replacement by drawing translated text in the target bbox.
        """
        logger.info("Editing text in %s: '%s' -> '%s'", action.bbox, action.original, action.translated)
        try:
            from PIL import Image
            src_img = Image.open(image_path).convert("RGB")
            source_arr = np.array(src_img)
            x1, y1, x2, y2 = [int(v) for v in action.bbox[:4]]
            x1, x2 = max(0, min(x1, x2)), min(src_img.width, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(src_img.height, max(y1, y2))
            if x2 <= x1 or y2 <= y1:
                return None
            bbox = [x1, y1, x2, y2]

            candidate, fg, bg = self._render_text_candidate(src_img, action, bbox=bbox)
            candidate_arr = np.array(candidate)
            failed, metrics = self._fails_text_quality_gate(source_arr, candidate_arr, bbox, fg, bg)

            final_img = candidate
            if failed:
                size_scale = 1.15 if metrics.get("occupancy_ratio", 0.0) < self._text_quality_min_occupancy() else 0.9
                retry_img, retry_fg, retry_bg = self._render_text_candidate(
                    src_img,
                    action,
                    bbox=bbox,
                    size_scale=size_scale,
                    force_high_contrast=True,
                )
                retry_arr = np.array(retry_img)
                retry_failed, retry_metrics = self._fails_text_quality_gate(
                    source_arr, retry_arr, bbox, retry_fg, retry_bg
                )
                if retry_failed:
                    logger.warning(
                        "Rejected text edit for bbox %s by text quality gate (first=%s, retry=%s).",
                        action.bbox,
                        metrics,
                        retry_metrics,
                    )
                    return None
                final_img = retry_img

            fd, out_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            final_img.save(out_path)
            skip_text_local_gate = bool(self._text_quality_config.get("skip_local_quality_gate", True))
            if (not skip_text_local_gate) and self._fails_local_quality_gate(
                image_path, out_path, bbox, edit_kind="text"
            ):
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
        candidates = self._font_candidates_for_family(family, weight)
        for name in candidates:
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _render_text_candidate(
        self,
        src_img,
        action: EditTextAction,
        bbox: List[int],
        size_scale: float = 1.0,
        force_high_contrast: bool = False,
    ):
        from PIL import ImageDraw
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        img = src_img.copy()
        draw = ImageDraw.Draw(img)
        style = action.style or {}
        bg = self._as_rgb_tuple(style.get("background_color"), fallback=(245, 245, 245))
        fg = self._as_rgb_tuple(style.get("text_color"), fallback=(20, 20, 20))
        if force_high_contrast:
            fg = self._pick_high_contrast_text_color(bg)
        draw.rectangle([x1, y1, x2, y2], fill=bg)
        text = action.translated.strip() or action.original
        box_h = max(1, y2 - y1)
        src_font_size = int(style.get("font_size", max(12, int(box_h * 0.65))))
        scaled_size = int(max(8, min(72, round(src_font_size * max(0.7, min(1.3, size_scale))))))
        font = self._fit_font_to_box(
            text=text,
            width=(x2 - x1),
            height=(y2 - y1),
            family=style.get("font_family"),
            size=scaled_size,
            weight=style.get("font_weight"),
        )
        tx, ty = self._compute_text_origin(
            draw=draw,
            text=text,
            font=font,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )
        draw.text((tx, ty), text, fill=fg, font=font)
        return img, fg, bg

    def _as_rgb_tuple(self, value: Any, fallback: tuple) -> tuple:
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                return (int(value[0]), int(value[1]), int(value[2]))
            except Exception:
                return fallback
        return fallback

    def _pick_high_contrast_text_color(self, background_rgb: tuple) -> tuple:
        black = (10, 10, 10)
        white = (245, 245, 245)
        return white if self._contrast_ratio(white, background_rgb) >= self._contrast_ratio(black, background_rgb) else black

    def _relative_luminance(self, rgb: tuple) -> float:
        def _to_linear(c: float) -> float:
            c = c / 255.0
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

        r, g, b = [max(0.0, min(255.0, float(v))) for v in rgb]
        r_l, g_l, b_l = _to_linear(r), _to_linear(g), _to_linear(b)
        return 0.2126 * r_l + 0.7152 * g_l + 0.0722 * b_l

    def _contrast_ratio(self, fg_rgb: tuple, bg_rgb: tuple) -> float:
        l1 = self._relative_luminance(fg_rgb)
        l2 = self._relative_luminance(bg_rgb)
        lighter = max(l1, l2)
        darker = min(l1, l2)
        return float((lighter + 0.05) / (darker + 0.05))

    def _text_quality_min_occupancy(self) -> float:
        return float(self._text_quality_config.get("min_bbox_occupancy", 0.02))

    def _fails_text_quality_gate(
        self,
        source_arr: np.ndarray,
        output_arr: np.ndarray,
        bbox: List[int],
        fg_rgb: tuple,
        bg_rgb: tuple,
    ) -> tuple:
        if not self._text_quality_config.get("enabled", True):
            return False, {}
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        x1, x2 = max(0, min(x1, x2)), min(source_arr.shape[1], max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(source_arr.shape[0], max(y1, y2))
        if x2 <= x1 or y2 <= y1:
            return False, {}
        src_patch = source_arr[y1:y2, x1:x2].astype(np.float32)
        out_patch = output_arr[y1:y2, x1:x2].astype(np.float32)
        diff = np.linalg.norm(out_patch - src_patch, axis=2)
        change_mask = diff > float(self._text_quality_config.get("changed_pixel_threshold", 10.0))
        occupancy_ratio = float(change_mask.mean())
        color_delta = float(np.linalg.norm(np.array(fg_rgb, dtype=np.float32) - np.array(bg_rgb, dtype=np.float32)))
        contrast_ratio = self._contrast_ratio(fg_rgb, bg_rgb)

        min_occ = self._text_quality_min_occupancy()
        max_occ = float(self._text_quality_config.get("max_bbox_occupancy", 0.9))
        min_delta = float(self._text_quality_config.get("min_color_delta", 35.0))
        min_contrast = float(self._text_quality_config.get("min_contrast_ratio", 2.4))
        failed = occupancy_ratio < min_occ or occupancy_ratio > max_occ or color_delta < min_delta or contrast_ratio < min_contrast
        metrics = {
            "occupancy_ratio": round(occupancy_ratio, 4),
            "color_delta": round(color_delta, 2),
            "contrast_ratio": round(float(contrast_ratio), 2),
        }
        return failed, metrics

    def _font_candidates_for_family(self, family: Optional[str], weight: Optional[str]) -> List[str]:
        candidates: List[str] = []
        family_text = (family or "").strip()
        family_lower = family_text.lower()
        is_bold = isinstance(weight, str) and weight.lower() in {"bold", "700", "800", "900"}

        if family_text:
            candidates.append(family_text)
            base_name = os.path.basename(family_text)
            if base_name and base_name not in candidates:
                candidates.append(base_name)

            simple = re.sub(r"[^a-z0-9]", "", family_lower)
            family_map = {
                "arial": ("arial.ttf", "arialbd.ttf"),
                "calibri": ("calibri.ttf", "calibrib.ttf"),
                "segoeui": ("segoeui.ttf", "segoeuib.ttf"),
                "timesnewroman": ("times.ttf", "timesbd.ttf"),
                "verdana": ("verdana.ttf", "verdanab.ttf"),
                "tahoma": ("tahoma.ttf",),
            }
            for key, mapped in family_map.items():
                if key in simple:
                    if is_bold and len(mapped) > 1:
                        candidates.extend([mapped[1], mapped[0]])
                    else:
                        candidates.extend(list(mapped))
                    break

        if is_bold:
            candidates.extend(["arialbd.ttf", "calibrib.ttf", "segoeuib.ttf", "timesbd.ttf", "verdanab.ttf"])
        candidates.extend(["arial.ttf", "calibri.ttf", "segoeui.ttf", "times.ttf", "verdana.ttf"])

        seen = set()
        deduped: List[str] = []
        for name in candidates:
            key = (name or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(name)
        return deduped

    def _fit_font_to_box(
        self,
        text: str,
        width: int,
        height: int,
        family: Optional[str],
        size: int,
        weight: Optional[str],
    ):
        from PIL import Image, ImageDraw
        probe = Image.new("RGB", (max(8, width), max(8, height)), color=(255, 255, 255))
        draw = ImageDraw.Draw(probe)
        target_w = max(4, int(width) - 8)
        target_h = max(4, int(height) - 6)
        min_size = 8
        cur_size = int(max(min_size, size))

        while cur_size >= min_size:
            font = self._load_font(family, cur_size, weight)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = max(1, int(bbox[2] - bbox[0]))
            th = max(1, int(bbox[3] - bbox[1]))
            if tw <= target_w and th <= target_h:
                return font
            cur_size -= 1
        return self._load_font(family, min_size, weight)

    def _compute_text_origin(self, draw, text: str, font, x1: int, y1: int, x2: int, y2: int) -> List[int]:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = max(1, int(bbox[2] - bbox[0]))
        th = max(1, int(bbox[3] - bbox[1]))
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        tx = x1 + max(2, min(width - tw - 2, 4))
        ty = y1 + max(1, (height - th) // 2)
        return [int(tx), int(ty)]

    def _fails_local_quality_gate(
        self,
        source_path: str,
        output_path: str,
        bbox: List[int],
        edit_kind: str = "object",
    ) -> bool:
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
            if edit_kind == "text":
                max_mean = float(
                    self._quality_gate_config.get(
                        "text_max_mean_distance",
                        self._quality_gate_config.get("max_mean_distance", 95.0) * 1.45,
                    )
                )
                max_std = float(
                    self._quality_gate_config.get(
                        "text_max_std_distance",
                        self._quality_gate_config.get("max_std_distance", 80.0) * 1.35,
                    )
                )
            else:
                max_mean = float(self._quality_gate_config.get("max_mean_distance", 95.0))
                max_std = float(self._quality_gate_config.get("max_std_distance", 80.0))
            if mean_dist > max_mean or std_dist > max_std:
                return True

            use_ssim = bool(self._quality_gate_config.get("use_ssim", True))
            if edit_kind == "text":
                use_ssim = bool(self._quality_gate_config.get("text_use_ssim", False))
            if use_ssim:
                if self._fails_ssim_gate(src, out, x1, y1, x2, y2, nx1, ny1, nx2, ny2):
                    return True

            use_clip_local = bool(self._quality_gate_config.get("use_clip_local", True))
            if edit_kind == "text":
                use_clip_local = bool(self._quality_gate_config.get("text_use_clip_local", False))
            if use_clip_local:
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

    def get_run_metrics(self) -> Dict[str, float]:
        return dict(self._run_metrics)

    def passes_composite_validation(
        self,
        image_path: str,
        target_culture: str,
        target_objects: List[str],
        prompt_scores: Optional[List[float]] = None,
    ) -> bool:
        culture = cultural_score(image_path, target_culture)
        object_score = object_presence_score(image_path, target_objects)
        grounding = float(sum(prompt_scores or [0.0]) / max(1, len(prompt_scores or [0.0])))
        threshold = float(self._validation_config.get("composite_threshold", 1.5))
        composite = culture + object_score + grounding
        logger.info(
            "Composite validation: culture=%.4f object=%.4f grounding=%.4f composite=%.4f threshold=%.4f",
            culture,
            object_score,
            grounding,
            composite,
            threshold,
        )
        return composite >= threshold
