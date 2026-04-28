import argparse
import json
import logging
import os
import shutil
import sys

from src.realization.engine import RealizationEngine
from src.realization.schema import adapt_plan_to_edit_format, validate_edit_plan
from src.realization.models import EditPlan

logger = logging.getLogger(__name__)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _get_font(img_width, size_factor=40):
    """Get a font for drawing; fallback to default if system font missing."""
    try:
        from PIL import ImageFont
        return ImageFont.truetype("arial.ttf", min(24, max(12, img_width // size_factor)))
    except (OSError, IOError):
        try:
            return ImageFont.load_default()
        except Exception:
            return None


def _apply_mock_instance_changes(
    plan: EditPlan, input_path: str, output_path: str, target_culture: str
) -> None:
    """
    When inpainting is not available, draw per-instance changes: for each replace
    action with a bbox, draw a semi-transparent tint and a label (new object name)
    so the output shows edits per instance instead of only a single overlay.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        _apply_mock_overlay(input_path, output_path, target_culture)
        return
    img = Image.open(input_path).convert("RGBA")
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    font = _get_font(w, 50)
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
    for i, action in enumerate(plan.replace):
        if not action.bbox or len(action.bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in action.bbox[:4]]
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        color = (100, 150, 255, 80)
        draw_overlay.rectangle([x1, y1, x2, y2], fill=color, outline=(80, 120, 200, 200), width=2)
        label = (action.new or "?").strip()
        if len(label) > 25:
            label = label[:22] + "..."
        if font:
            try:
                tb = draw_overlay.textbbox((0, 0), label, font=font)
                lw, lh = tb[2] - tb[0], tb[3] - tb[1]
                lx = x1 + 4
                ly = y1 + 4
                if ly + lh > y2 - 2:
                    ly = y2 - lh - 2
                if lx + lw > x2 - 2:
                    lx = x2 - lw - 2
                draw_overlay.rectangle([lx - 2, ly - 1, lx + lw + 2, ly + lh + 1], fill=(0, 0, 0, 180))
                draw_overlay.text((lx, ly), label, fill=(255, 255, 255, 255), font=font)
            except Exception:
                pass
    out_rgba = Image.alpha_composite(img, overlay)
    out_rgb = out_rgba.convert("RGB")
    draw_final = ImageDraw.Draw(out_rgb)
    corner_label = f"Adapted for {target_culture}"
    font_corner = _get_font(w, 40)
    if font_corner:
        try:
            tb = draw_final.textbbox((0, 0), corner_label, font=font_corner)
            cw, ch = tb[2] - tb[0], tb[3] - tb[1]
            cx, cy = w - cw - 16, h - ch - 16
            draw_final.rectangle([cx - 4, cy - 2, cx + cw + 4, cy + ch + 2], fill=(0, 0, 0), outline=(255, 255, 255))
            draw_final.text((cx, cy), corner_label, fill=(255, 255, 255), font=font_corner)
        except Exception:
            pass
    out_rgb.save(output_path, "PNG")


def _apply_mock_overlay(input_path: str, output_path: str, target_culture: str) -> None:
    """
    In mock mode, copy the input image and add a small adaptation label so the
    output is visibly different. Real pixel-level edits (inpainting, text
    replacement) are not implemented yet.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        shutil.copy2(input_path, output_path)
        return
    img = Image.open(input_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    label = f"Adapted for {target_culture}"
    font = _get_font(img.width)
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
    if font:
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = img.width - tw - 16
        y = img.height - th - 16
        draw.rectangle((x - 4, y - 2, x + tw + 4, y + th + 2), fill=(0, 0, 0), outline=(255, 255, 255))
        draw.text((x, y), label, fill=(255, 255, 255), font=font)
    img.save(output_path, "PNG")


def _apply_mock_text_changes(plan: EditPlan, input_path: str, output_path: str, target_culture: str) -> None:
    """
    Apply text-region replacements in mock mode so Stage 3 still produces visible
    transcreation edits when text actions exist but pixel-generation is unavailable.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        _apply_mock_overlay(input_path, output_path, target_culture)
        return

    img = Image.open(input_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for action in plan.edit_text or []:
        if not action.bbox or len(action.bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in action.bbox[:4]]
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        style = action.style or {}
        bg = tuple(style.get("background_color", [245, 245, 245])[:3])
        fg = tuple(style.get("text_color", [20, 20, 20])[:3])
        draw.rectangle([x1, y1, x2, y2], fill=bg)

        text = (action.translated or action.original or "").strip()
        if not text:
            continue
        box_h = max(1, y2 - y1)
        size = int(style.get("font_size", max(10, int(box_h * 0.65))))
        size = max(8, min(72, size))
        font = _get_font(max(x2 - x1, 10), size_factor=max(12, int(500 / max(8, size))))
        if font is None:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
        if not font:
            continue
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        tx = x1 + 4
        ty = y1 + max(1, (box_h - th) // 2)
        if tx + tw > x2 - 2:
            tx = max(x1 + 2, x2 - tw - 2)
        draw.text((tx, ty), text, fill=fg, font=font)

    # Add corner label to indicate adaptation context in mock mode.
    label = f"Adapted for {target_culture}"
    font = _get_font(img.width)
    if font:
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        x = img.width - tw - 16
        y = img.height - th - 16
        draw.rectangle((x - 4, y - 2, x + tw + 4, y + th + 2), fill=(0, 0, 0), outline=(255, 255, 255))
        draw.text((x, y), label, fill=(255, 255, 255), font=font)

    img.save(output_path, "PNG")


def _plan_actionability_summary(plan: EditPlan) -> dict:
    """Return counts used to decide whether realization can apply visible edits."""
    replace_count = len(plan.replace or [])
    replace_with_bbox_count = sum(
        1 for r in (plan.replace or []) if r.bbox and len(r.bbox) >= 4
    )
    return {
        "replace_count": replace_count,
        "replace_with_bbox_count": replace_with_bbox_count,
        "edit_text_count": len(plan.edit_text or []),
        "has_adjust_style": bool(plan.adjust_style),
    }

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    parser = argparse.ArgumentParser(description="Stage 3: Visual Realization Engine")
    parser.add_argument("--img", required=True, help="Path to the input image file")
    parser.add_argument("--plan", required=True, help="Path to the Edit-Plan JSON file")
    parser.add_argument("--output", required=True, help="Path to save the generated image")
    parser.add_argument("--config", help="Optional path to a configuration JSON file")
    parser.add_argument(
        "--allow-empty-plan",
        action="store_true",
        help=(
            "Allow mock output even when no actionable edits exist in the plan. "
            "By default realization fails fast to avoid silent no-op outputs."
        ),
    )

    args = parser.parse_args()

    # validate inputs
    if not os.path.exists(args.img):
        logger.error("Error: Input image not found at %s", args.img)
        sys.exit(1)
    
    if not os.path.exists(args.plan):
        logger.error("Error: Edit-Plan file not found at %s", args.plan)
        sys.exit(1)

    try:
        # Load plan and convert reasoning format to EditPlan if needed
        raw_plan = load_json(args.plan)
        plan_data = adapt_plan_to_edit_format(raw_plan)
        plan = validate_edit_plan(plan_data)
        edit_plan = raw_plan.get("edit_plan") or {}
        target_culture = edit_plan.get("target_culture") or raw_plan.get("target_culture", "target")
        plan_summary = _plan_actionability_summary(plan)

        # Load config if provided
        config = load_json(args.config) if args.config else {}
        config["target_culture"] = config.get("target_culture") or target_culture
        use_inpainting = bool(config.get("use_inpainting"))

        # Fail fast for plans that cannot produce visible cultural edits.
        replace_count = plan_summary["replace_count"]
        replace_with_bbox_count = plan_summary["replace_with_bbox_count"]
        edit_text_count = plan_summary["edit_text_count"]
        has_adjust_style = plan_summary["has_adjust_style"]
        has_any_action = bool(replace_count or edit_text_count or has_adjust_style)

        if not has_any_action and not args.allow_empty_plan:
            logger.error(
                "No actionable edits found in plan. transformations/replace is empty and there are no text/style edits. "
                "Re-run Stage 2 with a target culture that yields substitutions, or pass --allow-empty-plan to force mock output."
            )
            sys.exit(2)

        if (
            replace_count > 0
            and replace_with_bbox_count == 0
            and edit_text_count == 0
            and not has_adjust_style
            and not args.allow_empty_plan
        ):
            logger.error(
                "Plan contains replacements but no bounding boxes, so realization cannot inpaint any object. "
                "Use Stage 2 JSON that includes objects with original_class_name + bbox, or pass --allow-empty-plan to force mock output."
            )
            sys.exit(2)

        if use_inpainting and replace_count > 0 and replace_with_bbox_count == 0:
            logger.warning(
                "use_inpainting=true but no replace action has a bbox; inpainting cannot run for these replacements."
            )

        # Initialize engine
        engine = RealizationEngine(config=config)

        # Generate
        logger.info("Generating image based on plan: %s", args.plan)
        output_path = engine.generate(plan, args.img)

        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # If the engine wrote a real file at output_path, copy it to the requested output
        if os.path.exists(output_path):
            shutil.copy2(output_path, args.output)
            logger.info("Success! Image generated at: %s", output_path)
        else:
            logger.error(
                "Realization failed: no generated image was produced by the inpainting backend. "
                "Fallback rendering is disabled."
            )
            sys.exit(3)

        logger.info("Saved result to: %s", args.output)

    except Exception as e:
        logger.error("An error occurred during realization: %s", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
