import json
from src.realization.models import EditPlan


def _norm_label(value: str) -> str:
    return (value or "").strip().lower()


def _build_transformation_maps(transformations: list) -> tuple:
    """
    Build lookup maps from reasoning transformations so object replacements can
    still be derived when objects do not carry original_class_name.
    """
    by_original = {}
    by_object_id = {}
    for t in transformations or []:
        if not isinstance(t, dict):
            continue
        original = _norm_label(t.get("original_object"))
        target = (t.get("target_object") or "").strip()
        if original and target:
            by_original[original] = target
        oid = t.get("object_id")
        if oid is not None and target:
            by_object_id[str(oid)] = target
    return by_original, by_object_id


def get_edit_plan_schema() -> dict:
    """
    Returns the JSON schema for the EditPlan.
    """
    return EditPlan.model_json_schema()


def adapt_plan_to_edit_format(plan_data: dict) -> dict:
    """
    Converts reasoning-stage output (preservations, transformations) into
    EditPlan shape (preserve, replace, edit_text, adjust_style) so realization
    can consume it. If the plan already has EditPlan keys, returns as-is.

    When plan_data is a full Stage 2 JSON with "edit_plan" and "objects",
    build replace actions from objects that have original_class_name (so each
    replace has object_id, original, new, bbox for inpainting).
    """
    if "preserve" in plan_data and "replace" in plan_data:
        return plan_data

    # Stage 2 format: edit_plan embedded + objects with original_class_name and bbox
    edit_plan = plan_data.get("edit_plan") or {}
    objects = plan_data.get("objects", [])
    if isinstance(edit_plan, dict) and edit_plan:
        preservations = edit_plan.get("preservations", [])
        transformations = edit_plan.get("transformations", [])
        tr_by_original, tr_by_object_id = _build_transformation_maps(transformations)
        visual_by_original = {}
        for t in transformations:
            if not isinstance(t, dict):
                continue
            key = _norm_label(t.get("original_object"))
            if key:
                visual_by_original[key] = t.get("visual_attributes")
        replace_list = []
        for obj in objects if isinstance(objects, list) else []:
            if not isinstance(obj, dict):
                continue
            oid = obj.get("id", len(replace_list))
            original = (
                obj.get("original_class_name")
                or obj.get("original_label")
                or obj.get("label")
                or obj.get("class_name")
                or ""
            )
            current = (obj.get("class_name") or obj.get("label") or "").strip()
            mapped_target = tr_by_object_id.get(str(oid))
            if not mapped_target:
                mapped_target = tr_by_original.get(_norm_label(original)) or tr_by_original.get(
                    _norm_label(current)
                )
            new_label = (
                obj.get("target_class_name")
                or obj.get("target_label")
                or obj.get("new_class_name")
                or obj.get("replacement_label")
                or mapped_target
                or ""
            )
            if not original or not new_label:
                continue
            # Keep actionable replacements when Stage 2 already rewrote class_name to target.
            # In that case current may equal new_label, but original_class_name still indicates
            # a real transform that should be executed with this object's bbox.
            has_original_marker = bool(obj.get("original_class_name") or obj.get("original_label"))
            if _norm_label(new_label) == _norm_label(original):
                continue
            if (
                _norm_label(new_label) == _norm_label(current)
                and not has_original_marker
            ):
                continue
            bbox = obj.get("bbox")
            visual_attributes = visual_by_original.get(_norm_label(original))
            if not visual_attributes:
                visual_attributes = visual_by_original.get(_norm_label(current))
            replace_list.append({
                "object_id": oid,
                "original": original,
                "new": new_label,
                "bbox": bbox if isinstance(bbox, list) and len(bbox) >= 4 else None,
                "constraints": {
                    "visual_attributes": visual_attributes,
                    "scene_adaptation": edit_plan.get("scene_adaptation"),
                },
            })

        # Optional region-level replacements for infographic/document rows when
        # object detection has no actionable bbox objects.
        for i, region in enumerate(edit_plan.get("region_replace", []) or []):
            if not isinstance(region, dict):
                continue
            original = (region.get("original") or "").strip()
            new_label = (region.get("new") or "").strip()
            bbox = region.get("bbox")
            if not original or not new_label:
                continue
            if _norm_label(new_label) == _norm_label(original):
                continue
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            replace_list.append({
                "object_id": int(region.get("object_id", 100000 + i)),
                "original": original,
                "new": new_label,
                "bbox": bbox[:4],
                "constraints": {
                    "visual_attributes": region.get("visual_attributes"),
                    "scene_adaptation": edit_plan.get("scene_adaptation"),
                },
            })
        return {
            "preserve": [p.get("original_object", "") for p in preservations if isinstance(p, dict)],
            "replace": replace_list,
            "edit_text": plan_data.get("edit_text", []) or edit_plan.get("edit_text", []),
            "adjust_style": plan_data.get("adjust_style") or edit_plan.get("adjust_style"),
        }

    preservations = plan_data.get("preservations", [])
    transformations = plan_data.get("transformations", [])
    return {
        "preserve": [p.get("original_object", "") for p in preservations if isinstance(p, dict)],
        "replace": [
            {
                "object_id": i,
                "original": t.get("original_object", ""),
                "new": t.get("target_object", ""),
            }
            for i, t in enumerate(transformations)
            if isinstance(t, dict)
        ],
        "edit_text": plan_data.get("edit_text", []),
        "adjust_style": plan_data.get("adjust_style"),
    }


def validate_edit_plan(plan_data: dict) -> EditPlan:
    """
    Validates a dictionary against the EditPlan schema.
    """
    return EditPlan(**plan_data)
