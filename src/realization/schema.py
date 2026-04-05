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
    if edit_plan and objects:
        preservations = edit_plan.get("preservations", [])
        transformations = edit_plan.get("transformations", [])
        tr_by_original, tr_by_object_id = _build_transformation_maps(transformations)
        replace_list = []
        for obj in objects:
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
            if _norm_label(new_label) in {_norm_label(original), _norm_label(current)}:
                continue
            bbox = obj.get("bbox")
            replace_list.append({
                "object_id": oid,
                "original": original,
                "new": new_label,
                "bbox": bbox if isinstance(bbox, list) and len(bbox) >= 4 else None,
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
