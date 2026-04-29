from typing import Any, Dict, Tuple

from src.realization.prompt_config import get_prompt, get_prompt_list


def build_prompt(
    original_label: str,
    new_label: str,
    target_culture: str,
    constraints: Dict[str, Any] = None,
    bbox: list[int] | None = None,
) -> Tuple[str, str]:
    constraints = constraints or {}
    visual = constraints.get("visual_attributes") or {}
    scene = constraints.get("scene_adaptation") or {}

    shape = str(visual.get("shape") or "authentic form")
    color = str(visual.get("color") or "locally appropriate colors")
    texture = str(visual.get("texture") or "natural texture")
    context_default = get_prompt("prompt_builder.context_default", "{target_culture} local context")
    context = str(visual.get("context") or context_default.format(target_culture=target_culture))

    scene_default = get_prompt("prompt_builder.scene_default", "{target_culture} everyday setting")
    scene_name = str(scene.get("scene") or scene_default.format(target_culture=target_culture))
    scene_elements = scene.get("elements") or get_prompt_list(
        "prompt_builder.scene_elements_default",
        ["local props", "cultural markers"],
    )
    scene_lighting = str(scene.get("lighting") or "natural lighting")
    scene_style = str(scene.get("style") or "realistic")
    scene_elements_text = ", ".join(str(x) for x in scene_elements)
    bbox_instruction = ""
    if isinstance(bbox, list) and len(bbox) >= 4:
        bbox_instruction = (
            f"Edit only inside bbox [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]. "
            "Do not alter pixels outside the masked target region."
        )

    if "scene region" in (original_label or "").lower():
        new_label = (
            f"{new_label}. Keep infographic layout, panel geometry, icon placement, and non-target text blocks stable"
        )

    prompt_template = get_prompt(
        "prompt_builder.prompt_template",
        (
            "Replace {original_label} with {new_label}. Target culture: {target_culture}. "
            "Keep shape: {shape}. Keep color: {color}. Keep texture: {texture}. Keep context: {context}. "
            "Place within scene: {scene_name}. Include elements: {scene_elements}. "
            "Lighting: {scene_lighting}. Style: {scene_style}. "
            "Preserve camera angle, perspective, and surroundings outside mask. "
            "{bbox_instruction}"
        ),
    )
    prompt = prompt_template.format(
        original_label=original_label,
        new_label=new_label,
        target_culture=target_culture,
        shape=shape,
        color=color,
        texture=texture,
        context=context,
        scene_name=scene_name,
        scene_elements=scene_elements_text,
        scene_lighting=scene_lighting,
        scene_style=scene_style,
        bbox_instruction=bbox_instruction,
    )
    negative_template = get_prompt(
        "prompt_builder.negative_template",
        "elements from {original_label}, western setting, mismatched culture artifacts",
    )
    negative = negative_template.format(original_label=original_label)
    return prompt, negative
