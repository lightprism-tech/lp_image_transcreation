import logging
import re
from typing import List

logger = logging.getLogger(__name__)

_CLIP_COMPONENTS = None


def _get_clip_components():
    global _CLIP_COMPONENTS
    if _CLIP_COMPONENTS is not None:
        return _CLIP_COMPONENTS
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        model_name = "openai/clip-vit-base-patch32"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        _CLIP_COMPONENTS = (model, processor, device)
    except Exception as exc:
        logger.debug("CLIP metrics unavailable: %s", exc)
        _CLIP_COMPONENTS = ()
    return _CLIP_COMPONENTS


def _clip_image_text_similarity(image_path: str, text: str) -> float:
    comps = _get_clip_components()
    if not comps:
        return 0.0
    try:
        import torch
        from PIL import Image

        model, processor, device = comps
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=[text], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sim = float((image_features * text_features).sum().item())
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))
    except Exception as exc:
        logger.debug("Failed CLIP similarity scoring: %s", exc)
        return 0.0


def cultural_score(image_path: str, target_culture: str) -> float:
    target = (target_culture or "").strip()
    if not target:
        return 0.0
    prompt = f"A culturally authentic image from {target} with distinct local context."
    return _clip_image_text_similarity(image_path, prompt)


def object_presence_score(image_path: str, target_objects: List[str]) -> float:
    objects = [o.strip() for o in (target_objects or []) if isinstance(o, str) and o.strip()]
    if not objects:
        return 0.0
    scores = []
    for obj in objects:
        prompt = f"An image containing {obj}."
        scores.append(_clip_image_text_similarity(image_path, prompt))
    return float(sum(scores) / max(1, len(scores)))


def prompt_grounding_score(prompt: str) -> float:
    text = (prompt or "").strip().lower()
    if not text:
        return 0.0
    has_attributes = all(token in text for token in ["color", "shape", "texture", "context"])
    has_cultural_cue = any(token in text for token in ["cultural", "regional", "traditional", "local"])
    length_score = min(1.0, len(text.split()) / 40.0)
    lexical_density = len(set(re.findall(r"[a-zA-Z]+", text))) / max(1, len(text.split()))
    base = 0.35 * length_score + 0.35 * min(1.0, lexical_density)
    if has_attributes:
        base += 0.2
    if has_cultural_cue:
        base += 0.1
    return max(0.0, min(1.0, base))
