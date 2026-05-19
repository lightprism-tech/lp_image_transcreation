import copy
import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from src.reasoning.schemas import (
    ReasoningInput, TranscreationPlan, Transformation, Preservation, CulturalNode,
    StylePriors,
)
from src.reasoning.knowledge_loader import KnowledgeLoader
from src.reasoning.llm_client import LLMClient
from src.reasoning.prompt_config import get_prompt, get_prompt_list
from src.reasoning.policy_config import (
    get_policy_int,
    get_policy_set,
    get_policy,
    get_policy_dict,
    get_policy_list,
)

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _env_set(name: str, default: Set[str]) -> Set[str]:
    value = os.getenv(name)
    if value is None:
        return set(default)
    parsed = {part.strip().lower() for part in value.split(",") if part.strip()}
    return parsed or set(default)


# Default uses KG first, then Groq candidate generation when the KG has no match.
STRICT_KB_GROUNDED_TRANSFORMS = _env_bool("REASONING_STRICT_KB_GROUNDED_TRANSFORMS", False)
MIN_CULTURAL_DENSITY = _env_int("REASONING_MIN_CULTURAL_DENSITY", 2)
_SCENE_OVERRIDE_ALLOWED_IMAGE_TYPES = _env_set(
    "REASONING_SCENE_OVERRIDE_ALLOWED_IMAGE_TYPES",
    {"infographic", "poster", "ui"},
)
ALLOW_CROSS_TYPE_FORCED_FALLBACK = _env_bool("REASONING_ALLOW_CROSS_TYPE_FORCED_FALLBACK", False)
_DEFAULT_PLACEHOLDER_TEXT_TOKENS = {
    str(t).strip().lower()
    for t in get_prompt_list("tokens.placeholder_text", [])
    if str(t).strip()
}
PLACEHOLDER_TEXT_TOKENS = _env_set(
    "REASONING_PLACEHOLDER_TEXT_TOKENS", _DEFAULT_PLACEHOLDER_TEXT_TOKENS
)
_DEFAULT_INFOGRAPHIC_CULTURAL_CUE_TOKENS = {
    str(t).strip().lower()
    for t in get_prompt_list("tokens.infographic_cultural_cues", [])
    if str(t).strip()
}
INFOGRAPHIC_CULTURAL_CUE_TOKENS = _env_set(
    "REASONING_INFOGRAPHIC_CULTURAL_CUE_TOKENS", _DEFAULT_INFOGRAPHIC_CULTURAL_CUE_TOKENS
)
ENABLE_LLM_PLACEHOLDER_CLASSIFIER = _env_bool("REASONING_ENABLE_LLM_PLACEHOLDER_CLASSIFIER", True)


def apply_plan_to_input(input_data: Dict[str, Any], plan: TranscreationPlan) -> Dict[str, Any]:
    """
    Apply the transcreation plan to the input data in-place. Returns a deep copy of
    the input with the same structure; only values are replaced according to the
    target culture (class_name, caption, scene description, attributes where relevant).
    """
    output = copy.deepcopy(input_data)
    trans_map = {t.original_object.lower(): t for t in plan.transformations}

    for orig_lower, trans in trans_map.items():
        target = trans.target_object
        # Replace in objects
        for obj in output.get("objects", []):
            cn = obj.get("class_name") or obj.get("label") or ""
            if cn.lower() == orig_lower:
                obj["original_class_name"] = cn  # Keep for realization (inpainting/bbox mapping)
                obj["class_name"] = target
                if obj.get("label") is not None:
                    obj["label"] = target
                if obj.get("caption"):
                    obj["caption"] = _replace_word_in_text(obj["caption"], trans.original_object, target)
                attrs = obj.get("attributes") or {}
                if isinstance(attrs.get("clothing"), list):
                    obj["attributes"]["clothing"] = [
                        target if (isinstance(c, str) and c.lower() == orig_lower) else c
                        for c in attrs["clothing"]
                    ]
                if getattr(trans, "visual_attributes", None):
                    obj["visual_attributes"] = dict(trans.visual_attributes)
        # Replace in scene description
        scene = output.get("scene") or {}
        if scene.get("description"):
            scene["description"] = _replace_word_in_text(
                scene["description"], trans.original_object, target
            )

    return output


def _replace_word_in_text(text: str, original: str, target: str) -> str:
    """Replace whole-word occurrences of original with target, case-insensitive."""
    if not original or not text:
        return text
    pattern = re.compile(r"\b" + re.escape(original) + r"\b", re.IGNORECASE)
    return pattern.sub(target, text)


def _infer_cultural_type(
    obj_label: str, obj: Dict[str, Any], label_to_type: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """Infer a cultural type from KB label_to_type or object attributes so we can fetch KB candidates."""
    label_to_type = label_to_type or {}
    normalized_label_to_type: Dict[str, str] = {}
    for key, value in label_to_type.items():
        key_text = str(key or "").strip().lower()
        value_text = str(value or "").strip()
        if key_text and value_text:
            normalized_label_to_type[key_text] = value_text
    label_lower = (obj_label or "").lower()
    if label_lower in normalized_label_to_type:
        return normalized_label_to_type[label_lower]
    # Fuzzy token-level match for detector labels like "there_plate_sushi".
    label_tokens = [t for t in re.split(r"[^a-z0-9]+", label_lower) if t]
    for token in label_tokens:
        if token in normalized_label_to_type:
            return normalized_label_to_type[token]
    # If label_to_type is missing for this label, avoid hardcoded class maps.
    # Keep only lightweight signal extraction from object attributes.
    attrs = obj.get("attributes") or {}
    clothing = attrs.get("clothing")
    if isinstance(clothing, list) and clothing:
        return "CLOTHING"
    if isinstance(clothing, str) and clothing:
        return "CLOTHING"
    semantic_type = str(obj.get("semantic_type") or "").strip().lower()
    if semantic_type in {"icon", "symbol", "logo"}:
        return "SYMBOL"
    return None


def _filter_candidates_by_avoid(candidates: List[str], avoid_list: List[str]) -> Tuple[List[str], List[str]]:
    """
    Filter out candidates that appear in the avoid list (e.g. stereotypical edits).
    Returns (filtered_candidates, adherence_notes).
    """
    filtered = []
    adherence = []
    for c in candidates:
        c_lower = c.lower()
        excluded = False
        for av in avoid_list:
            av_lower = av.lower()
            if c_lower in av_lower or av_lower in c_lower:
                excluded = True
                adherence.append("Avoided '%s' per avoid list: '%s'" % (c, av[:60] + ("..." if len(av) > 60 else "")))
                break
        if not excluded:
            filtered.append(c)
    return filtered, adherence


def _build_text_edits_for_document(
    scene_graph: Dict[str, Any], target_culture: str, llm_client: Optional[LLMClient] = None
) -> List[Dict[str, Any]]:
    """
    Build simple text-region transcreation edits for document/infographic style images.
    This provides actionable edit_text entries for Stage 3 when many cultural objects
    are not detected.
    """
    text = scene_graph.get("text") or {}
    extracted = text.get("extracted") or []
    if not isinstance(extracted, list):
        return []

    edits: List[Dict[str, Any]] = []
    rewrite_cache: Dict[str, str] = {}
    max_edits = 8
    scene_context = ((scene_graph.get("scene") or {}).get("description") or "").strip()
    full_text = (text.get("full_text") or "").strip()
    for item in extracted:
        if not isinstance(item, dict):
            continue
        original = item.get("text")
        bbox = item.get("bbox")
        if not isinstance(original, str) or not original.strip():
            continue
        original_clean = original.strip()
        # Skip symbol-only/noise OCR tokens (e.g. "*") to avoid low-quality text edits.
        if len(original_clean) <= 1:
            continue
        if not re.search(r"[A-Za-z0-9]", original_clean):
            continue
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        culture_title = _rewrite_culture_title_text(original_clean, target_culture, scene_context)
        if culture_title:
            translated = culture_title
            rewrite_cache[_normalize_key(original_clean)] = translated
        elif _is_placeholder_text_dynamic(original_clean, llm_client=llm_client):
            logger.info("Skipping placeholder OCR text rewrite: '%s'", original_clean[:80])
            continue
        else:
            cache_key = _normalize_key(original_clean)
            translated = rewrite_cache.get(cache_key)
            if translated is None:
                translated = _rewrite_text_for_region(
                    original=original_clean,
                    target_culture=target_culture,
                    image_type=((scene_graph.get("image_type") or {}).get("type") or ""),
                    scene_context=scene_context,
                    full_text=full_text,
                    bbox=bbox,
                    style=item.get("style") if isinstance(item.get("style"), dict) else None,
                    llm_client=llm_client,
                )
                rewrite_cache[cache_key] = translated

        if translated != original:
            logger.info(
                "Text rewrite planned: '%s' -> '%s' (bbox=%s)",
                original[:80],
                translated[:80],
                bbox[:4],
            )
            edits.append(
                {
                    "bbox": [int(round(float(x))) for x in bbox[:4]],
                    "original": original,
                    "translated": translated,
                    "style": item.get("style") if isinstance(item.get("style"), dict) else None,
                }
            )
        if len(edits) >= max_edits:
            break
    return edits


def _rewrite_text_for_region(
    original: str,
    target_culture: str,
    image_type: str,
    scene_context: str,
    full_text: str,
    bbox: Optional[List[int]] = None,
    style: Optional[Dict[str, Any]] = None,
    llm_client: Optional[LLMClient] = None,
) -> str:
    original_clean = (original or "").strip()
    if not original_clean:
        return original
    if _is_placeholder_text_dynamic(original_clean, llm_client=llm_client):
        return original_clean
    if llm_client is None:
        candidate = re.sub(
            r"\bglobal\b",
            f"{target_culture} local",
            original_clean,
            flags=re.IGNORECASE,
        )
        return _validate_rewrite_constraints(original_clean, candidate, bbox=bbox, style=style)
    try:
        max_chars = _estimate_max_chars_for_region(original_clean, bbox=bbox, style=style)
        template = get_prompt(
            "rewrite_text_region.template",
            (
                "Rewrite one OCR text region for cultural transcreation.\n"
                "Target culture: {target_culture}\n"
                "Image type: {image_type}\n"
                "Scene context: {scene_context}\n"
                "Neighbor text context: {neighbor_text}\n"
                "Original region text: {original_text}\n\n"
                'Return exactly one JSON object: {"candidates":["...","...","..."]}\n'
                "Rules: keep semantics and intent, localize wording naturally for {target_culture}, preserve tone/register (formal vs informal), keep brand/product names unchanged when present, preserve placeholder/lorem text unchanged, avoid object substitution language, avoid stereotypes, avoid adding culture labels unless they already exist in the original text, and each candidate length <= {max_chars}."
            ),
        )
        prompt = template.format(
            target_culture=target_culture,
            image_type=image_type,
            scene_context=scene_context,
            neighbor_text=full_text[:500],
            original_text=original_clean,
            max_chars=max_chars,
        )
        result = llm_client.generate_reasoning(prompt)
        candidates = result.get("candidates")
        if isinstance(candidates, list):
            best = _pick_best_rewrite_candidate(original_clean, candidates, bbox=bbox, style=style)
            if best:
                return best
        rewritten = result.get("rewritten_text")
        if isinstance(rewritten, str) and rewritten.strip():
            return _validate_rewrite_constraints(original_clean, rewritten.strip(), bbox=bbox, style=style)
    except Exception:
        pass
    return original_clean


def _estimate_max_chars_for_region(original: str, bbox: Optional[List[int]], style: Optional[Dict[str, Any]]) -> int:
    if not bbox or len(bbox) < 4:
        return max(12, int(len(original) * 1.3))
    width = max(1.0, float(max(bbox[0], bbox[2]) - min(bbox[0], bbox[2])))
    height = max(1.0, float(max(bbox[1], bbox[3]) - min(bbox[1], bbox[3])))
    font_size = float((style or {}).get("font_size", 14))
    avg_char_w = max(5.0, font_size * 0.55)
    max_chars_per_line = int(width / avg_char_w)
    max_lines = max(1, int(height / (font_size * 1.35)))
    max_chars = max_chars_per_line * max_lines
    return max(8, min(200, max_chars))


def _is_placeholder_text(text: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]+", "", text or "")
    if compact and len(set(compact.lower())) == 1 and len(compact) >= 3:
        return True
    words = re.findall(r"[A-Za-z]+", (text or "").lower())
    if not words:
        return False
    placeholder_hits = sum(1 for word in words if word in PLACEHOLDER_TEXT_TOKENS)
    return placeholder_hits >= 2


def _is_placeholder_text_dynamic(text: str, llm_client: Optional[LLMClient] = None) -> bool:
    """Heuristic placeholder detection with optional LLM fallback."""
    if _is_placeholder_text(text):
        return True
    if (not ENABLE_LLM_PLACEHOLDER_CLASSIFIER) or llm_client is None:
        return False
    cleaned = (text or "").strip()
    if len(cleaned) < 8:
        return False
    words = re.findall(r"[A-Za-z]+", cleaned.lower())
    if not words:
        return False
    unique_ratio = float(len(set(words))) / float(len(words))
    likely_ambiguous = (len(words) >= 4 and unique_ratio <= 0.65) or (
        len(words) >= 8 and unique_ratio <= 0.8
    )
    if not likely_ambiguous:
        return False
    try:
        template = get_prompt(
            "classify_placeholder_text.template",
            (
                "Classify whether the OCR text is placeholder/dummy filler text.\n"
                "Text: {text}\n\n"
                'Return exactly one JSON object: {"is_placeholder": true or false}'
            ),
        )
        result = llm_client.generate_reasoning(template.format(text=cleaned[:200]))
        return bool(result.get("is_placeholder") is True)
    except Exception:
        return False


def _rewrite_culture_title_text(original: str, target_culture: str, scene_context: str) -> Optional[str]:
    """Rewrite prominent source-culture title text when Stage 1 context identifies it."""
    original_key = _normalize_key(original)
    target_key = _normalize_key(target_culture)
    context_key = _normalize_key(scene_context)
    if not original_key or not target_key or original_key == target_key:
        return None
    if original_key not in context_key:
        return None
    if len(original_key.split()) != 1:
        return None
    if original.strip().isupper():
        return target_culture.upper()
    return target_culture


def _validate_rewrite_constraints(original: str, candidate: str, bbox: Optional[List[int]], style: Optional[Dict[str, Any]]) -> str:
    if _is_placeholder_text(original):
        return original
    text = re.sub(r"\s+", " ", (candidate or "").strip())
    if not text:
        return original
    max_chars = _estimate_max_chars_for_region(original, bbox=bbox, style=style)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    min_len = max(2, int(len(original) * 0.45))
    max_len = max(int(len(original) * 1.55), min_len)
    if len(text) < min_len:
        return original
    if len(text) > max_len:
        return text[:max_len].rstrip()
    return text


def _pick_best_rewrite_candidate(
    original: str, candidates: List[Any], bbox: Optional[List[int]], style: Optional[Dict[str, Any]]
) -> Optional[str]:
    best = None
    best_score = -1e9
    max_chars = _estimate_max_chars_for_region(original, bbox=bbox, style=style)
    for c in candidates:
        if not isinstance(c, str):
            continue
        val = _validate_rewrite_constraints(original, c, bbox=bbox, style=style)
        len_penalty = abs(len(val) - len(original))
        overflow_penalty = max(0, len(val) - max_chars) * 2.5
        score = -float(len_penalty + overflow_penalty)
        if score > best_score:
            best_score = score
            best = val
    return best


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _normalize_text(value).lower()).strip()


def _tokenize_label(value: Any) -> List[str]:
    return [t for t in _normalize_key(value).split() if t]


def _scope_excluded_types() -> Set[str]:
    """Node types that are too coarse for bbox-level visual substitution."""
    return get_policy_set("scope_excluded_types")


def _fallback_localized_obj_type() -> str:
    return str(get_policy("fallback_localized_obj_type")).strip().upper()


def _semantic_type_fallback(semantic_type: str) -> Optional[str]:
    mapping = get_policy("semantic_type_fallbacks")
    if not isinstance(mapping, dict):
        return None
    value = mapping.get((semantic_type or "").strip().lower())
    if not value:
        return None
    return str(value).strip().upper()


def _type_inference_stopwords() -> Set[str]:
    return {str(t).strip().lower() for t in get_policy_list("type_inference_stopwords") if str(t).strip()}


def _filter_type_tokens(tokens: Set[str]) -> Set[str]:
    stopwords = _type_inference_stopwords()
    return {t for t in tokens if t and t not in stopwords and len(t) > 1}


def _infer_type_from_label_cues(obj: Dict[str, Any], source_label: str = "") -> Optional[str]:
    """Map perception label/caption tokens to a cultural type using policy keyword cues."""
    cues = get_policy_dict("type_label_cues")
    if not cues:
        return None
    tokens = _filter_type_tokens(set(_tokenize_label(_object_signal_text(obj, source_label))))
    best_type = None
    best_len = 0
    for cue, node_type in cues.items():
        if cue in tokens and len(cue) > best_len:
            best_type = node_type
            best_len = len(cue)
    return best_type


def _build_type_token_index(kg_loader: KnowledgeLoader) -> Dict[str, Set[str]]:
    """
    Build a dynamic index of cultural-type -> tokens from KB node labels.
    Used to infer object types from captions/labels without hardcoded maps.
    """
    index: Dict[str, Set[str]] = {}
    excluded = _scope_excluded_types()
    raw_labels = kg_loader.get_all_labels()
    if not isinstance(raw_labels, list):
        return index
    for label in raw_labels:
        if not isinstance(label, str) or not label.strip():
            continue
        node = kg_loader.find_node(label)
        if not node:
            continue
        node_type = str(node.type or "").upper()
        if not node_type or node_type in excluded:
            continue
        bucket = index.setdefault(node_type, set())
        bucket.update(_tokenize_label(label))
    return index


def _object_signal_text(obj: Dict[str, Any], label: str = "") -> str:
    parts: List[str] = []
    for field in ("label", "class_name", "original_class_name", "detector_label", "caption"):
        value = _normalize_text(obj.get(field))
        if value:
            parts.append(value)
    if label:
        parts.append(label)
    for item in obj.get("caption_candidates") or []:
        if isinstance(item, dict):
            value = _normalize_text(item.get("caption"))
            if value:
                parts.append(value)
    return " ".join(parts)


def _infer_cultural_type_from_kb_signals(
    obj: Dict[str, Any],
    type_token_index: Dict[str, Set[str]],
    label_to_type: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Infer a cultural type by matching object text tokens against KB-derived type vocabularies.
    """
    label_to_type = label_to_type or {}
    label_text = _normalize_text(obj.get("label") or obj.get("class_name"))
    label_lower = label_text.lower()
    if label_lower in label_to_type:
        mapped = str(label_to_type[label_lower] or "").upper()
        if mapped and mapped not in _scope_excluded_types():
            return mapped
    for token in _tokenize_label(label_text):
        if token in label_to_type:
            mapped = str(label_to_type[token] or "").upper()
            if mapped and mapped not in _scope_excluded_types():
                return mapped

    tokens = _filter_type_tokens(set(_tokenize_label(_object_signal_text(obj))))
    if not tokens or not type_token_index:
        return None

    best_type = None
    best_score = 0
    for node_type, hint_tokens in type_token_index.items():
        if not hint_tokens:
            continue
        score = len(tokens.intersection(_filter_type_tokens(hint_tokens)))
        if score > best_score:
            best_score = score
            best_type = node_type
    min_overlap = get_policy_int("type_inference_min_token_overlap")
    if best_score >= min_overlap and best_type:
        return best_type
    return None


def _resolve_obj_type_for_localized_edit(
    obj: Dict[str, Any],
    source_label: str,
    kg_loader: KnowledgeLoader,
    type_token_index: Dict[str, Set[str]],
) -> Tuple[str, str]:
    """
    Resolve object type and source culture from the original perception label/caption.
    Returns (obj_type, source_culture).
    """
    label_to_type = kg_loader.get_label_to_type()
    cued_type = _infer_type_from_label_cues(obj, source_label)
    inferred_kb = _infer_cultural_type_from_kb_signals(obj, type_token_index, label_to_type)
    inferred_from_attrs = _infer_cultural_type(source_label, obj, label_to_type)
    if inferred_from_attrs and inferred_from_attrs.upper() in _scope_excluded_types():
        inferred_from_attrs = None

    source_culture = "Unknown"
    obj_type = "object"
    source_node = kg_loader.find_node(source_label)
    if source_node:
        node_type = str(source_node.type or "").upper()
        source_culture = kg_loader.get_culture_of_node(source_node.id) or "Unknown"
        if node_type in _scope_excluded_types():
            obj_type = cued_type or inferred_kb or inferred_from_attrs or _fallback_localized_obj_type()
        else:
            obj_type = node_type
    elif cued_type:
        obj_type = cued_type
    elif inferred_kb:
        obj_type = inferred_kb
    elif inferred_from_attrs:
        obj_type = inferred_from_attrs

    semantic_type = str(obj.get("semantic_type") or "").strip().lower()
    semantic_fallback = _semantic_type_fallback(semantic_type)
    if semantic_fallback and obj_type in {"object", ""}:
        obj_type = semantic_fallback
    elif semantic_fallback and obj_type == "FOOD" and cued_type and cued_type != "FOOD":
        obj_type = cued_type
    elif semantic_fallback and obj_type == "FOOD" and not cued_type:
        semantic_tokens = _filter_type_tokens(set(_tokenize_label(_object_signal_text(obj, source_label))))
        food_tokens = _filter_type_tokens(type_token_index.get("FOOD") or set())
        if len(semantic_tokens.intersection(food_tokens)) < get_policy_int("type_inference_min_token_overlap"):
            obj_type = semantic_fallback
    return obj_type, source_culture


def _filter_labels_for_grounding(
    known_labels: List[str],
    kg_loader: KnowledgeLoader,
    exclude_types: Optional[Set[str]] = None,
    allowed_types: Optional[Set[str]] = None,
) -> List[str]:
    excluded = exclude_types or _scope_excluded_types()
    allowed = {str(t).upper() for t in (allowed_types or set()) if str(t).strip()}
    filtered: List[str] = []
    for label in known_labels:
        node = kg_loader.find_node(label)
        if not node:
            continue
        node_type = str(node.type or "").upper()
        if node_type in excluded:
            continue
        if allowed and node_type not in allowed:
            continue
        filtered.append(label)
    return filtered


def _embedding_grounding_enabled() -> bool:
    value = get_policy("use_embedding_label_grounding")
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _prioritize_unused_candidates(
    candidates: List[str],
    used_targets: Optional[Set[str]] = None,
) -> List[str]:
    """Deprioritize targets already chosen for other objects in the same plan."""
    if not candidates:
        return []
    used_keys = {_normalize_key(t) for t in (used_targets or set()) if _normalize_key(t)}
    if not used_keys:
        return list(candidates)
    fresh = [c for c in candidates if _normalize_key(c) not in used_keys]
    reused = [c for c in candidates if _normalize_key(c) in used_keys]
    return fresh + reused


def _select_target_with_diversity(
    candidates: List[str],
    raw_target: Any,
    used_targets: Optional[Set[str]] = None,
) -> Optional[str]:
    ranked = _prioritize_unused_candidates(candidates, used_targets)
    used_keys = {_normalize_key(t) for t in (used_targets or set()) if _normalize_key(t)}
    grounded = _select_grounded_target(raw_target, ranked)
    if grounded and _normalize_key(grounded) not in used_keys:
        return grounded
    for candidate in ranked:
        if _normalize_key(candidate) not in used_keys:
            return candidate
    return ranked[0] if ranked else None


def _dedupe_transformations_by_original(transformations: List[Transformation]) -> List[Transformation]:
    seen: Set[str] = set()
    deduped: List[Transformation] = []
    for item in transformations:
        key = _normalize_key(item.original_object)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _recover_grounded_label(
    obj: Dict[str, Any],
    kg_loader: KnowledgeLoader,
    *,
    exclude_scope_types: bool = False,
    allowed_types: Optional[Set[str]] = None,
) -> Optional[str]:
    """
    Recover a KB label hint for noisy detector outputs.

    Returns a label only when token evidence supports it. Embedding fallback is optional,
    type-filtered, and never used without token overlap to the source object text.
    """
    raw_candidates: List[str] = []
    for field in ("label", "class_name", "original_class_name", "detector_label", "caption"):
        value = _normalize_text(obj.get(field))
        if value:
            raw_candidates.append(value)
    for item in obj.get("caption_candidates") or []:
        if isinstance(item, dict):
            value = _normalize_text(item.get("caption"))
            if value:
                raw_candidates.append(value)
    if not raw_candidates:
        return None

    known_labels = kg_loader.get_all_labels()
    if not known_labels:
        return None
    excluded_types = _scope_excluded_types() if exclude_scope_types else None
    searchable_labels = _filter_labels_for_grounding(
        known_labels,
        kg_loader,
        exclude_types=excluded_types,
        allowed_types=allowed_types,
    )
    if not searchable_labels:
        return None
    source_tokens = _filter_type_tokens(set(_tokenize_label(" ".join(raw_candidates))))
    known_map = {_normalize_key(label): label for label in searchable_labels if _normalize_key(label)}

    # Pass 1: exact normalized match
    for candidate in raw_candidates:
        key = _normalize_key(candidate)
        if key in known_map:
            return known_map[key]

    # Pass 2: token overlap (dynamic matching to KB labels)
    for candidate in raw_candidates:
        c_tokens = _filter_type_tokens(set(_tokenize_label(candidate)))
        if not c_tokens:
            continue
        best_label = None
        best_overlap = 0
        for known in searchable_labels:
            k_tokens = _filter_type_tokens(set(_tokenize_label(known)))
            if not k_tokens:
                continue
            overlap = len(c_tokens.intersection(k_tokens))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = known
        min_overlap = get_policy_int("grounding_min_label_token_overlap")
        if best_label and best_overlap >= min_overlap:
            return best_label

    if not _embedding_grounding_enabled():
        return None

    # Pass 3: embedding ranking only when token overlap confirms the match
    ranked = kg_loader.rank_candidates_by_embedding(" ".join(raw_candidates[:4]), searchable_labels)
    min_emb_overlap = get_policy_int("grounding_min_embedding_token_overlap")
    for label in ranked[:8]:
        if not isinstance(label, str) or not label.strip():
            continue
        label_tokens = _filter_type_tokens(set(_tokenize_label(label)))
        if len(source_tokens.intersection(label_tokens)) >= min_emb_overlap:
            return label
    return None


def _reasoning_strategy() -> str:
    value = str(get_policy("reasoning_strategy")).strip().lower()
    if value in {"llm_first", "kg_first"}:
        return value
    return "llm_first"


def _ground_llm_target_to_kb(
    raw_target: str,
    kg_loader: KnowledgeLoader,
    kb_pool: List[str],
    rank_query: str = "",
) -> Optional[str]:
    """
    Map a free-form LLM target to a knowledge-base label after LLM-first reasoning.
    """
    target = _normalize_text(raw_target)
    if not target or not kb_pool:
        return None
    exact = _select_grounded_target(target, kb_pool)
    if exact:
        return exact
    node = kg_loader.find_node(target)
    if node and node.label in kb_pool:
        return node.label
    query = f"{target} {rank_query}".strip()
    ranked = kg_loader.rank_candidates_by_embedding(query, kb_pool)
    if not isinstance(ranked, list) or not ranked:
        return None
    matched = _select_grounded_target(target, ranked)
    if matched:
        return matched
    return ranked[0] if ranked else None


def _select_grounded_target(raw_target: Any, candidates: List[str]) -> Optional[str]:
    """
    Map LLM target output to one of grounded candidates.
    Accept exact/normalized matches and simple containment.
    """
    if not candidates:
        return None
    target = _normalize_text(raw_target)
    if not target:
        return None
    target_key = _normalize_key(target)
    if not target_key:
        return None
    for c in candidates:
        if _normalize_key(c) == target_key:
            return c
    for c in candidates:
        c_key = _normalize_key(c)
        if c_key and (c_key in target_key or target_key in c_key):
            return c
    return None


def _normalize_reasoning_result(
    reasoning_result: Dict[str, Any],
    candidates: List[str],
    original_label: str,
    used_targets: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Enforce grounded and actionable decisions for downstream realization.
    """
    action_raw = _normalize_key(reasoning_result.get("action"))
    action = "transform" if action_raw == "transform" else "preserve"
    try:
        confidence = float(reasoning_result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    rationale = _normalize_text(reasoning_result.get("rationale")) or "No rationale provided."

    ranked_candidates = _prioritize_unused_candidates(candidates, used_targets)
    target_object = _normalize_text(reasoning_result.get("target_object"))
    grounded_target = _select_target_with_diversity(ranked_candidates, target_object, used_targets)
    if action == "transform":
        if grounded_target:
            target_object = grounded_target
        elif ranked_candidates:
            # Keep realization actionable and KG-grounded.
            target_object = ranked_candidates[0]
            rationale = f"{rationale} Grounded to top KB candidate for reliability."
            confidence = max(confidence, 0.55)
        elif not target_object:
            action = "preserve"
            target_object = original_label
    else:
        target_object = original_label

    return {
        "action": action,
        "target_object": target_object,
        "rationale": rationale,
        "confidence": confidence,
    }


def _enforce_candidate_constrained_target(
    action: str,
    target: str,
    candidate_names: List[str],
) -> None:
    if action != "transform":
        return
    if not candidate_names:
        raise ValueError("Non-KG target")
    normalized_target = _normalize_key(target)
    candidate_keys = {_normalize_key(c) for c in candidate_names}
    if normalized_target not in candidate_keys:
        raise ValueError("Non-KG target")


def cultural_density(plan: Dict[str, Any]) -> int:
    return len(plan.get("edit_plan", []) or []) + len(plan.get("region_replace", []) or [])


def _scene_mismatch_score(scene_context: str, scene_adaptation: Dict[str, Any]) -> float:
    context = _normalize_key(scene_context)
    scene_name = _normalize_key(scene_adaptation.get("scene"))
    if not context or not scene_name:
        return 0.0
    context_tokens = set(context.split())
    scene_tokens = set(scene_name.split())
    overlap = len(context_tokens.intersection(scene_tokens))
    union = max(1, len(context_tokens.union(scene_tokens)))
    mismatch = 1.0 - (overlap / union)
    return max(0.0, min(1.0, mismatch))


def _needs_scene_override(
    scene_context: str,
    scene_adaptation: Dict[str, Any],
    transformations: List[Transformation],
) -> bool:
    if not transformations:
        return False
    mismatch = _scene_mismatch_score(scene_context, scene_adaptation)
    return mismatch >= 0.85


def _coerce_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _infer_scene_canvas_size(scene_graph: Dict[str, Any]) -> Tuple[int, int]:
    """Infer canvas dimensions when Stage 1 does not emit explicit width/height."""
    metadata = scene_graph.get("metadata") if isinstance(scene_graph.get("metadata"), dict) else {}
    width = _coerce_positive_int(scene_graph.get("width") or metadata.get("width"))
    height = _coerce_positive_int(scene_graph.get("height") or metadata.get("height"))
    if width and height:
        return width, height

    max_x = 0.0
    max_y = 0.0

    def collect_bbox(bbox: Any) -> None:
        nonlocal max_x, max_y
        if isinstance(bbox, list) and len(bbox) >= 4:
            try:
                max_x = max(max_x, float(bbox[0]), float(bbox[2]))
                max_y = max(max_y, float(bbox[1]), float(bbox[3]))
            except (TypeError, ValueError):
                return

    def collect_polygon(polygon: Any) -> None:
        nonlocal max_x, max_y
        if not isinstance(polygon, list):
            return
        for point in polygon:
            if isinstance(point, list) and len(point) >= 2:
                try:
                    max_x = max(max_x, float(point[0]))
                    max_y = max(max_y, float(point[1]))
                except (TypeError, ValueError):
                    continue

    for obj in scene_graph.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        collect_bbox(obj.get("bbox"))
        segmentation = obj.get("segmentation")
        if isinstance(segmentation, dict):
            collect_polygon(segmentation.get("polygon"))

    text = scene_graph.get("text") if isinstance(scene_graph.get("text"), dict) else {}
    for collection_name in ("regions", "extracted"):
        for item in text.get(collection_name) or []:
            if not isinstance(item, dict):
                continue
            collect_bbox(item.get("bbox"))
            collect_polygon(item.get("polygon"))

    inferred_width = _coerce_positive_int(max_x)
    inferred_height = _coerce_positive_int(max_y)
    return inferred_width or 1024, inferred_height or 1024


def _build_scene_override_target(target_culture: str, scene_adaptation: Optional[Dict[str, Any]] = None) -> str:
    scene_adaptation = scene_adaptation or {}
    scene_name = str(scene_adaptation.get("scene") or "").strip()
    elements = [
        str(value).strip()
        for value in (scene_adaptation.get("elements") or [])
        if str(value).strip()
    ]
    details = ", ".join(elements[:3])
    if scene_name and details:
        return f"{target_culture} {scene_name} cultural infographic with {details}"
    if scene_name:
        return f"{target_culture} {scene_name} cultural infographic"
    return f"{target_culture} cultural infographic"


def _build_scene_override_region(
    scene_graph: Dict[str, Any],
    target_culture: str,
    scene_adaptation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    width, height = _infer_scene_canvas_size(scene_graph)
    target = _build_scene_override_target(target_culture, scene_adaptation)
    return {
        "object_id": 900000,
        "original": "scene region",
        "new": target,
        "target": target,
        "bbox": [0, 0, width, height],
        "source": "scene_override",
        "visual_attributes": {
            "shape": "full frame composition",
            "color": f"{target_culture} culturally consistent palette",
            "texture": "environment-level realistic texture",
            "context": target,
        },
    }


def _build_scene_level_transformation(
    target_culture: str,
    scene_adaptation: Dict[str, Any],
    reason: str,
) -> Transformation:
    target = f"{target_culture} cultural environment"
    return Transformation(
        original_object="scene region",
        original_type="SCENE",
        target_object=target,
        rationale=reason,
        confidence=0.6,
        visual_attributes={
            "shape": "full frame composition",
            "color": "culturally consistent palette",
            "texture": "environment-level realistic texture",
            "context": str(scene_adaptation.get("scene") or target),
        },
    )


def _allows_full_scene_override(scene_graph: Dict[str, Any]) -> bool:
    """
    Full-frame scene overrides are safe mostly for layout-heavy content where the
    whole canvas is expected to be redesigned. For natural-photo inputs, applying
    a full-frame inpaint often produces soft/blurred outputs.
    """
    image_type = ((scene_graph.get("image_type") or {}).get("type") or "").strip().lower()
    return image_type in _SCENE_OVERRIDE_ALLOWED_IMAGE_TYPES and _is_infographic_mode(scene_graph)


def _enforce_multi_object_coordination(
    transformations: List[Transformation],
    scene_adaptation: Dict[str, Any],
) -> List[Transformation]:
    if len(transformations) <= 1:
        return transformations
    scene_text = " ".join(
        [
            str(scene_adaptation.get("scene") or ""),
            " ".join(str(x) for x in (scene_adaptation.get("elements") or [])),
        ]
    ).lower()
    coordinated: List[Transformation] = []
    for t in transformations:
        attrs = dict(t.visual_attributes or {})
        context_text = str(attrs.get("context") or "").lower()
        if "western" in context_text and "western" not in scene_text:
            attrs["context"] = f"{scene_adaptation.get('scene', 'local cultural setting')} context"
        coordinated.append(
            Transformation(
                original_object=t.original_object,
                original_type=t.original_type,
                target_object=t.target_object,
                rationale=t.rationale,
                confidence=t.confidence,
                visual_attributes=attrs,
            )
        )
    return coordinated


def _build_culture_consistency_graph(
    target_culture: str,
    transformations: List[Transformation],
    region_replace: List[Dict[str, Any]],
) -> Dict[str, Any]:
    nodes = []
    edges = []
    for t in transformations:
        node_id = f"obj:{_normalize_key(t.target_object)}"
        nodes.append(
            {
                "id": node_id,
                "label": t.target_object,
                "culture": target_culture,
                "category": t.original_type,
            }
        )
    for r in region_replace:
        target = str(r.get("target") or r.get("new") or "").strip()
        if not target:
            continue
        node_id = f"region:{_normalize_key(target)}"
        nodes.append(
            {
                "id": node_id,
                "label": target,
                "culture": target_culture,
                "category": "REGION",
            }
        )
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            edges.append(
                {
                    "source": nodes[i]["id"],
                    "target": nodes[j]["id"],
                    "relation": "same_target_culture",
                }
            )
    return {"nodes": nodes, "edges": edges}


def _build_forced_cultural_transformation(
    scene_objects: List[Dict[str, Any]],
    target_culture: str,
    kg_loader: KnowledgeLoader,
) -> Optional[Transformation]:
    if not scene_objects:
        return None
    anchor = max(
        [obj for obj in scene_objects if isinstance(obj, dict)],
        key=lambda o: float(o.get("confidence", 0.0) or 0.0),
        default=None,
    )
    if not anchor:
        return None
    original = str(anchor.get("label") or anchor.get("class_name") or "").strip()
    if not original:
        return None
    label_to_type = kg_loader.get_label_to_type()
    inferred_type = _infer_cultural_type(original, anchor, label_to_type)
    available_types = [
        str(t).upper()
        for t in (kg_loader.get_cultural_types() or set())
        if str(t).upper() != "COUNTRY"
    ]
    # Rank types dynamically by target-culture availability in KG.
    scored_types: List[Tuple[int, str]] = []
    for t in available_types:
        count = len(kg_loader.get_nodes_by_type_and_culture(t, target_culture) or [])
        if count > 0:
            scored_types.append((count, t))
    scored_types.sort(key=lambda x: x[0], reverse=True)
    fallback_types: List[str] = [t for _, t in scored_types]
    # If we can infer source type, prefer it first when present.
    if inferred_type:
        inferred_upper = str(inferred_type).upper()
        if inferred_upper in fallback_types:
            fallback_types = [inferred_upper] + [t for t in fallback_types if t != inferred_upper]
        elif inferred_upper in available_types:
            fallback_types = [inferred_upper] + fallback_types
    # By default keep forced fallback type-safe. Cross-type fallback can be enabled
    # only via env when broader coverage is preferred over semantic fidelity.
    if not ALLOW_CROSS_TYPE_FORCED_FALLBACK:
        if inferred_type:
            inferred_upper = str(inferred_type).upper()
            fallback_types = [t for t in fallback_types if t == inferred_upper]
        else:
            logger.info(
                "Skipping forced fallback for '%s': inferred type unavailable and cross-type fallback disabled.",
                original,
            )
            return None
    target_label = None
    chosen_type = "CULTURAL"
    for t in fallback_types:
        nodes = kg_loader.get_nodes_by_type_and_culture(t, target_culture) or []
        if nodes:
            target_label = str(nodes[0].label)
            chosen_type = t
            break
    if not target_label:
        logger.info(
            "Skipping forced fallback for '%s': no target found for allowed types=%s.",
            original,
            fallback_types,
        )
        return None
    visual_attributes = kg_loader.get_visual_attributes(
        label=target_label,
        obj_type=chosen_type,
        culture_name=target_culture,
    )
    if not isinstance(visual_attributes, dict):
        visual_attributes = {
            "shape": "recognizable local form",
            "color": "locally appropriate colors",
            "texture": "natural texture",
            "context": f"placed in {target_culture} local context",
        }
    return Transformation(
        original_object=original,
        original_type=chosen_type,
        target_object=target_label,
        rationale=(
            "Forced cultural fallback applied because no grounded object-level transform "
            "was produced and transcreation coverage is required."
        ),
        confidence=0.51,
        visual_attributes=visual_attributes,
    )


def _infer_food_term_from_text(
    full_text: str,
    label_to_type: Optional[Dict[str, str]] = None,
    kb_entry: Optional[Any] = None,
) -> Optional[str]:
    """
    Dynamically infer source food term from OCR/scene text using KB mappings.
    No hardcoded object names.
    """
    text = (full_text or "").lower()
    if not text:
        return None

    food_terms: Set[str] = set()
    for label, obj_type in (label_to_type or {}).items():
        if str(obj_type).upper() == "FOOD" and isinstance(label, str) and label.strip():
            food_terms.add(label.strip().lower())

    if kb_entry is not None:
        try:
            subs = (kb_entry.substitutions or {}).get("FOOD", [])
            for se in subs:
                src = getattr(se, "source", None)
                if isinstance(src, str) and src.strip():
                    food_terms.add(src.strip().lower())
        except Exception:
            pass

    if not food_terms:
        return None

    # Prefer the most specific term first (multi-word/longer labels).
    for term in sorted(food_terms, key=lambda t: len(t), reverse=True):
        pattern = r"\b" + re.escape(term) + r"s?\b"
        if re.search(pattern, text):
            return term
    return None


def _extract_food_terms_from_text(
    full_text: str,
    label_to_type: Optional[Dict[str, str]] = None,
    kb_entry: Optional[Any] = None,
) -> List[str]:
    """
    Return all FOOD terms found in text using dynamic KG mappings.
    """
    text = (full_text or "").lower()
    if not text:
        return []
    food_terms: Set[str] = set()
    for label, obj_type in (label_to_type or {}).items():
        if str(obj_type).upper() == "FOOD" and isinstance(label, str) and label.strip():
            food_terms.add(label.strip().lower())
    if kb_entry is not None:
        try:
            subs = (kb_entry.substitutions or {}).get("FOOD", [])
            for se in subs:
                src = getattr(se, "source", None)
                if isinstance(src, str) and src.strip():
                    food_terms.add(src.strip().lower())
        except Exception:
            pass

    hits: List[str] = []
    for term in sorted(food_terms, key=lambda t: len(t), reverse=True):
        pattern = r"\b" + re.escape(term) + r"s?\b"
        if re.search(pattern, text):
            hits.append(term)
    return hits


def _infer_row_label_bboxes(extracted: List[Dict[str, Any]]) -> List[List[int]]:
    """
    Infer row-label anchors using OCR geometry only (no hardcoded token list).
    """
    boxes: List[List[int]] = []
    for item in extracted or []:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
    if len(boxes) < 2:
        return []

    x1_vals = [b[0] for b in boxes]
    x1_min, x1_max = min(x1_vals), max(x1_vals)
    x_span = max(1, x1_max - x1_min)
    left_band_max = x1_min + int(0.35 * x_span)
    max_width = max((b[2] - b[0]) for b in boxes)

    candidates: List[List[int]] = []
    for b in boxes:
        x1, y1, x2, y2 = b
        w = x2 - x1
        h = y2 - y1
        if x1 > left_band_max:
            continue
        if h < 10:
            continue
        if w > int(0.75 * max(1, max_width)):
            continue
        candidates.append(b)

    candidates.sort(key=lambda b: (b[1], b[0]))
    merged: List[List[int]] = []
    for b in candidates:
        if not merged:
            merged.append(b)
            continue
        prev = merged[-1]
        if abs(b[1] - prev[1]) <= 6:
            prev_w = prev[2] - prev[0]
            cur_w = b[2] - b[0]
            if (b[0], -cur_w) < (prev[0], -prev_w):
                merged[-1] = b
            continue
        merged.append(b)
    return merged


def _is_infographic_mode(scene_graph: Dict[str, Any]) -> bool:
    image_info = scene_graph.get("image_type") if isinstance(scene_graph.get("image_type"), dict) else {}
    image_type = ((image_info or {}).get("type") or "").lower()
    confidence = float((image_info or {}).get("confidence", 1.0) if "confidence" in (image_info or {}) else 1.0)
    quality_flags = set((image_info or {}).get("quality_flags") or [])
    extracted = ((scene_graph.get("text") or {}).get("extracted") or [])
    text_count = len(extracted) if isinstance(extracted, list) else 0
    analysis = scene_graph.get("infographic_analysis") or {}
    icon_cluster_count = int(analysis.get("icon_cluster_count", 0) or 0)
    # Keep strict safeguards for true infographic/document layouts.
    # Posters frequently contain natural-image objects and should stay transformable
    # unless stronger infographic signals are present.
    if image_type == "infographic":
        return confidence >= 0.5 or text_count >= 6 or icon_cluster_count >= 3
    if image_type == "document":
        return text_count >= 6 and "low_confidence_image_type" not in quality_flags
    if image_type == "ui":
        return True
    if analysis.get("enabled") and icon_cluster_count >= 3:
        return True
    return text_count >= 6


def _is_ambiguous_person_in_infographic(
    infographic_mode: bool, obj_label: str, obj: Dict[str, Any], scene_context: str
) -> bool:
    if not infographic_mode:
        return False
    if (obj_label or "").lower() not in {"person", "people", "man", "woman", "child"}:
        return False
    confidence = float(obj.get("confidence", 0.0) or 0.0)
    min_conf = 0.72
    has_context = any(token in scene_context.lower() for token in ["portrait", "photo", "real person"])
    return confidence < min_conf or not has_context


def _should_preserve_non_text_in_infographic(
    infographic_mode: bool, obj_type: str, obj: Dict[str, Any]
) -> bool:
    """
    In infographic mode, preserve unknown/non-cultural objects to avoid semantic drift,
    but allow high-confidence, culturally meaningful icon-like objects.
    """
    if not infographic_mode:
        return False

    obj_type_upper = (obj_type or "").upper()
    if obj_type_upper in {"TEXT", "SYMBOL", "ART"}:
        return False

    semantic_type = (obj.get("semantic_type") or "").lower()
    confidence = float(obj.get("confidence", 0.0) or 0.0)
    has_icon_anchor = semantic_type in {"icon", "symbol", "logo"} or obj.get("icon_cluster_id") is not None
    label_text = str(obj.get("label") or obj.get("class_name") or "").lower()
    has_cultural_cue = any(token in label_text for token in INFOGRAPHIC_CULTURAL_CUE_TOKENS)

    if has_cultural_cue:
        return False

    if obj_type_upper in {"FOOD", "SPORT", "CLOTHING"} and has_icon_anchor and confidence >= 0.65:
        return False

    return True


class CulturalReasoningEngine:
    """
    Stage 2: Cultural Reasoning with a grounded knowledge base.
    Given S(Is) (scene graph) and K(ct) (cultural KB), selects candidate edits E
    under preservation constraints via: identify misaligned elements, retrieve
    candidates from KB, filter by avoid lists and context, decide preservations,
    output structured plan.
    """
    def __init__(
        self,
        knowledge_graph_path: str,
        debug_plan: bool = False,
        debug_kg_selection: bool = False,
        strict_mode: bool = False,
    ):
        self.kg_loader = KnowledgeLoader(knowledge_graph_path)
        self.llm_client = LLMClient()
        self.debug_plan = debug_plan
        self.debug_kg_selection = debug_kg_selection
        self.strict_mode = strict_mode
        self._type_token_index = _build_type_token_index(self.kg_loader)
        self._debug_trace: Dict[str, Any] = {
            "raw_plan": [],
            "normalized_plan": [],
            "kg_selections": [],
        }

    def _collect_kb_candidate_pool(
        self,
        *,
        target_culture: str,
        source_obj_label: str,
        obj_type: str,
        obj_label: str,
        grounded_hint: Optional[str],
        avoid_list: List[str],
        obj: Dict[str, Any],
        scene_context: str,
        used_targets: Set[str],
    ) -> Tuple[List[str], List[str]]:
        candidate_labels = self.kg_loader.get_candidates_from_kb(
            target_culture, source_obj_label, obj_type
        )
        if grounded_hint and grounded_hint != source_obj_label:
            hint_candidates = self.kg_loader.get_candidates_from_kb(
                target_culture, grounded_hint, obj_type
            )
            for label in hint_candidates:
                if label not in candidate_labels:
                    candidate_labels.append(label)
        if not candidate_labels:
            nodes = self.kg_loader.get_nodes_by_type_and_culture(obj_type, target_culture)
            candidate_labels = [c.label for c in nodes]
        preferred = self.kg_loader.get_preferred_substitution(obj_label, target_culture)
        if preferred and preferred in candidate_labels:
            candidate_labels = [preferred] + [c for c in candidate_labels if c != preferred]
        candidate_labels, notes = _filter_candidates_by_avoid(candidate_labels, avoid_list)
        if candidate_labels:
            rank_query = f"{_object_signal_text(obj, obj_label)} {obj_type} {scene_context}".strip()
            ranked = self.kg_loader.rank_candidates_by_embedding(rank_query, candidate_labels)
            if isinstance(ranked, list) and all(isinstance(x, str) for x in ranked):
                candidate_labels = ranked
        candidate_labels = _prioritize_unused_candidates(candidate_labels, used_targets)
        return candidate_labels, notes

    def _run_kg_first_object_reasoning(
        self,
        *,
        obj: Dict[str, Any],
        obj_label: str,
        source_obj_label: str,
        obj_type: str,
        source_culture: str,
        target_culture: str,
        grounded_hint: Optional[str],
        avoid_list: List[str],
        scene_context: str,
        used_targets: Set[str],
        has_local_edit_region: bool,
        context: str,
        style_priors: Optional[StylePriors],
        sensitivity_notes: List[str],
    ) -> Tuple[Optional[Dict[str, Any]], List[str], List[str]]:
        candidate_labels, notes = self._collect_kb_candidate_pool(
            target_culture=target_culture,
            source_obj_label=source_obj_label,
            obj_type=obj_type,
            obj_label=obj_label,
            grounded_hint=grounded_hint,
            avoid_list=avoid_list,
            obj=obj,
            scene_context=scene_context,
            used_targets=used_targets,
        )
        if candidate_labels:
            logger.info(
                "KB candidates found: label=%s, type=%s, count=%d, top=%s",
                obj_label,
                obj_type,
                len(candidate_labels),
                candidate_labels[0],
            )
        if not candidate_labels and (STRICT_KB_GROUNDED_TRANSFORMS or not has_local_edit_region):
            logger.info(
                "Reasoning preserve: %s (no KB candidates for type=%s, KB-first policy)",
                obj_label,
                obj_type,
            )
            return None, [], notes
        if not candidate_labels:
            logger.info(
                "No KB candidates for label=%s, type=%s. Requesting LLM candidate generation.",
                obj_label,
                obj_type,
            )
            candidate_labels = self.llm_client.generate_candidates(
                obj_label=obj_label,
                obj_type=obj_type,
                target_culture=target_culture,
                context=context,
                avoid_list=avoid_list,
            )
            if not isinstance(candidate_labels, list):
                candidate_labels = []
        logger.info(
            "Candidates after avoid filter: label=%s, count=%d",
            obj_label,
            len(candidate_labels),
        )
        prompt = self._construct_prompt(
            obj_label=obj_label,
            obj_type=obj_type,
            source_culture=source_culture,
            target_culture=target_culture,
            candidates=candidate_labels,
            context=context,
            avoid_list=avoid_list,
            style_priors=style_priors,
            sensitivity_notes=sensitivity_notes,
            used_targets=used_targets,
            reasoning_strategy="kg_first",
        )
        reasoning_result = self.llm_client.generate_reasoning(prompt)
        if self.debug_plan:
            self._debug_trace["raw_plan"].append({"object": obj_label, "raw_result": reasoning_result})
        reasoning_result = _normalize_reasoning_result(
            reasoning_result=reasoning_result,
            candidates=candidate_labels,
            original_label=obj_label,
            used_targets=used_targets,
        )
        if self.strict_mode:
            _enforce_candidate_constrained_target(
                action=str(reasoning_result.get("action", "")),
                target=str(reasoning_result.get("target_object", "")),
                candidate_names=candidate_labels,
            )
        if self.debug_plan:
            self._debug_trace["normalized_plan"].append(
                {"object": obj_label, "normalized_result": reasoning_result}
            )
        return reasoning_result, candidate_labels, notes

    def _run_llm_first_object_reasoning(
        self,
        *,
        obj: Dict[str, Any],
        obj_label: str,
        source_obj_label: str,
        obj_type: str,
        source_culture: str,
        target_culture: str,
        grounded_hint: Optional[str],
        avoid_list: List[str],
        scene_context: str,
        used_targets: Set[str],
        has_local_edit_region: bool,
        context: str,
        style_priors: Optional[StylePriors],
        sensitivity_notes: List[str],
    ) -> Tuple[Optional[Dict[str, Any]], List[str], List[str]]:
        candidate_labels, notes = self._collect_kb_candidate_pool(
            target_culture=target_culture,
            source_obj_label=source_obj_label,
            obj_type=obj_type,
            obj_label=obj_label,
            grounded_hint=grounded_hint,
            avoid_list=avoid_list,
            obj=obj,
            scene_context=scene_context,
            used_targets=used_targets,
        )
        if not candidate_labels and (STRICT_KB_GROUNDED_TRANSFORMS or not has_local_edit_region):
            logger.info(
                "Reasoning preserve: %s (no KB pool for grounding, LLM-first policy)",
                obj_label,
            )
            return None, [], notes
        prompt = self._construct_prompt(
            obj_label=obj_label,
            obj_type=obj_type,
            source_culture=source_culture,
            target_culture=target_culture,
            candidates=[],
            context=context,
            avoid_list=avoid_list,
            style_priors=style_priors,
            sensitivity_notes=sensitivity_notes or [],
            used_targets=used_targets,
            reasoning_strategy="llm_first",
        )
        raw_result = self.llm_client.generate_reasoning(prompt)
        if self.debug_plan:
            self._debug_trace["raw_plan"].append({"object": obj_label, "raw_result": raw_result})
        llm_target = _normalize_text(raw_result.get("target_object"))
        action_raw = _normalize_key(raw_result.get("action"))
        if action_raw == "transform" and llm_target:
            rank_query = f"{_object_signal_text(obj, obj_label)} {obj_type} {scene_context}".strip()
            grounded = _ground_llm_target_to_kb(
                llm_target,
                self.kg_loader,
                candidate_labels,
                rank_query=rank_query,
            )
            if grounded:
                logger.info(
                    "LLM-first KB grounding: label=%s, llm_target=%s, grounded=%s",
                    obj_label,
                    llm_target,
                    grounded,
                )
                raw_result["target_object"] = grounded
            elif candidate_labels:
                raw_result["target_object"] = candidate_labels[0]
                note = " KB catalog nearest match applied after LLM suggestion."
                raw_result["rationale"] = (_normalize_text(raw_result.get("rationale")) or "") + note
                logger.info(
                    "LLM-first KB fallback: label=%s, llm_target=%s, catalog=%s",
                    obj_label,
                    llm_target,
                    candidate_labels[0],
                )
            else:
                raw_result["action"] = "preserve"
                raw_result["target_object"] = obj_label
                logger.info(
                    "LLM-first preserve: label=%s (LLM target '%s' not in KB catalog)",
                    obj_label,
                    llm_target,
                )
        reasoning_result = _normalize_reasoning_result(
            reasoning_result=raw_result,
            candidates=candidate_labels,
            original_label=obj_label,
            used_targets=used_targets,
        )
        if self.strict_mode and candidate_labels:
            _enforce_candidate_constrained_target(
                action=str(reasoning_result.get("action", "")),
                target=str(reasoning_result.get("target_object", "")),
                candidate_names=candidate_labels,
            )
        if self.debug_plan:
            self._debug_trace["normalized_plan"].append(
                {"object": obj_label, "normalized_result": reasoning_result}
            )
        return reasoning_result, candidate_labels, notes

    def analyze_image(self, input_data: ReasoningInput) -> TranscreationPlan:
        logger.info("Starting analysis for target culture: %s", input_data.target_culture)
        self._debug_trace = {
            "raw_plan": [],
            "normalized_plan": [],
            "kg_selections": [],
        }
        scene_objects = input_data.scene_graph.get("objects", [])
        target_culture = input_data.target_culture
        transformations: List[Transformation] = []
        preservations: List[Preservation] = []
        avoidance_adherence: List[str] = []
        infographic_mode = _is_infographic_mode(input_data.scene_graph)
        scene_context = input_data.scene_graph.get("scene", {}).get("description", "")
        scene_candidates = self.kg_loader.get_scene_candidates(target_culture)
        scene_adaptation = scene_candidates[0] if scene_candidates else {}
        if not scene_adaptation and self.strict_mode:
            raise ValueError("Missing scene")
        scene_adaptation = {
            "scene": scene_adaptation.get("name", f"{target_culture} local scene"),
            "elements": scene_adaptation.get("elements", ["local details"]),
            "lighting": scene_adaptation.get("lighting", "natural"),
            "style": scene_adaptation.get("style", "vibrant"),
        }

        # Resolve avoid list: KB avoid + CLI override
        kb_avoid = self.kg_loader.get_avoid_list(target_culture)
        avoid_list = list(kb_avoid) if kb_avoid else []
        for a in input_data.avoid_list:
            if a and a not in avoid_list:
                avoid_list.append(a)

        used_targets: Set[str] = set()
        for obj in scene_objects:
            source_obj_label = obj.get("label") or obj.get("class_name")
            if not source_obj_label:
                continue
            obj_label = source_obj_label
            source_has_kg_match = self.kg_loader.find_node(source_obj_label) is not None
            needs_grounding = (not source_has_kg_match) and (
                "detector_caption_mismatch" in (obj.get("quality_flags") or [])
                or "uncertain_label" in (obj.get("quality_flags") or [])
                or str(obj.get("semantic_type") or "").lower() in {"icon", "symbol", "logo"}
            )
            has_local_edit_region = isinstance(obj.get("bbox"), list) and len(obj.get("bbox") or []) >= 4
            obj_type, source_culture = _resolve_obj_type_for_localized_edit(
                obj=obj,
                source_label=source_obj_label,
                kg_loader=self.kg_loader,
                type_token_index=self._type_token_index,
            )
            grounded_hint = None
            if needs_grounding:
                allowed_types = {obj_type} if obj_type and obj_type not in {"object", ""} else None
                grounded_hint = _recover_grounded_label(
                    obj,
                    self.kg_loader,
                    exclude_scope_types=has_local_edit_region,
                    allowed_types=allowed_types,
                )
                if grounded_hint and _normalize_key(grounded_hint) != _normalize_key(source_obj_label):
                    logger.info(
                        "Grounding hint for source=%s (type=%s): %s",
                        source_obj_label,
                        obj_type,
                        grounded_hint,
                    )
            logger.info(
                "Reasoning object start: label=%s, type=%s, confidence=%s",
                obj_label,
                obj_type,
                obj.get("confidence"),
            )
            if _is_ambiguous_person_in_infographic(infographic_mode, obj_label, obj, scene_context):
                preservations.append(Preservation(
                    original_object=obj_label,
                    rationale="Infographic safety policy: ambiguous person detection preserved.",
                ))
                logger.info("Reasoning preserve: %s (ambiguous infographic person policy)", obj_label)
                continue

            if _should_preserve_non_text_in_infographic(infographic_mode, obj_type, obj):
                preservations.append(Preservation(
                    original_object=obj_label,
                    rationale="Infographic mode: preserving COCO-style object to avoid semantic drift.",
                ))
                logger.info("Reasoning preserve: %s (infographic COCO-style policy)", obj_label)
                continue

            context = input_data.scene_graph.get("scene", {}).get("description", "")
            style_priors = self.kg_loader.get_style_priors(target_culture)
            sensitivity_notes = self.kg_loader.get_sensitivity_notes(target_culture) or []
            strategy = _reasoning_strategy()
            logger.info("Reasoning strategy: %s for label=%s", strategy, obj_label)

            if strategy == "llm_first":
                reasoning_result, candidate_labels, avoid_notes = self._run_llm_first_object_reasoning(
                    obj=obj,
                    obj_label=obj_label,
                    source_obj_label=source_obj_label,
                    obj_type=obj_type,
                    source_culture=source_culture,
                    target_culture=target_culture,
                    grounded_hint=grounded_hint,
                    avoid_list=avoid_list,
                    scene_context=scene_context,
                    used_targets=used_targets,
                    has_local_edit_region=has_local_edit_region,
                    context=context,
                    style_priors=style_priors,
                    sensitivity_notes=sensitivity_notes,
                )
            else:
                reasoning_result, candidate_labels, avoid_notes = self._run_kg_first_object_reasoning(
                    obj=obj,
                    obj_label=obj_label,
                    source_obj_label=source_obj_label,
                    obj_type=obj_type,
                    source_culture=source_culture,
                    target_culture=target_culture,
                    grounded_hint=grounded_hint,
                    avoid_list=avoid_list,
                    scene_context=scene_context,
                    used_targets=used_targets,
                    has_local_edit_region=has_local_edit_region,
                    context=context,
                    style_priors=style_priors,
                    sensitivity_notes=sensitivity_notes,
                )

            if reasoning_result is None:
                preservations.append(Preservation(
                    original_object=source_obj_label,
                    rationale="No grounded KB catalog entries; preserved by reasoning policy.",
                ))
                continue
            avoidance_adherence.extend(avoid_notes)

            logger.info(
                "LLM reasoning result (%s): label=%s, action=%s, target=%s, confidence=%s",
                strategy,
                obj_label,
                reasoning_result.get("action"),
                reasoning_result.get("target_object"),
                reasoning_result.get("confidence"),
            )

            # (5) Output structured plan (transformation or preservation)
            action = reasoning_result.get("action")
            if action == "transform":
                selected_target = str(reasoning_result.get("target_object", "Unknown"))
                visual_attributes = self.kg_loader.get_visual_attributes(
                    label=selected_target,
                    obj_type=obj_type,
                    culture_name=target_culture,
                )
                if not isinstance(visual_attributes, dict):
                    visual_attributes = {
                        "shape": "recognizable local form",
                        "color": "locally appropriate colors",
                        "texture": "natural texture",
                        "context": f"placed in {target_culture} local context",
                    }
                transformations.append(Transformation(
                    original_object=source_obj_label,
                    original_type=obj_type,
                    target_object=selected_target,
                    rationale=reasoning_result.get("rationale", "No rationale provided."),
                    confidence=float(reasoning_result.get("confidence", 0.0)),
                    visual_attributes=visual_attributes,
                ))
                used_targets.add(selected_target)
                logger.info(
                    "Reasoning transform: %s (%s) -> %s",
                    obj_label,
                    obj_type,
                    reasoning_result.get("target_object", "Unknown"),
                )
                if self.debug_kg_selection:
                    selected_target = reasoning_result.get("target_object", "Unknown")
                    selected_node = self.kg_loader.find_node(selected_target)
                    self._debug_trace["kg_selections"].append(
                        {
                            "source_object": obj_label,
                            "selected_target_name": selected_target,
                            "selected_target_id": selected_node.id if selected_node else None,
                            "visual_attributes": visual_attributes,
                        }
                    )
            elif candidate_labels and (infographic_mode or self.strict_mode):
                # Deterministic fallback: if grounded candidates exist but model preserves,
                # select top grounded candidate so downstream realization has actionable edits.
                fallback_target = _select_target_with_diversity(
                    candidate_labels,
                    candidate_labels[0],
                    used_targets,
                ) or candidate_labels[0]
                fallback_visual_attributes = self.kg_loader.get_visual_attributes(
                    label=fallback_target,
                    obj_type=obj_type,
                    culture_name=target_culture,
                )
                if not isinstance(fallback_visual_attributes, dict):
                    fallback_visual_attributes = {
                        "shape": "recognizable local form",
                        "color": "locally appropriate colors",
                        "texture": "natural texture",
                        "context": f"placed in {target_culture} local context",
                    }
                transformations.append(Transformation(
                    original_object=source_obj_label,
                    original_type=obj_type,
                    target_object=fallback_target,
                    rationale="Grounded fallback: candidates available in KB; selected top candidate.",
                    confidence=0.55,
                    visual_attributes=fallback_visual_attributes,
                ))
                used_targets.add(fallback_target)
                logger.info(
                    "Applied grounded fallback transform for '%s' -> '%s' (type=%s)",
                    obj_label,
                    fallback_target,
                    obj_type,
                )
                if self.debug_kg_selection:
                    selected_node = self.kg_loader.find_node(fallback_target)
                    self._debug_trace["kg_selections"].append(
                        {
                            "source_object": obj_label,
                            "selected_target_name": fallback_target,
                            "selected_target_id": selected_node.id if selected_node else None,
                        }
                    )
            else:
                rationale = reasoning_result.get("rationale", "Preserved by default.")
                if isinstance(rationale, str) and "LLM Service Unavailable" in rationale:
                    rationale = "No grounded candidates in KB for this object; preserved."
                preservations.append(Preservation(
                    original_object=source_obj_label,
                    rationale=rationale,
                ))
                logger.info("Reasoning preserve: %s (rationale=%s)", obj_label, rationale)

        logger.info(
            "Reasoning complete: transformations=%d, preservations=%d, avoid_notes=%d",
            len(transformations),
            len(preservations),
            len(avoidance_adherence),
        )
        transformations = _dedupe_transformations_by_original(transformations)
        transformations = _enforce_multi_object_coordination(transformations, scene_adaptation)
        dynamic_min_density = MIN_CULTURAL_DENSITY if len(scene_objects) > 1 else 1
        density = len(transformations)
        if self.strict_mode and density < dynamic_min_density:
            forced = _build_forced_cultural_transformation(
                scene_objects=scene_objects,
                target_culture=target_culture,
                kg_loader=self.kg_loader,
            )
            if forced is not None:
                transformations.append(forced)
                density = len(transformations)
                logger.warning(
                    "Cultural density was low (%d). Applied forced fallback transformation: %s -> %s",
                    density - 1,
                    forced.original_object,
                    forced.target_object,
                )
        region_replace = self.build_region_replacements(input_data)
        # Prefer localized object/region edits when we already have actionable object transforms.
        # Full-scene override is a last resort because it can dilute bbox-level realization quality.
        has_object_transforms = len(transformations) > 0
        if (not has_object_transforms) and _needs_scene_override(scene_context, scene_adaptation, transformations):
            if _allows_full_scene_override(input_data.scene_graph):
                logger.info("Scene override activated due to strong mismatch.")
                scene_adaptation["override_mode"] = "full_region_regenerate"
                scene_adaptation["override_reason"] = "strong scene-culture mismatch"
                region_replace.append(
                    _build_scene_override_region(input_data.scene_graph, target_culture, scene_adaptation)
                )
                transformations.append(
                    _build_scene_level_transformation(
                        target_culture=target_culture,
                        scene_adaptation=scene_adaptation,
                        reason="Scene-level cultural transformation required due to strong scene-culture mismatch.",
                    )
                )
            else:
                logger.info(
                    "Scene override skipped for image_type=%s to avoid full-frame blur.",
                    ((input_data.scene_graph.get("image_type") or {}).get("type") or "unknown"),
                )
                scene_adaptation["override_mode"] = "none"
                scene_adaptation["override_reason"] = "disabled_for_photo_like_image_type"
        else:
            scene_adaptation["override_mode"] = "none"
            if has_object_transforms:
                scene_adaptation["override_reason"] = "localized_object_transforms_present"
        scene_adaptation["consistency_graph"] = _build_culture_consistency_graph(
            target_culture=target_culture,
            transformations=transformations,
            region_replace=region_replace,
        )
        density += len(region_replace)
        if self.strict_mode and density < dynamic_min_density:
            if _allows_full_scene_override(input_data.scene_graph):
                logger.warning(
                    "Cultural density remains low after fallback (%d < %d). "
                    "Forcing scene override region to guarantee transcreation coverage.",
                    density,
                    dynamic_min_density,
                )
                region_replace.append(
                    _build_scene_override_region(input_data.scene_graph, target_culture, scene_adaptation)
                )
                if not transformations:
                    transformations.append(
                        _build_scene_level_transformation(
                            target_culture=target_culture,
                            scene_adaptation=scene_adaptation,
                            reason="Scene-level cultural transformation required by strict density guard.",
                        )
                    )
                scene_adaptation["override_mode"] = "full_region_regenerate"
                scene_adaptation["override_reason"] = "forced_by_density_guard"
            else:
                logger.warning(
                    "Cultural density remains low (%d < %d), but scene override is disabled for image_type=%s.",
                    density,
                    dynamic_min_density,
                    ((input_data.scene_graph.get("image_type") or {}).get("type") or "unknown"),
                )
                scene_adaptation["override_mode"] = "none"
                scene_adaptation["override_reason"] = "disabled_for_photo_like_image_type"
        return TranscreationPlan(
            target_culture=target_culture,
            transformations=transformations,
            preservations=preservations,
            avoidance_adherence=avoidance_adherence,
            scene_adaptation=scene_adaptation,
            region_replace=region_replace,
        )

    def get_debug_trace(self) -> Dict[str, Any]:
        return copy.deepcopy(self._debug_trace)

    def build_text_edits(self, input_data: ReasoningInput) -> List[Dict[str, Any]]:
        """Create edit_text actions for document/infographic-heavy images."""
        image_type = ((input_data.scene_graph.get("image_type") or {}).get("type") or "").lower()
        if image_type not in {"document", "poster", "ui", "social_media", "infographic"}:
            return []
        return _build_text_edits_for_document(
            input_data.scene_graph, input_data.target_culture, llm_client=self.llm_client
        )

    def build_region_replacements(self, input_data: ReasoningInput) -> List[Dict[str, Any]]:
        """
        Build region-level fallback replacements for infographic tables/charts when
        object detectors miss repeated visual symbols/icons.
        """
        scene_graph = input_data.scene_graph or {}
        image_type = ((scene_graph.get("image_type") or {}).get("type") or "").lower()
        if image_type not in {"infographic", "document", "poster"}:
            return []
        objects = scene_graph.get("objects") or []
        if isinstance(objects, list) and len(objects) > 0:
            return []

        extracted = ((scene_graph.get("text") or {}).get("extracted") or [])
        if not isinstance(extracted, list) or not extracted:
            return []
        full_text = ((scene_graph.get("text") or {}).get("full_text") or "")
        label_to_type = self.kg_loader.get_label_to_type()
        kb_entry = self.kg_loader.get_kb_entry(input_data.target_culture)
        source_food_terms = _extract_food_terms_from_text(
            full_text,
            label_to_type=label_to_type,
            kb_entry=kb_entry,
        )
        if not source_food_terms:
            src_food = _infer_food_term_from_text(
                full_text,
                label_to_type=label_to_type,
                kb_entry=kb_entry,
            )
            source_food_terms = [src_food] if src_food else []
        if not source_food_terms:
            return []

        src_food = source_food_terms[0]
        if len(source_food_terms) > 1:
            try:
                template = get_prompt(
                    "choose_source_food.template",
                    (
                        "Select one source food term from OCR text for region-level icon transcreation.\n"
                        "Target culture: {target_culture}\n"
                        "Scene context: {scene_context}\n"
                        "OCR full text: {ocr_text}\n"
                        "Detected source food terms: {source_food_terms}\n\n"
                        'Return JSON only: {"source_food":"one_term_from_list"}'
                    ),
                )
                prompt = template.format(
                    target_culture=input_data.target_culture,
                    scene_context=((scene_graph.get("scene") or {}).get("description") or ""),
                    ocr_text=full_text[:500],
                    source_food_terms=source_food_terms,
                )
                res = self.llm_client.generate_reasoning(prompt)
                chosen = _normalize_text(res.get("source_food"))
                if chosen and _normalize_key(chosen) in {_normalize_key(t) for t in source_food_terms}:
                    src_food = next(t for t in source_food_terms if _normalize_key(t) == _normalize_key(chosen))
            except Exception:
                pass

        target = self.kg_loader.get_preferred_substitution(src_food, input_data.target_culture)
        candidates = self.kg_loader.get_candidates_from_kb(input_data.target_culture, src_food, "FOOD")
        if not target and candidates:
            target = candidates[0]
        if candidates:
            llm_target = None
            try:
                template = get_prompt(
                    "choose_target_food.template",
                    (
                        "Choose the best target replacement for region-level icon transcreation.\n"
                        "Source food term: {src_food}\n"
                        "Target culture: {target_culture}\n"
                        "Scene context: {scene_context}\n"
                        "Grounded candidate targets: {candidates}\n\n"
                        'Return JSON only: {"target_food":"one_candidate"}'
                    ),
                )
                prompt = template.format(
                    src_food=src_food,
                    target_culture=input_data.target_culture,
                    scene_context=((scene_graph.get("scene") or {}).get("description") or ""),
                    candidates=candidates,
                )
                res = self.llm_client.generate_reasoning(prompt)
                llm_target = _select_grounded_target(res.get("target_food"), candidates)
            except Exception:
                llm_target = None
            if llm_target:
                target = llm_target
        if not target:
            return []
        if candidates and self.strict_mode:
            _enforce_candidate_constrained_target(
                action="transform",
                target=target,
                candidate_names=candidates,
            )
        elif not candidates and self.strict_mode:
            raise ValueError("Non-KG target")

        day_rows = []
        max_x2 = 0
        for item in extracted:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox")
            if isinstance(bbox, list) and len(bbox) >= 4:
                max_x2 = max(max_x2, int(round(float(max(bbox[0], bbox[2])))))
        day_rows = _infer_row_label_bboxes(extracted)
        if not day_rows or max_x2 <= 0:
            return []

        replacements: List[Dict[str, Any]] = []
        for idx, row in enumerate(day_rows):
            x1, y1, x2, y2 = row
            rx1 = x2 + 8
            ry1 = y1
            rx2 = max_x2
            ry2 = y2
            if rx2 - rx1 < 24 or ry2 - ry1 < 12:
                continue
            replacements.append(
                {
                    "object_id": 100000 + idx,
                    "original": f"{src_food} icon",
                    "new": f"{target} icon",
                    "target": target,
                    "bbox": [rx1, ry1, rx2, ry2],
                    "source": "infographic_region_fallback",
                    "visual_attributes": self.kg_loader.get_visual_attributes(
                        label=target,
                        obj_type="FOOD",
                        culture_name=input_data.target_culture,
                    ),
                }
            )
        return replacements

    def _construct_prompt(
        self,
        obj_label: str,
        obj_type: str,
        source_culture: str,
        target_culture: str,
        candidates: List[str],
        context: str,
        avoid_list: List[str],
        style_priors: Optional[StylePriors] = None,
        sensitivity_notes: Optional[List[str]] = None,
        used_targets: Optional[Set[str]] = None,
        reasoning_strategy: Optional[str] = None,
    ) -> str:
        sensitivity_notes = sensitivity_notes or []
        style_block = ""
        if style_priors:
            style_block = (
                "Style priors for %s: palette %s; motifs %s.\n"
                % (target_culture, style_priors.palette, style_priors.motifs)
            )
        sensitivity_block = ""
        if sensitivity_notes:
            sensitivity_block = "Sensitivity notes (follow these): " + "; ".join(sensitivity_notes) + "\n"
        diversity_block = ""
        if used_targets:
            diversity_block = (
                "Already used targets in this image plan (prefer a different candidate when possible): "
                + ", ".join(sorted(used_targets))
                + ".\n"
            )

        strategy = (reasoning_strategy or _reasoning_strategy()).strip().lower()
        if strategy == "llm_first":
            template = get_prompt(
                "object_reasoning_llm_first.template",
                (
                    "You are a cultural adaptation expert for {target_culture}. "
                    "Object: '{obj_label}' (type: {obj_type}). Scene: {context}\n"
                    "{style_block}{sensitivity_block}{diversity_block}"
                    "Avoid: {avoid_list}\n"
                    'Return JSON: {{"action":"transform|preserve","target_object":"...","rationale":"...","confidence":0.0-1.0}}'
                ),
            )
            return template.format(
                target_culture=target_culture,
                obj_label=obj_label,
                obj_type=obj_type,
                source_culture=source_culture,
                context=context,
                style_block=style_block,
                sensitivity_block=sensitivity_block,
                diversity_block=diversity_block,
                avoid_list=avoid_list,
            )

        if candidates:
            candidate_instruction = (
                "Grounded candidates for substitution (you MUST choose one of these if you transform): %s. "
                "When candidates are provided, prefer action 'transform' and set target_object to one of them. "
                "Prefer candidates that match the object type (%s) and are not already used for other regions."
                % (candidates, obj_type)
            )
        else:
            candidate_instruction = "No grounded candidates in the knowledge base for this type. Use action 'preserve' unless you have a strong reason to suggest a different substitute."

        template = get_prompt(
            "object_reasoning.template",
            (
                "You are a cultural adaptation expert. For an image adapted to {target_culture}, decide for the object '{obj_label}' (type: {obj_type}).\n"
                "Source culture: {source_culture}\n\n"
                "Scene: {context}\n"
                "{style_block}{sensitivity_block}{diversity_block}{candidate_instruction}\n"
                "Avoid (do not use): {avoid_list}\n"
                "Decision policy:\n"
                "- Prefer substitutes that are commonly recognized in {target_culture} for this scene context.\n"
                "- Reject substitutions that are culturally generic, ambiguous, or not context-plausible.\n"
                "- Keep non-target elements unchanged and preserve scene realism.\n"
                "- Keep replacements visually actionable for image editing (concrete object nouns, not abstract concepts).\n\n"
                "KG-first planning rubric:\n"
                "1) If grounded candidates exist, rank them by: (a) scene fit, (b) cultural salience, (c) visual editability.\n"
                "2) Choose the top-ranked grounded candidate for target_object when action is \"transform\".\n"
                "3) If all grounded candidates are unsuitable, set action to \"preserve\" and explain why.\n"
                "4) Never invent a new target_object outside grounded candidates when they are provided.\n"
                "5) Confidence guidance: >=0.75 for strong grounded fit, 0.55-0.74 for usable fit, <0.55 for preserve.\n\n"
                "Respond with exactly one JSON object, no markdown, no code block, no extra text. Use this structure only:\n"
                "{{\"action\": \"transform\" or \"preserve\", \"target_object\": \"substitute name or same as original\", \"rationale\": \"one sentence\", \"confidence\": 0.0 to 1.0}}"
            ),
        )
        return template.format(
            target_culture=target_culture,
            obj_label=obj_label,
            obj_type=obj_type,
            source_culture=source_culture,
            context=context,
            style_block=style_block,
            sensitivity_block=sensitivity_block,
            diversity_block=diversity_block,
            candidate_instruction=candidate_instruction,
            avoid_list=avoid_list,
        )
