import copy
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from src.reasoning.schemas import (
    ReasoningInput, TranscreationPlan, Transformation, Preservation, CulturalNode,
    StylePriors,
)
from src.reasoning.knowledge_loader import KnowledgeLoader
from src.reasoning.llm_client import LLMClient

logger = logging.getLogger(__name__)
STRICT_KB_GROUNDED_TRANSFORMS = True


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
    label_lower = (obj_label or "").lower()
    if label_lower in label_to_type:
        return label_to_type[label_lower]
    # Heuristic fallback when KB label_to_type is missing or incomplete.
    # This keeps Stage 2 actionable for common detector labels.
    heuristic_map = {
        "person": "CLOTHING",
        "shirt": "CLOTHING",
        "t-shirt": "CLOTHING",
        "jacket": "CLOTHING",
        "dress": "CLOTHING",
        "hat": "CLOTHING",
        "handbag": "CLOTHING",
        "backpack": "CLOTHING",
        "bicycle": "SPORT",
        "sports ball": "SPORT",
        "baseball bat": "SPORT",
        "skateboard": "SPORT",
        "surfboard": "SPORT",
        "skis": "SPORT",
        "snowboard": "SPORT",
        "tennis racket": "SPORT",
        "hamburger": "FOOD",
        "pizza": "FOOD",
        "hot dog": "FOOD",
        "sandwich": "FOOD",
        "donut": "FOOD",
        "cake": "FOOD",
    }
    if label_lower in heuristic_map:
        return heuristic_map[label_lower]
    attrs = obj.get("attributes") or {}
    clothing = attrs.get("clothing")
    if isinstance(clothing, list) and clothing:
        return "CLOTHING"
    if isinstance(clothing, str) and clothing:
        return "CLOTHING"
    # Additional fallback for frequent detector labels in infographic/ad images.
    if label_lower in ("potted plant", "bench", "boat"):
        return "ART"
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
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue

        translated = _rewrite_text_for_region(
            original=original,
            target_culture=target_culture,
            image_type=((scene_graph.get("image_type") or {}).get("type") or ""),
            scene_context=scene_context,
            full_text=full_text,
            bbox=bbox,
            style=item.get("style") if isinstance(item.get("style"), dict) else None,
            llm_client=llm_client,
        )

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
        prompt = (
            "Rewrite one OCR text region for cultural transcreation.\n"
            f"Target culture: {target_culture}\n"
            f"Image type: {image_type}\n"
            f"Scene context: {scene_context}\n"
            f"Neighbor text context: {full_text[:500]}\n"
            f"Original region text: {original_clean}\n\n"
            "Return exactly one JSON object: "
            '{"candidates":["...","...","..."]}\n'
            f"Rules: keep semantics, avoid object substitution language, and each candidate length <= {max_chars}."
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


def _validate_rewrite_constraints(original: str, candidate: str, bbox: Optional[List[int]], style: Optional[Dict[str, Any]]) -> str:
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


def _is_infographic_mode(scene_graph: Dict[str, Any]) -> bool:
    image_type = ((scene_graph.get("image_type") or {}).get("type") or "").lower()
    # Keep strict safeguards for true infographic/document layouts.
    # Posters frequently contain natural-image objects and should stay transformable
    # unless stronger infographic signals are present.
    if image_type in {"infographic", "document"}:
        return True
    analysis = scene_graph.get("infographic_analysis") or {}
    if analysis.get("enabled") and int(analysis.get("icon_cluster_count", 0) or 0) >= 3:
        return True
    extracted = ((scene_graph.get("text") or {}).get("extracted") or [])
    return isinstance(extracted, list) and len(extracted) >= 6


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

    if obj_type_upper in {"FOOD", "SPORT", "CLOTHING"} and (has_icon_anchor or confidence >= 0.65):
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
    def __init__(self, knowledge_graph_path: str):
        self.kg_loader = KnowledgeLoader(knowledge_graph_path)
        self.llm_client = LLMClient()

    def analyze_image(self, input_data: ReasoningInput) -> TranscreationPlan:
        logger.info("Starting analysis for target culture: %s", input_data.target_culture)
        scene_objects = input_data.scene_graph.get("objects", [])
        target_culture = input_data.target_culture
        transformations: List[Transformation] = []
        preservations: List[Preservation] = []
        avoidance_adherence: List[str] = []
        infographic_mode = _is_infographic_mode(input_data.scene_graph)
        scene_context = input_data.scene_graph.get("scene", {}).get("description", "")

        # Resolve avoid list: KB avoid + CLI override
        kb_avoid = self.kg_loader.get_avoid_list(target_culture)
        avoid_list = list(kb_avoid) if kb_avoid else []
        for a in input_data.avoid_list:
            if a and a not in avoid_list:
                avoid_list.append(a)

        for obj in scene_objects:
            obj_label = obj.get("label") or obj.get("class_name")
            if not obj_label:
                continue
            logger.info(
                "Reasoning object start: label=%s, confidence=%s",
                obj_label,
                obj.get("confidence"),
            )
            if _is_ambiguous_person_in_infographic(infographic_mode, obj_label, obj, scene_context):
                preservations.append(Preservation(
                    original_object=obj_label,
                    rationale="Infographic safety policy: ambiguous person detection preserved.",
                ))
                logger.info("Reasoning preserve: %s (ambiguous infographic person policy)", obj_label)
                continue

            # (1) Identify type and source culture: from graph node or infer from KB label_to_type / attributes
            kg_node = self.kg_loader.find_node(obj_label)
            source_culture = "Unknown"
            obj_type = "object"
            label_to_type = self.kg_loader.get_label_to_type()
            if kg_node:
                obj_type = kg_node.type
                source_culture = self.kg_loader.get_culture_of_node(kg_node.id) or "Unknown"
                logger.info(
                    "KB node matched: label=%s, type=%s, source_culture=%s",
                    obj_label,
                    obj_type,
                    source_culture,
                )
            else:
                inferred = _infer_cultural_type(obj_label, obj, label_to_type)
                if inferred:
                    obj_type = inferred
                    logger.info("Type inferred from mapping/heuristics: label=%s -> type=%s", obj_label, obj_type)

            if _should_preserve_non_text_in_infographic(infographic_mode, obj_type, obj):
                preservations.append(Preservation(
                    original_object=obj_label,
                    rationale="Infographic mode: preserving COCO-style object to avoid semantic drift.",
                ))
                logger.info("Reasoning preserve: %s (infographic COCO-style policy)", obj_label)
                continue

            # Only change culture-related types: obj_type must be in the KG's cultural types
            cultural_types = self.kg_loader.get_cultural_types()
            if cultural_types and obj_type not in cultural_types:
                preservations.append(Preservation(
                    original_object=obj_label,
                    rationale="Not a culture-related type in KB; preserved.",
                ))
                logger.info("Reasoning preserve: %s (type=%s not cultural in KB)", obj_label, obj_type)
                continue

            # (2) Retrieve candidate substitutes from KB first (or from graph by type + culture)
            candidate_labels = self.kg_loader.get_candidates_from_kb(target_culture, obj_label, obj_type)
            if candidate_labels:
                logger.info(
                    "KB candidates found: label=%s, type=%s, count=%d, top=%s",
                    obj_label,
                    obj_type,
                    len(candidate_labels),
                    candidate_labels[0],
                )
            if not candidate_labels:
                candidates = self.kg_loader.get_nodes_by_type_and_culture(obj_type, target_culture)
                candidate_labels = [c.label for c in candidates]
                if candidate_labels:
                    logger.info(
                        "Type+culture candidates found: label=%s, type=%s, count=%d, top=%s",
                        obj_label,
                        obj_type,
                        len(candidate_labels),
                        candidate_labels[0],
                    )
            # Strict KB-gated policy: do not synthesize transform candidates from LLM
            # when the knowledge graph has no grounded options.
            if not candidate_labels and STRICT_KB_GROUNDED_TRANSFORMS:
                preservations.append(Preservation(
                    original_object=obj_label,
                    rationale="No grounded KB candidates; preserved by KB-first policy.",
                ))
                logger.info(
                    "Reasoning preserve: %s (no KB candidates for type=%s, strict KB-first policy)",
                    obj_label,
                    obj_type,
                )
                continue
            # Optional legacy fallback path if strict policy is disabled.
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
                    context=input_data.scene_graph.get("scene", {}).get("description", ""),
                    avoid_list=avoid_list,
                )
                logger.info(
                    "LLM candidates received: label=%s, count=%d, sample=%s",
                    obj_label,
                    len(candidate_labels),
                    candidate_labels[:3],
                )
            # Prefer substitute from KB when defined (e.g. bicycle + India -> Cricket)
            preferred = self.kg_loader.get_preferred_substitution(obj_label, target_culture)
            if preferred and preferred in candidate_labels:
                candidate_labels = [preferred] + [c for c in candidate_labels if c != preferred]
                logger.info("Preferred substitution applied: label=%s -> %s", obj_label, preferred)

            # (3) Filter candidates using avoid lists and context compatibility
            candidate_labels, notes = _filter_candidates_by_avoid(candidate_labels, avoid_list)
            avoidance_adherence.extend(notes)
            logger.info(
                "Candidates after avoid filter: label=%s, count=%d",
                obj_label,
                len(candidate_labels),
            )

            context = input_data.scene_graph.get("scene", {}).get("description", "")
            style_priors = self.kg_loader.get_style_priors(target_culture)
            sensitivity_notes = self.kg_loader.get_sensitivity_notes(target_culture)

            prompt = self._construct_prompt(
                obj_label=obj_label,
                obj_type=obj_type,
                source_culture=source_culture,
                target_culture=target_culture,
                candidates=candidate_labels,
                context=context,
                avoid_list=avoid_list,
                style_priors=style_priors,
                sensitivity_notes=sensitivity_notes or [],
            )

            # (4) LLM decides transform vs preserve
            reasoning_result = self.llm_client.generate_reasoning(prompt)
            logger.info(
                "LLM reasoning result: label=%s, action=%s, target=%s, confidence=%s",
                obj_label,
                reasoning_result.get("action"),
                reasoning_result.get("target_object"),
                reasoning_result.get("confidence"),
            )

            # (5) Output structured plan (transformation or preservation)
            action = reasoning_result.get("action")
            if action == "transform":
                transformations.append(Transformation(
                    original_object=obj_label,
                    original_type=obj_type,
                    target_object=reasoning_result.get("target_object", "Unknown"),
                    rationale=reasoning_result.get("rationale", "No rationale provided."),
                    confidence=float(reasoning_result.get("confidence", 0.0)),
                ))
                logger.info(
                    "Reasoning transform: %s (%s) -> %s",
                    obj_label,
                    obj_type,
                    reasoning_result.get("target_object", "Unknown"),
                )
            elif candidate_labels:
                # Deterministic fallback: if grounded candidates exist but model preserves,
                # select top grounded candidate so downstream realization has actionable edits.
                fallback_target = candidate_labels[0]
                transformations.append(Transformation(
                    original_object=obj_label,
                    original_type=obj_type,
                    target_object=fallback_target,
                    rationale="Grounded fallback: candidates available in KB; selected top candidate.",
                    confidence=0.55,
                ))
                logger.info(
                    "Applied grounded fallback transform for '%s' -> '%s' (type=%s)",
                    obj_label,
                    fallback_target,
                    obj_type,
                )
            else:
                rationale = reasoning_result.get("rationale", "Preserved by default.")
                if isinstance(rationale, str) and "LLM Service Unavailable" in rationale:
                    rationale = "No grounded candidates in KB for this object; preserved."
                preservations.append(Preservation(
                    original_object=obj_label,
                    rationale=rationale,
                ))
                logger.info("Reasoning preserve: %s (rationale=%s)", obj_label, rationale)

        logger.info(
            "Reasoning complete: transformations=%d, preservations=%d, avoid_notes=%d",
            len(transformations),
            len(preservations),
            len(avoidance_adherence),
        )
        return TranscreationPlan(
            target_culture=target_culture,
            transformations=transformations,
            preservations=preservations,
            avoidance_adherence=avoidance_adherence,
        )

    def build_text_edits(self, input_data: ReasoningInput) -> List[Dict[str, Any]]:
        """Create edit_text actions for document/infographic-heavy images."""
        image_type = ((input_data.scene_graph.get("image_type") or {}).get("type") or "").lower()
        if image_type not in {"document", "poster", "ui", "social_media", "infographic"}:
            return []
        return _build_text_edits_for_document(
            input_data.scene_graph, input_data.target_culture, llm_client=self.llm_client
        )

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

        if candidates:
            candidate_instruction = (
                "Grounded candidates for substitution (you MUST choose one of these if you transform): %s. "
                "When candidates are provided, prefer action 'transform' and set target_object to one of them."
                % candidates
            )
        else:
            candidate_instruction = "No grounded candidates in the knowledge base for this type. Use action 'preserve' unless you have a strong reason to suggest a different substitute."

        return f"""You are a cultural adaptation expert. For an image adapted to {target_culture}, decide for the object '{obj_label}' (type: {obj_type}).
Source culture: {source_culture}

Scene: {context}
{style_block}{sensitivity_block}{candidate_instruction}
Avoid (do not use): {avoid_list}

Respond with exactly one JSON object, no markdown, no code block, no extra text. Use this structure only:
{{"action": "transform" or "preserve", "target_object": "substitute name or same as original", "rationale": "one sentence", "confidence": 0.0 to 1.0}}
"""
