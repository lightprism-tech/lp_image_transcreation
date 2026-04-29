import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path for "src.*" imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    # Perception package imports use "perception.*", which lives under src/.
    sys.path.append(str(SRC_ROOT))

from src.reasoning.engine import CulturalReasoningEngine, apply_plan_to_input
from src.reasoning.schemas import ReasoningInput
from src.realization.engine import RealizationEngine
from src.realization.schema import adapt_plan_to_edit_format, validate_edit_plan
from src.utilities.terminal_logger import configure_terminal_logger, print_startup_logo

logger = logging.getLogger("pipeline_main")
_STAGE_LOGGER_NAMES = {
    "1": "stage1_perception",
    "2": "stage2_reasoning",
    "3": "stage3_realization",
}
_REASONING_ENGINE_CACHE: Dict[str, CulturalReasoningEngine] = {}
_REALIZATION_ENGINE_CACHE: Dict[str, RealizationEngine] = {}
DEFAULT_REALIZATION_CONFIG_PATH = PROJECT_ROOT / "data" / "config" / "realization_config.json"


def _stage_logger(stage_id: str) -> logging.Logger:
    return logging.getLogger(_STAGE_LOGGER_NAMES.get(stage_id, "pipeline_main"))


def _stage_banner(stage_id: str, title: str) -> None:
    _stage_logger(stage_id).info("========== %s (Stage %s) ==========", title, stage_id)


def _stage_log(stage_id: str, status: str, message: str = "") -> None:
    suffix = f" - {message}" if message else ""
    _stage_logger(stage_id).info("[STAGE %s][%s]%s", stage_id, status, suffix)


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _default_output_paths(image_path: Path, output_dir: Path) -> Dict[str, Path]:
    stem = image_path.stem
    return {
        "perception_json": output_dir / "json" / f"{stem}_stage1_perception.json",
        "reasoning_json": output_dir / "json" / f"{stem}_stage2_reasoning.json",
        "final_image": output_dir / "images" / f"{stem}_stage3_realized.png",
    }


def _resolve_run_output_dir(output_dir: str, run_name: str) -> Path:
    if run_name:
        return Path(output_dir) / run_name
    return Path(output_dir) / datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_cached_scene_graph(path: Path) -> Dict[str, Any]:
    _stage_logger("1").info("Using cached stage-1 perception JSON: %s", path)
    return _load_json(path)


def _load_cached_reasoning_graph(path: Path) -> Dict[str, Any]:
    _stage_logger("2").info("Using cached stage-2 reasoning JSON: %s", path)
    return _load_json(path)


def _normalize_stage2_objects(scene_graph: Dict[str, Any]) -> None:
    """
    Ensure stage-2 objects keep a stable schema required by downstream consumers.
    """
    objects = scene_graph.get("objects")
    if not isinstance(objects, list) or not objects:
        visual_regions = scene_graph.get("visual_regions")
        if isinstance(visual_regions, list) and visual_regions:
            scene_graph["objects"] = [
                {
                    "id": region.get("id", idx),
                    "bbox": region.get("bbox", []),
                    "class_name": region.get("type") or "unknown_visual_region",
                    "label": region.get("type") or "unknown_visual_region",
                    "original_class_name": region.get("type") or "unknown_visual_region",
                    "caption": region.get("description", ""),
                    "confidence": region.get("confidence", 0.0),
                    "quality_flags": region.get("quality_flags", []),
                }
                for idx, region in enumerate(visual_regions)
                if isinstance(region, dict)
            ]
            objects = scene_graph.get("objects")
    if not isinstance(objects, list):
        scene_graph["objects"] = []
        return

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        class_name = obj.get("class_name") or obj.get("label") or "unknown_object"
        obj["class_name"] = class_name
        obj["original_class_name"] = obj.get("original_class_name") or class_name
        if "bbox" not in obj:
            obj["bbox"] = []


def _log_stage2_actionability(scene_graph: Dict[str, Any], edit_plan: Dict[str, Any]) -> None:
    """
    Log why Stage 2 may have no actionable edits, without injecting fake transforms.
    """
    transformations = edit_plan.get("transformations")
    transform_count = len(transformations) if isinstance(transformations, list) else 0
    text_edits = edit_plan.get("edit_text")
    text_edit_count = len(text_edits) if isinstance(text_edits, list) else 0
    region_replace = edit_plan.get("region_replace")
    region_replace_count = len(region_replace) if isinstance(region_replace, list) else 0
    objects = scene_graph.get("objects") if isinstance(scene_graph.get("objects"), list) else []
    object_count = len(objects)
    image_type = ((scene_graph.get("image_type") or {}).get("type") or "unknown").lower()

    if transform_count > 0 or text_edit_count > 0 or region_replace_count > 0:
        return

    message = (
        "Stage-2 produced no actionable edits (transformations=0, edit_text=0, region_replace=0). "
        "image_type=%s, detected_objects=%d."
    )
    if image_type in {"infographic", "document", "poster"}:
        _stage_logger("2").warning(
            message + " This commonly happens for infographic/document-style images with no detectable replaceable objects.",
            image_type,
            object_count,
        )
    else:
        _stage_logger("2").info(
            message + " No grounded cultural substitutions were found for this image.",
            image_type,
            object_count,
        )


def _get_reasoning_engine(
    knowledge_graph_path: Path,
    use_model_cache: bool,
    strict_mode: bool = True,
) -> CulturalReasoningEngine:
    key = f"{knowledge_graph_path.resolve()}::{int(bool(strict_mode))}"
    if use_model_cache and key in _REASONING_ENGINE_CACHE:
        _stage_logger("2").info("Using cached reasoning engine for KG: %s", key)
        return _REASONING_ENGINE_CACHE[key]
    _stage_logger("2").info("Initializing reasoning engine for KG: %s", key)
    engine = CulturalReasoningEngine(str(knowledge_graph_path), strict_mode=strict_mode)
    if use_model_cache:
        _REASONING_ENGINE_CACHE[key] = engine
    return engine


def _get_realization_engine(config: Dict[str, Any], use_model_cache: bool) -> RealizationEngine:
    # Stable cache key so equivalent configs share one initialized engine/model.
    key = json.dumps(config, sort_keys=True, default=str)
    if use_model_cache and key in _REALIZATION_ENGINE_CACHE:
        _stage_logger("3").info("Using cached realization engine for config key")
        return _REALIZATION_ENGINE_CACHE[key]
    _stage_logger("3").info("Initializing realization engine")
    engine = RealizationEngine(config=config)
    if use_model_cache:
        _REALIZATION_ENGINE_CACHE[key] = engine
    return engine


def _build_run_metrics_payload(
    stage2_trace: Dict[str, Any],
    stage3_metrics: Dict[str, Any],
    run_context: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "run_context": run_context,
        "stage2": stage2_trace,
        "stage3": stage3_metrics,
    }


def _score_below_threshold(metrics: Dict[str, Any], validation_cfg: Dict[str, Any]) -> List[str]:
    failures: List[str] = []
    min_cultural = float(validation_cfg.get("min_cultural_score", 0.7))
    min_object = float(validation_cfg.get("min_object_presence_score", 0.7))
    cultural = float(metrics.get("cultural_score", 0.0) or 0.0)
    object_presence = float(metrics.get("object_presence_score", 0.0) or 0.0)
    if cultural < min_cultural:
        failures.append(f"cultural_score {cultural:.4f} < {min_cultural:.4f}")
    if object_presence < min_object:
        failures.append(f"object_presence_score {object_presence:.4f} < {min_object:.4f}")
    return failures


def _validate_stage3_quality(
    realization_engine: RealizationEngine,
    generated_path: str,
    target_culture: str,
    target_objects: List[str],
    validation_cfg: Dict[str, Any],
) -> List[str]:
    failures: List[str] = []
    if not realization_engine.passes_composite_validation(
        image_path=generated_path,
        target_culture=target_culture,
        target_objects=target_objects,
    ):
        failures.append("composite validation failed")
    failures.extend(_score_below_threshold(realization_engine.get_run_metrics(), validation_cfg))
    return failures


def _stage3_quality_score(metrics: Dict[str, Any]) -> float:
    return float(metrics.get("cultural_score", 0.0) or 0.0) + float(
        metrics.get("object_presence_score", 0.0) or 0.0
    )


def _edit_plan_has_actions(edit_plan: Any) -> bool:
    """True when Stage 3 has a real edit to execute."""
    return bool(
        getattr(edit_plan, "replace", None)
        or getattr(edit_plan, "edit_text", None)
        or getattr(edit_plan, "adjust_style", None)
    )


def _stage3_skip_reason(edit_plan: Any) -> str:
    """Build a human-readable reason for Stage 3 skip."""
    replace_count = len(getattr(edit_plan, "replace", None) or [])
    edit_text_count = len(getattr(edit_plan, "edit_text", None) or [])
    has_adjust_style = bool(getattr(edit_plan, "adjust_style", None))
    return (
        "no actionable edits in plan "
        f"(replace={replace_count}, edit_text={edit_text_count}, adjust_style={has_adjust_style})"
    )


def _generate_with_strict_quality(
    realization_engine: RealizationEngine,
    edit_plan: Any,
    image_path: Path,
    target_culture: str,
    target_objects: List[str],
    validation_cfg: Dict[str, Any],
) -> str:
    max_attempts = max(1, int(validation_cfg.get("max_quality_attempts", 2)))
    strict_quality = bool(validation_cfg.get("strict_quality_gate", True))
    return_best_effort = bool(validation_cfg.get("return_best_attempt_on_quality_failure", True))
    generated_path = ""
    best_path = ""
    best_metrics: Dict[str, Any] = {}
    best_failures: List[str] = []
    best_score = -1.0
    last_failures: List[str] = []
    for attempt in range(1, max_attempts + 1):
        generated_path = realization_engine.generate(edit_plan, str(image_path))
        if not generated_path or not os.path.exists(generated_path):
            last_failures = ["realization did not produce an image"]
        else:
            current_metrics = realization_engine.get_run_metrics()
            current_score = _stage3_quality_score(current_metrics)
            if current_score > best_score:
                best_score = current_score
                best_path = generated_path
                best_metrics = current_metrics
            last_failures = _validate_stage3_quality(
                realization_engine=realization_engine,
                generated_path=generated_path,
                target_culture=target_culture,
                target_objects=target_objects,
                validation_cfg=validation_cfg,
            )
            if current_score >= best_score:
                best_failures = list(last_failures)
        if not last_failures:
            realization_engine._run_metrics = {
                **realization_engine.get_run_metrics(),
                "quality_gate_passed": True,
                "quality_failures": [],
            }
            return generated_path
        _stage_logger("3").warning(
            "Stage-3 quality attempt %d/%d failed: %s",
            attempt,
            max_attempts,
            "; ".join(last_failures),
        )
    if best_path and return_best_effort:
        _stage_logger("3").warning(
            "All Stage-3 quality attempts missed strict thresholds; selecting best attempt with score %.4f.",
            best_score,
        )
        realization_engine._run_metrics = {
            **best_metrics,
            "quality_gate_passed": False,
            "quality_failures": best_failures or last_failures,
            "selected_best_effort": True,
        }
        return best_path
    if strict_quality:
        raise RuntimeError("Stage 3 failed strict quality gate: " + "; ".join(last_failures))
    return generated_path


def _append_feedback_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    if path.exists():
        try:
            records = _load_json(path) if isinstance(_load_json(path), list) else []
        except Exception:
            records = []
    records.append(record)
    _save_json(records, path)


def _resolve_stage2_image_path(stage2_data: Dict[str, Any], stage2_json_path: Path) -> Path:
    metadata = stage2_data.get("metadata") if isinstance(stage2_data, dict) else {}
    raw_path = ""
    if isinstance(metadata, dict):
        raw_path = metadata.get("image_path") or ""

    candidates: List[Path] = []
    if raw_path:
        candidates.append(Path(raw_path))
        normalized = raw_path.replace("\\", "/")
        if normalized.startswith("/data/"):
            # Docker path -> local workspace path fallback for Windows/Linux local runs.
            rel = normalized[len("/data/") :]
            candidates.append(PROJECT_ROOT / "data" / rel)

    image_name = metadata.get("image_name") if isinstance(metadata, dict) else None
    if image_name:
        candidates.append(PROJECT_ROOT / "data" / "input" / "samples" / str(image_name))
        candidates.append(stage2_json_path.parent / str(image_name))

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not resolve image path from stage-2 JSON. "
        "Checked metadata.image_path and common local fallbacks under data/input/samples."
    )


def run_realization_from_stage2_json(
    stage2_json_path: Path,
    final_image_output: Path,
    realization_config_path: Path = None,
    use_model_cache: bool = True,
    debug_prompt: bool = False,
    metrics_output: Path = None,
) -> Dict[str, Any]:
    _stage_log("2", "LOAD", f"stage-2 reasoning JSON: {stage2_json_path}")
    stage2_data = _load_json(stage2_json_path)
    image_path = _resolve_stage2_image_path(stage2_data, stage2_json_path)
    _stage_log("2", "READY", f"resolved input image: {image_path}")
    _stage_logger("3").info("Realization-only mode enabled")

    plan_data = adapt_plan_to_edit_format(stage2_data)
    edit_plan = validate_edit_plan(plan_data)
    target_culture = (stage2_data.get("edit_plan") or {}).get("target_culture", "target")
    _stage_logger("3").info(
        "Loaded edit plan (replace=%d, preserve=%d, edit_text=%d)",
        len(edit_plan.replace),
        len(edit_plan.preserve),
        len(edit_plan.edit_text),
    )

    config: Dict[str, Any] = {}
    if realization_config_path:
        config = _load_json(realization_config_path)
    config["target_culture"] = config.get("target_culture") or target_culture
    config["debug_prompt"] = bool(debug_prompt)
    validation_cfg = config.get("validation") if isinstance(config.get("validation"), dict) else {}

    if not _edit_plan_has_actions(edit_plan):
        from shutil import copy2

        final_image_output.parent.mkdir(parents=True, exist_ok=True)
        copy2(image_path, final_image_output)
        skip_reason = _stage3_skip_reason(edit_plan)
        _stage_log("3", "SKIPPED", f"{skip_reason}; copied source image")
        run_metrics_path = metrics_output or (stage2_json_path.parent / f"{stage2_json_path.stem}_run_metrics.json")
        run_metrics_payload = _build_run_metrics_payload(
            stage2_trace={},
            stage3_metrics={
                "quality_gate_passed": False,
                "quality_failures": ["no_actionable_edits"],
                "skipped": True,
                "skip_reason": skip_reason,
            },
            run_context={
                "stage2_json": str(stage2_json_path),
                "resolved_image": str(image_path),
                "final_image_output": str(final_image_output),
            },
        )
        _save_json(run_metrics_payload, run_metrics_path)
        logger.info("Saved run metrics to: %s", run_metrics_path)
        return {
            "stage2_json": str(stage2_json_path),
            "resolved_image": str(image_path),
            "final_image_output": str(final_image_output),
            "metrics_output": str(run_metrics_path),
        }

    _stage_log("3", "START", "realization")
    realization_engine = _get_realization_engine(config=config, use_model_cache=use_model_cache)
    target_objects = [r.new for r in edit_plan.replace if isinstance(r.new, str) and r.new.strip()]
    generated_path = _generate_with_strict_quality(
        realization_engine=realization_engine,
        edit_plan=edit_plan,
        image_path=image_path,
        target_culture=target_culture,
        target_objects=target_objects,
        validation_cfg=validation_cfg,
    )

    final_image_output.parent.mkdir(parents=True, exist_ok=True)
    if generated_path and os.path.exists(generated_path):
        from shutil import copy2

        copy2(generated_path, final_image_output)
        _stage_log("3", "DONE", f"generated output copied: {final_image_output}")
    else:
        raise RuntimeError(
            "Stage 3 did not produce a generated image. Fallback rendering is disabled."
        )

    run_metrics_path = metrics_output or (stage2_json_path.parent / f"{stage2_json_path.stem}_run_metrics.json")
    run_metrics_payload = _build_run_metrics_payload(
        stage2_trace={},
        stage3_metrics=realization_engine.get_run_metrics(),
        run_context={
            "stage2_json": str(stage2_json_path),
            "resolved_image": str(image_path),
            "final_image_output": str(final_image_output),
        },
    )
    _save_json(run_metrics_payload, run_metrics_path)
    logger.info("Saved run metrics to: %s", run_metrics_path)
    feedback_path = run_metrics_path.parent / "feedback_outcomes.json"
    _append_feedback_record(
        feedback_path,
        {
            "input": str(image_path.name),
            "selected": target_objects,
            "score": run_metrics_payload.get("stage3", {}).get("cultural_score", 0.0),
        },
    )
    logger.info("Updated feedback outcomes: %s", feedback_path)

    return {
        "stage2_json": str(stage2_json_path),
        "resolved_image": str(image_path),
        "final_image_output": str(final_image_output),
        "metrics_output": str(run_metrics_path),
    }


def run_full_pipeline(
    image_path: Path,
    target_culture: str,
    knowledge_graph_path: Path,
    avoid_list: List[str],
    perception_output: Path,
    reasoning_output: Path,
    final_image_output: Path,
    realization_config_path: Path = None,
    use_cache: bool = True,
    use_model_cache: bool = True,
    debug_plan: bool = False,
    debug_prompt: bool = False,
    debug_kg_selection: bool = False,
    metrics_output: Path = None,
) -> Dict[str, Any]:
    from src.perception.main import main as run_perception

    logger.info("Pipeline start: image=%s target=%s", image_path, target_culture)
    logger.info("Cache mode: %s", "enabled" if use_cache else "disabled")
    logger.info("Output targets: stage1=%s stage2=%s stage3=%s", perception_output, reasoning_output, final_image_output)
    _stage_banner("1", "Perception")

    if use_cache and perception_output.exists():
        _stage_log("1", "CACHED", f"{perception_output}")
        scene_graph = _load_cached_scene_graph(perception_output)
    else:
        _stage_log("1", "START", f"perception on image: {image_path}")
        try:
            scene_graph = run_perception(str(image_path), str(perception_output))
            _stage_log("1", "DONE", f"{perception_output}")
            _stage_logger("1").info(
                "Perception summary: objects=%d text_regions=%d image_type=%s",
                len(scene_graph.get("objects") or []) if isinstance(scene_graph, dict) else 0,
                len(((scene_graph.get("text") or {}).get("regions") or [])) if isinstance(scene_graph, dict) else 0,
                ((scene_graph.get("image_type") or {}).get("type") or "unknown") if isinstance(scene_graph, dict) else "unknown",
            )
        except Exception:
            _stage_log("1", "FAILED", "perception failed")
            logger.error("Stage 1 failed while running perception.", exc_info=True)
            raise

    logger.info(
        "Handoff: Stage 1 output ready for Stage 2 (scene graph with %d top-level keys)",
        len(scene_graph) if isinstance(scene_graph, dict) else 0,
    )
    if isinstance(scene_graph, dict):
        _normalize_stage2_objects(scene_graph)
    _stage_banner("2", "Reasoning")
    _stage_logger("2").info("Stage 2 input ready: scene graph prepared from Stage 1.")

    if use_cache and reasoning_output.exists():
        _stage_log("2", "CACHED", f"{reasoning_output}")
        adapted_scene_graph = _load_cached_reasoning_graph(reasoning_output)
        _normalize_stage2_objects(adapted_scene_graph)
        if not isinstance(adapted_scene_graph.get("edit_plan"), dict):
            adapted_scene_graph["edit_plan"] = {}
        if not adapted_scene_graph["edit_plan"].get("target_culture"):
            adapted_scene_graph["edit_plan"]["target_culture"] = target_culture
        _log_stage2_actionability(
            scene_graph=adapted_scene_graph,
            edit_plan=adapted_scene_graph["edit_plan"],
        )
        _save_json(adapted_scene_graph, reasoning_output)
    else:
        _stage_log("2", "START", f"reasoning for target culture: {target_culture}")
        try:
            engine = _get_reasoning_engine(
                knowledge_graph_path=knowledge_graph_path,
                use_model_cache=use_model_cache,
                strict_mode=True,
            )
            engine.debug_plan = debug_plan
            engine.debug_kg_selection = debug_kg_selection
            reasoning_input = ReasoningInput(
                scene_graph=scene_graph,
                target_culture=target_culture,
                avoid_list=avoid_list,
            )
            plan = engine.analyze_image(reasoning_input)
            edit_text = engine.build_text_edits(reasoning_input)
            region_replace = list(plan.region_replace or [])
            adapted_scene_graph = apply_plan_to_input(scene_graph, plan)
            if edit_text:
                adapted_scene_graph["edit_text"] = edit_text
            if region_replace:
                adapted_scene_graph["region_replace"] = region_replace
            _normalize_stage2_objects(adapted_scene_graph)
            adapted_scene_graph["edit_plan"] = {
                "target_culture": plan.target_culture,
                "transformations": [t.model_dump() for t in plan.transformations],
                "preservations": [p.model_dump() for p in plan.preservations],
                "edit_text": edit_text,
                "region_replace": region_replace,
                "scene_adaptation": plan.scene_adaptation,
            }
            _log_stage2_actionability(
                scene_graph=adapted_scene_graph,
                edit_plan=adapted_scene_graph["edit_plan"],
            )
            _save_json(adapted_scene_graph, reasoning_output)
            _stage_log("2", "DONE", f"{reasoning_output}")
            _stage_logger("2").info(
                "Reasoning summary: transforms=%d preserve=%d edit_text=%d region_replace=%d",
                len(plan.transformations),
                len(plan.preservations),
                len(edit_text),
                len(region_replace),
            )
            if debug_plan:
                _stage_logger("2").info("Stage-2 raw plan trace: %s", engine.get_debug_trace().get("raw_plan"))
                _stage_logger("2").info(
                    "Stage-2 normalized plan trace: %s",
                    engine.get_debug_trace().get("normalized_plan"),
                )
            if debug_kg_selection:
                _stage_logger("2").info(
                    "Stage-2 KG selections: %s",
                    engine.get_debug_trace().get("kg_selections"),
                )
        except Exception:
            _stage_log("2", "FAILED", "reasoning failed")
            logger.error("Stage 2 failed while running reasoning.", exc_info=True)
            raise

    logger.info(
        "Handoff: Stage 2 output ready for Stage 3 (reasoning JSON path: %s)",
        reasoning_output,
    )
    _stage_banner("3", "Realization")
    _stage_logger("3").info("Stage 3 input ready: edit plan prepared from Stage 2.")
    _stage_log("3", "START", "realization")
    plan_data = adapt_plan_to_edit_format(adapted_scene_graph)
    edit_plan = validate_edit_plan(plan_data)

    config: Dict[str, Any] = {}
    if realization_config_path:
        with open(realization_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    config["target_culture"] = config.get("target_culture") or target_culture
    config["debug_prompt"] = bool(debug_prompt)
    validation_cfg = config.get("validation") if isinstance(config.get("validation"), dict) else {}

    if not _edit_plan_has_actions(edit_plan):
        from shutil import copy2

        final_image_output.parent.mkdir(parents=True, exist_ok=True)
        copy2(image_path, final_image_output)
        skip_reason = _stage3_skip_reason(edit_plan)
        _stage_log("3", "SKIPPED", f"{skip_reason}; copied source image")
        run_metrics_path = metrics_output or (reasoning_output.parent / f"{image_path.stem}_run_metrics.json")
        run_metrics_payload = _build_run_metrics_payload(
            stage2_trace=engine.get_debug_trace() if "engine" in locals() else {},
            stage3_metrics={
                "quality_gate_passed": False,
                "quality_failures": ["no_actionable_edits"],
                "skipped": True,
                "skip_reason": skip_reason,
            },
            run_context={
                "image_path": str(image_path),
                "target_culture": target_culture,
                "reasoning_output": str(reasoning_output),
                "final_image_output": str(final_image_output),
            },
        )
        _save_json(run_metrics_payload, run_metrics_path)
        logger.info("Saved run metrics to: %s", run_metrics_path)
        return {
            "perception_output": str(perception_output),
            "reasoning_output": str(reasoning_output),
            "final_image": str(final_image_output),
            "metrics_output": str(run_metrics_path),
        }

    try:
        realization_engine = _get_realization_engine(
            config=config,
            use_model_cache=use_model_cache,
        )
        target_objects = [r.new for r in edit_plan.replace if isinstance(r.new, str) and r.new.strip()]
        generated_path = _generate_with_strict_quality(
            realization_engine=realization_engine,
            edit_plan=edit_plan,
            image_path=image_path,
            target_culture=target_culture,
            target_objects=target_objects,
            validation_cfg=validation_cfg,
        )
        _stage_logger("3").info(
            "Realization plan summary: replace=%d preserve=%d edit_text=%d",
            len(edit_plan.replace),
            len(edit_plan.preserve),
            len(edit_plan.edit_text),
        )
    except Exception:
        _stage_log("3", "FAILED", "realization failed")
        logger.error("Stage 3 failed while running realization.", exc_info=True)
        raise

    final_image_output.parent.mkdir(parents=True, exist_ok=True)
    if generated_path and os.path.exists(generated_path):
        from shutil import copy2

        copy2(generated_path, final_image_output)
        _stage_log("3", "DONE", f"{final_image_output}")
    else:
        raise RuntimeError(
            "Stage 3 did not produce a generated image. Fallback rendering is disabled."
        )

    run_metrics_path = metrics_output or (reasoning_output.parent / f"{image_path.stem}_run_metrics.json")
    run_metrics_payload = _build_run_metrics_payload(
        stage2_trace=engine.get_debug_trace() if "engine" in locals() else {},
        stage3_metrics=realization_engine.get_run_metrics(),
        run_context={
            "image_path": str(image_path),
            "target_culture": target_culture,
            "reasoning_output": str(reasoning_output),
            "final_image_output": str(final_image_output),
        },
    )
    _save_json(run_metrics_payload, run_metrics_path)
    logger.info("Saved run metrics to: %s", run_metrics_path)
    feedback_path = run_metrics_path.parent / "feedback_outcomes.json"
    _append_feedback_record(
        feedback_path,
        {
            "input": str(image_path.name),
            "selected": target_objects,
            "score": run_metrics_payload.get("stage3", {}).get("cultural_score", 0.0),
        },
    )
    logger.info("Updated feedback outcomes: %s", feedback_path)

    return {
        "perception_output": str(perception_output),
        "reasoning_output": str(reasoning_output),
        "final_image_output": str(final_image_output),
        "metrics_output": str(run_metrics_path),
    }


def main() -> None:
    configure_terminal_logger(level=os.getenv("LOG_LEVEL", "INFO"))
    print_startup_logo()

    parser = argparse.ArgumentParser(
        description="Run full transcreation pipeline: Perception -> Reasoning -> Realization"
    )
    parser.add_argument("--img", required=False, help="Path to input image")
    parser.add_argument("--target", required=False, help="Target culture (e.g., India, Japan)")
    parser.add_argument(
        "--kg",
        default="data/knowledge_base/countries_graph.json",
        help="Path to knowledge graph JSON file (default: data/knowledge_base/countries_graph.json)",
    )
    parser.add_argument(
        "--stage2-json",
        default=None,
        help="Optional path to existing stage-2 reasoning JSON; runs realization only",
    )
    parser.add_argument(
        "--output-dir",
        default="data/output",
        help="Base output directory for generated files (default: data/output)",
    )
    parser.add_argument(
        "--run-name",
        default="my_run",
        help="Run folder name under --output-dir (default: my_run)",
    )
    parser.add_argument(
        "--perception-output",
        default=None,
        help="Optional explicit path for stage 1 output JSON",
    )
    parser.add_argument(
        "--reasoning-output",
        default=None,
        help="Optional explicit path for stage 2 output JSON",
    )
    parser.add_argument(
        "--final-image-output",
        default=None,
        help="Optional explicit path for stage 3 output image",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional realization config JSON path (for inpainting, prompt settings, etc.). "
            "Defaults to data/config/realization_config.json when not provided."
        ),
    )
    parser.add_argument(
        "--avoid",
        nargs="*",
        default=[],
        help="Optional list of items to avoid in reasoning substitutions",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable stage output cache and force all stages to run",
    )
    parser.add_argument(
        "--no-model-cache",
        action="store_true",
        help="Disable in-process model/engine cache and force re-initialization",
    )
    parser.add_argument(
        "--debug-plan",
        action="store_true",
        help="Log Stage-2 raw and normalized planning traces.",
    )
    parser.add_argument(
        "--debug-prompt",
        action="store_true",
        help="Log final prompts sent to Stage-3 inpainting.",
    )
    parser.add_argument(
        "--debug-kg-selection",
        action="store_true",
        help="Log selected knowledge-graph items (id/name) for replacements.",
    )
    parser.add_argument(
        "--metrics-output",
        default=None,
        help="Optional output path for per-run metrics JSON.",
    )

    args = parser.parse_args()

    if args.config:
        realization_config_path = Path(args.config)
    else:
        realization_config_path = DEFAULT_REALIZATION_CONFIG_PATH if DEFAULT_REALIZATION_CONFIG_PATH.exists() else None
    if realization_config_path:
        logger.info("Using realization config: %s", realization_config_path)

    if args.stage2_json:
        stage2_json_path = Path(args.stage2_json)
        if not stage2_json_path.exists():
            logger.error("Stage-2 JSON not found: %s", stage2_json_path)
            sys.exit(1)

        stage2_data = _load_json(stage2_json_path)
        resolved_image = _resolve_stage2_image_path(stage2_data, stage2_json_path)
        run_output_dir = _resolve_run_output_dir(args.output_dir, args.run_name)
        defaults = _default_output_paths(resolved_image, run_output_dir)
        final_image_output = (
            Path(args.final_image_output) if args.final_image_output else defaults["final_image"]
        )

        outputs = run_realization_from_stage2_json(
            stage2_json_path=stage2_json_path,
            final_image_output=final_image_output,
            realization_config_path=realization_config_path,
            use_model_cache=not args.no_model_cache,
            debug_prompt=args.debug_prompt,
            metrics_output=Path(args.metrics_output) if args.metrics_output else None,
        )
        logger.info("Realization-only run complete")
        logger.info("Stage-2 JSON: %s", outputs["stage2_json"])
        logger.info("Resolved image: %s", outputs["resolved_image"])
        logger.info("Final image: %s", outputs["final_image_output"])
        logger.info("Run metrics JSON: %s", outputs["metrics_output"])
        return

    if not args.img or not args.target:
        logger.error(
            "Full run requires --img and --target. "
            "Optional: --kg (defaults to data/knowledge_base/countries_graph.json). "
            "Or use --stage2-json for realization-only mode."
        )
        sys.exit(1)

    image_path = Path(args.img)
    knowledge_graph_path = Path(args.kg)
    if not image_path.exists():
        logger.error("Input image not found: %s", image_path)
        sys.exit(1)
    if not knowledge_graph_path.exists():
        logger.error("Knowledge graph not found: %s", knowledge_graph_path)
        sys.exit(1)

    run_output_dir = _resolve_run_output_dir(args.output_dir, args.run_name)
    defaults = _default_output_paths(image_path, run_output_dir)
    perception_output = Path(args.perception_output) if args.perception_output else defaults["perception_json"]
    reasoning_output = Path(args.reasoning_output) if args.reasoning_output else defaults["reasoning_json"]
    final_image_output = Path(args.final_image_output) if args.final_image_output else defaults["final_image"]
    metrics_output = Path(args.metrics_output) if args.metrics_output else None

    try:
        outputs = run_full_pipeline(
            image_path=image_path,
            target_culture=args.target,
            knowledge_graph_path=knowledge_graph_path,
            avoid_list=args.avoid,
            perception_output=perception_output,
            reasoning_output=reasoning_output,
            final_image_output=final_image_output,
            realization_config_path=realization_config_path,
            use_cache=not args.no_cache,
            use_model_cache=not args.no_model_cache,
            debug_plan=args.debug_plan,
            debug_prompt=args.debug_prompt,
            debug_kg_selection=args.debug_kg_selection,
            metrics_output=metrics_output,
        )
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        logger.debug("Pipeline traceback:\n%s", traceback.format_exc())
        sys.exit(1)

    logger.info("Pipeline complete")
    perception_out = outputs.get("perception_output") or outputs.get("perception_json")
    reasoning_out = outputs.get("reasoning_output") or outputs.get("reasoning_json")
    final_image_out = outputs.get("final_image_output") or outputs.get("final_image")
    metrics_out = outputs.get("metrics_output")

    if perception_out:
        logger.info("Perception JSON: %s", perception_out)
    if reasoning_out:
        logger.info("Reasoning JSON: %s", reasoning_out)
    if final_image_out:
        logger.info("Final image: %s", final_image_out)
    if metrics_out:
        logger.info("Run metrics JSON: %s", metrics_out)


if __name__ == "__main__":
    main()
