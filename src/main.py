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
from src.realization.main import _apply_mock_instance_changes, _apply_mock_overlay

logger = logging.getLogger(__name__)
_REASONING_ENGINE_CACHE: Dict[str, CulturalReasoningEngine] = {}
_REALIZATION_ENGINE_CACHE: Dict[str, RealizationEngine] = {}


def _stage_log(stage_id: str, status: str, message: str = "") -> None:
    suffix = f" - {message}" if message else ""
    logger.info("[STAGE %s][%s]%s", stage_id, status, suffix)


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
    logger.info("Using cached stage-1 perception JSON: %s", path)
    return _load_json(path)


def _load_cached_reasoning_graph(path: Path) -> Dict[str, Any]:
    logger.info("Using cached stage-2 reasoning JSON: %s", path)
    return _load_json(path)


def _normalize_stage2_objects(scene_graph: Dict[str, Any]) -> None:
    """
    Ensure stage-2 objects keep a stable schema required by downstream consumers.
    """
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
    objects = scene_graph.get("objects") if isinstance(scene_graph.get("objects"), list) else []
    object_count = len(objects)
    image_type = ((scene_graph.get("image_type") or {}).get("type") or "unknown").lower()

    if transform_count > 0 or text_edit_count > 0:
        return

    logger.warning(
        "Stage-2 produced no actionable edits (transformations=0, edit_text=0). "
        "image_type=%s, detected_objects=%d. "
        "This commonly happens for infographic/document images with no detectable replaceable objects.",
        image_type,
        object_count,
    )


def _get_reasoning_engine(knowledge_graph_path: Path, use_model_cache: bool) -> CulturalReasoningEngine:
    key = str(knowledge_graph_path.resolve())
    if use_model_cache and key in _REASONING_ENGINE_CACHE:
        logger.info("Using cached reasoning engine for KG: %s", key)
        return _REASONING_ENGINE_CACHE[key]
    logger.info("Initializing reasoning engine for KG: %s", key)
    engine = CulturalReasoningEngine(str(knowledge_graph_path))
    if use_model_cache:
        _REASONING_ENGINE_CACHE[key] = engine
    return engine


def _get_realization_engine(config: Dict[str, Any], use_model_cache: bool) -> RealizationEngine:
    # Stable cache key so equivalent configs share one initialized engine/model.
    key = json.dumps(config, sort_keys=True, default=str)
    if use_model_cache and key in _REALIZATION_ENGINE_CACHE:
        logger.info("Using cached realization engine for config key")
        return _REALIZATION_ENGINE_CACHE[key]
    logger.info("Initializing realization engine")
    engine = RealizationEngine(config=config)
    if use_model_cache:
        _REALIZATION_ENGINE_CACHE[key] = engine
    return engine


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
) -> Dict[str, Any]:
    _stage_log("2", "LOAD", f"stage-2 reasoning JSON: {stage2_json_path}")
    stage2_data = _load_json(stage2_json_path)
    image_path = _resolve_stage2_image_path(stage2_data, stage2_json_path)
    _stage_log("2", "READY", f"resolved input image: {image_path}")

    plan_data = adapt_plan_to_edit_format(stage2_data)
    edit_plan = validate_edit_plan(plan_data)
    target_culture = (stage2_data.get("edit_plan") or {}).get("target_culture", "target")

    config: Dict[str, Any] = {}
    if realization_config_path:
        config = _load_json(realization_config_path)
    config["target_culture"] = config.get("target_culture") or target_culture

    _stage_log("3", "START", "realization")
    realization_engine = _get_realization_engine(config=config, use_model_cache=use_model_cache)
    generated_path = realization_engine.generate(edit_plan, str(image_path))

    final_image_output.parent.mkdir(parents=True, exist_ok=True)
    if generated_path and os.path.exists(generated_path):
        from shutil import copy2

        copy2(generated_path, final_image_output)
        _stage_log("3", "DONE", f"generated output copied: {final_image_output}")
    else:
        has_bbox_replaces = any(r.bbox and len(r.bbox) >= 4 for r in edit_plan.replace)
        if has_bbox_replaces:
            _apply_mock_instance_changes(
                edit_plan, str(image_path), str(final_image_output), target_culture
            )
        else:
            _apply_mock_overlay(str(image_path), str(final_image_output), target_culture)
        _stage_log("3", "DONE", f"fallback output generated: {final_image_output}")

    return {
        "stage2_json": str(stage2_json_path),
        "resolved_image": str(image_path),
        "final_image_output": str(final_image_output),
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
) -> Dict[str, Any]:
    from src.perception.main import main as run_perception

    logger.info("Pipeline start: image=%s target=%s", image_path, target_culture)
    logger.info("Cache mode: %s", "enabled" if use_cache else "disabled")

    if use_cache and perception_output.exists():
        _stage_log("1", "CACHED", f"{perception_output}")
        scene_graph = _load_cached_scene_graph(perception_output)
    else:
        _stage_log("1", "START", f"perception on image: {image_path}")
        try:
            scene_graph = run_perception(str(image_path), str(perception_output))
            _stage_log("1", "DONE", f"{perception_output}")
        except Exception:
            _stage_log("1", "FAILED", "perception failed")
            logger.error("Stage 1 failed while running perception.", exc_info=True)
            raise

    logger.info(
        "Handoff: Stage 1 output ready for Stage 2 (scene graph with %d top-level keys)",
        len(scene_graph) if isinstance(scene_graph, dict) else 0,
    )

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
            )
            reasoning_input = ReasoningInput(
                scene_graph=scene_graph,
                target_culture=target_culture,
                avoid_list=avoid_list,
            )
            plan = engine.analyze_image(reasoning_input)
            edit_text = engine.build_text_edits(reasoning_input)
            adapted_scene_graph = apply_plan_to_input(scene_graph, plan)
            if edit_text:
                adapted_scene_graph["edit_text"] = edit_text
            _normalize_stage2_objects(adapted_scene_graph)
            adapted_scene_graph["edit_plan"] = {
                "target_culture": plan.target_culture,
                "transformations": [t.model_dump() for t in plan.transformations],
                "preservations": [p.model_dump() for p in plan.preservations],
                "edit_text": edit_text,
            }
            _log_stage2_actionability(
                scene_graph=adapted_scene_graph,
                edit_plan=adapted_scene_graph["edit_plan"],
            )
            _save_json(adapted_scene_graph, reasoning_output)
            _stage_log("2", "DONE", f"{reasoning_output}")
        except Exception:
            _stage_log("2", "FAILED", "reasoning failed")
            logger.error("Stage 2 failed while running reasoning.", exc_info=True)
            raise

    logger.info(
        "Handoff: Stage 2 output ready for Stage 3 (reasoning JSON path: %s)",
        reasoning_output,
    )
    _stage_log("3", "START", "realization")
    plan_data = adapt_plan_to_edit_format(adapted_scene_graph)
    edit_plan = validate_edit_plan(plan_data)

    config: Dict[str, Any] = {}
    if realization_config_path:
        with open(realization_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    config["target_culture"] = config.get("target_culture") or target_culture

    try:
        realization_engine = _get_realization_engine(
            config=config,
            use_model_cache=use_model_cache,
        )
        generated_path = realization_engine.generate(edit_plan, str(image_path))
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
        has_bbox_replaces = any(r.bbox and len(r.bbox) >= 4 for r in edit_plan.replace)
        if has_bbox_replaces:
            _apply_mock_instance_changes(
                edit_plan, str(image_path), str(final_image_output), target_culture
            )
        else:
            _apply_mock_overlay(str(image_path), str(final_image_output), target_culture)
        _stage_log("3", "DONE", f"fallback output: {final_image_output}")

    return {
        "perception_output": str(perception_output),
        "reasoning_output": str(reasoning_output),
        "final_image_output": str(final_image_output),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

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
        help="Optional realization config JSON path (for inpainting, prompt settings, etc.)",
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

    args = parser.parse_args()

    realization_config_path = Path(args.config) if args.config else None

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
        )
        logger.info("Realization-only run complete")
        logger.info("Stage-2 JSON: %s", outputs["stage2_json"])
        logger.info("Resolved image: %s", outputs["resolved_image"])
        logger.info("Final image: %s", outputs["final_image_output"])
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
        )
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        logger.debug("Pipeline traceback:\n%s", traceback.format_exc())
        sys.exit(1)

    logger.info("Pipeline complete")
    logger.info("Perception JSON: %s", outputs["perception_output"])
    logger.info("Reasoning JSON: %s", outputs["reasoning_output"])
    logger.info("Final image: %s", outputs["final_image_output"])


if __name__ == "__main__":
    main()
