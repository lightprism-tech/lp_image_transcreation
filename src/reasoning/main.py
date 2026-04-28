import argparse
import json
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.reasoning.engine import CulturalReasoningEngine, apply_plan_to_input
from src.reasoning.schemas import ReasoningInput

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Any, path: str):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False) # ensure_ascii=False for cultural chars


def _build_default_output_path(input_path: str, target_culture: str, output_dir: str, run_name: str = "") -> str:
    input_stem = Path(input_path).stem or "stage1"
    safe_target = (target_culture or "target").strip().replace(" ", "_")
    if not run_name:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return str(Path(output_dir) / run_name / "json" / f"{safe_target}_{input_stem}_stage2_reasoning.json")

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Cultural Reasoning")
    parser.add_argument("--input", required=True, help="Path to Stage 1 output JSON (Scene Graph)")
    parser.add_argument("--target", required=True, help="Target Culture (e.g., 'Japan', 'India')")
    parser.add_argument("--kg", required=True, help="Path to Knowledge Graph JSON (e.g. data/knowledge_base/countries_graph.json)")
    parser.add_argument("--output", default=None, help="Path to output JSON (Transcreation Plan). If omitted, a run folder path is auto-generated.")
    parser.add_argument("--output-dir", default="data/output", help="Base output directory when --output is not provided.")
    parser.add_argument("--run-name", default="", help="Optional run folder name under --output-dir (e.g. run_check).")
    parser.add_argument("--avoid", nargs="*", default=[], help="List of items to avoid")

    args = parser.parse_args()

    # 1. Load Input
    try:
        scene_graph = load_json(args.input)
    except FileNotFoundError:
        logger.error("Error: Input file '%s' not found.", args.input)
        sys.exit(1)

    # 2. Init Engine
    try:
        engine = CulturalReasoningEngine(args.kg, strict_mode=True)
    except FileNotFoundError:
        logger.error("Error: Knowledge Graph file '%s' not found.", args.kg)
        sys.exit(1)
    
    # 3. Prepare Input
    reasoning_input = ReasoningInput(
        scene_graph=scene_graph,
        target_culture=args.target,
        avoid_list=args.avoid
    )

    # 4. Run Analysis (LLM-backed reasoning using the knowledge graph)
    logger.info("Running Cultural Reasoning for target: %s...", args.target)
    plan = engine.analyze_image(reasoning_input)
    edit_text = engine.build_text_edits(reasoning_input)
    region_replace = list(plan.region_replace or [])

    # 5. Apply plan to input: same format as input, only data replaced for target culture
    output_data = apply_plan_to_input(scene_graph, plan)
    if edit_text:
        output_data["edit_text"] = edit_text
    if region_replace:
        output_data["region_replace"] = region_replace

    # 6. Embed edit_plan so realization can use this file (objects + plan + bboxes)
    output_data["edit_plan"] = {
        "target_culture": plan.target_culture,
        "transformations": [t.model_dump() for t in plan.transformations],
        "preservations": [p.model_dump() for p in plan.preservations],
        "edit_text": edit_text,
        "region_replace": region_replace,
        "scene_adaptation": plan.scene_adaptation,
    }

    # 7. Save output (preserves input structure; only values replaced)
    output_path = args.output or _build_default_output_path(
        input_path=args.input,
        target_culture=args.target,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
    save_json(output_data, output_path)
    logger.info("Output saved to: %s (same format as input, data adapted for %s)", output_path, args.target)

if __name__ == "__main__":
    main()
