import argparse
import json
import os
import sys
import logging
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.reasoning.engine import CulturalReasoningEngine
from src.reasoning.types import ReasoningInput

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
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False) # ensure_ascii=False for cultural chars

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Cultural Reasoning")
    parser.add_argument("--input", required=True, help="Path to Stage 1 output JSON (Scene Graph)")
    parser.add_argument("--target", required=True, help="Target Culture (e.g., 'Japan', 'India')")
    parser.add_argument("--kg", required=True, help="Path to Knowledge Graph JSON")
    parser.add_argument("--output", required=True, help="Path to output JSON (Transcreation Plan)")
    parser.add_argument("--avoid", nargs="*", default=[], help="List of items to avoid")

    args = parser.parse_args()

    # 1. Load Input
    try:
        scene_graph = load_json(args.input)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    # 2. Init Engine
    try:
        engine = CulturalReasoningEngine(args.kg)
    except FileNotFoundError:
        print(f"Error: Knowledge Graph file '{args.kg}' not found.")
        sys.exit(1)
    
    # 3. Prepare Input
    reasoning_input = ReasoningInput(
        scene_graph=scene_graph,
        target_culture=args.target,
        avoid_list=args.avoid
    )

    # 4. Run Analysis
    print(f"Running Cultural Reasoning for target: {args.target}...")
    plan = engine.analyze_image(reasoning_input)

    # 5. Save Output
    save_json(plan.model_dump(), args.output)
    print(f"Transcreation Plan saved to: {args.output}")

if __name__ == "__main__":
    main()
