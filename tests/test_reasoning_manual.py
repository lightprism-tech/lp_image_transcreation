import json
import logging
import os
import sys
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.reasoning.engine import CulturalReasoningEngine
from src.reasoning.schemas import ReasoningInput

logger = logging.getLogger(__name__)

# Mock the LLM Client to avoid actual API calls and dependency on .env for this test
def mock_llm_response(prompt):
    return {
        "action": "transform",
        "target_object": "Onigiri",
        "rationale": "Onigiri is a popular Japanese rice ball snack, equivalent to a burger in casual dining context.",
        "confidence": 0.95
    }

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger.info("Running Manual Verification for Stage 2...")

    # 1. Setup Mock Engine
    kg_path = "data/knowledge_base/countries_graph.json"
    if not os.path.exists(kg_path):
        logger.error("Error: KG not found at %s", kg_path)
        return

    try:
        engine = CulturalReasoningEngine(kg_path)
        # Mocking the LLM client's generate_reasoning method
        engine.llm_client.generate_reasoning = MagicMock(side_effect=mock_llm_response)
        
        # 2. Create Dummy Input (Stage 1 Output)
        scene_graph = {
            "scene": {
                "description": "A person eating a hamburger in a park.",
                "setting": "park"
            },
            "objects": [
                {"id": 1, "label": "Hamburger", "type": "FOOD"},
                {"id": 2, "label": "Tree", "type": "PLANT"} # Helper object
            ]
        }
        
        input_data = ReasoningInput(
            scene_graph=scene_graph,
            target_culture="Japan",
            avoid_list=["stereotypes"]
        )

        # 3. Run Analysis
        logger.info("Analyzing input...")
        plan = engine.analyze_image(input_data)

        # 4. verify Output
        logger.info("Transcreation Plan: %s", json.dumps(plan.model_dump(), indent=2))

        assert plan.target_culture == "Japan"
        found_burger_transform = False
        for t in plan.transformations:
            if t.original_object == "Hamburger" and t.target_object == "Onigiri":
                found_burger_transform = True
                logger.info("SUCCESS: Hamburger -> Onigiri transformation found.")

        if not found_burger_transform:
            logger.error("FAILURE: Hamburger transformation not found.")
            sys.exit(1)

    except Exception as e:
        logger.error("FAILURE: Exception occurred: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
