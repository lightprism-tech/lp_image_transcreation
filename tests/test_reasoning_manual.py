import os
import sys
import json
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.reasoning.engine import CulturalReasoningEngine
from src.reasoning.types import ReasoningInput

# Mock the LLM Client to avoid actual API calls and dependency on .env for this test
def mock_llm_response(prompt):
    return {
        "action": "transform",
        "target_object": "Onigiri",
        "rationale": "Onigiri is a popular Japanese rice ball snack, equivalent to a burger in casual dining context.",
        "confidence": 0.95
    }

def main():
    print("Running Manual Verification for Stage 2...")

    # 1. Setup Mock Engine
    # We need a real KG file, assuming it exists at reliable path
    kg_path = "data/knowledge_base/countries_graph.json"
    if not os.path.exists(kg_path):
        print(f"Error: KG not found at {kg_path}")
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
        print("Analyzing input...")
        plan = engine.analyze_image(input_data)
        
        # 4. verify Output
        print("\nTranscreation Plan:")
        print(json.dumps(plan.model_dump(), indent=2))
        
        # Basic Assertions
        assert plan.target_culture == "Japan"
        found_burger_transform = False
        for t in plan.transformations:
            if t.original_object == "Hamburger" and t.target_object == "Onigiri":
                found_burger_transform = True
                print("\nSUCCESS: Hamburger -> Onigiri transformation found.")
        
        if not found_burger_transform:
            print("\nFAILURE: Hamburger transformation not found.")
            sys.exit(1)

    except Exception as e:
        print(f"\nFAILURE: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
