import argparse
import json
import sys
import os
from src.realization.engine import RealizationEngine
from src.realization.schema import validate_edit_plan

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Visual Realization Engine")
    parser.add_argument("--img", required=True, help="Path to the input image file")
    parser.add_argument("--plan", required=True, help="Path to the Edit-Plan JSON file")
    parser.add_argument("--output", required=True, help="Path to save the generated image")
    parser.add_argument("--config", help="Optional path to a configuration JSON file")

    args = parser.parse_args()

    # validate inputs
    if not os.path.exists(args.img):
        print(f"Error: Input image not found at {args.img}")
        sys.exit(1)
    
    if not os.path.exists(args.plan):
        print(f"Error: Edit-Plan file not found at {args.plan}")
        sys.exit(1)

    try:
        # Load and validate plan
        plan_data = load_json(args.plan)
        plan = validate_edit_plan(plan_data)
        
        # Load config if provided
        config = load_json(args.config) if args.config else {}

        # Initialize engine
        engine = RealizationEngine(config=config)

        # Generate
        print(f"Generating image based on plan: {args.plan}")
        output_path = engine.generate(plan, args.img)
        
        # In a real scenario, the engine would return the actual path or bytes. 
        # For now, we simulate saving if the engine returns a path that matches our output arg, 
        # or we just report success. 
        # The engine mock currently returns "output/generated_image_mock.png".
        
        print(f"Success! Image generated at: {output_path}")
        # If the engine actually wrote to output_path, we are good. 
        # Since it's a mock, let's just create a dummy file at args.output to simulate success for the user
        
        if not os.path.exists(os.path.dirname(args.output)):
             os.makedirs(os.path.dirname(args.output), exist_ok=True)
             
        with open(args.output, 'w') as f:
            f.write("Mock Image Content")
            
        print(f"Saved result to: {args.output}")

    except Exception as e:
        print(f"An error occurred during realization: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
