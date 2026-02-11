# Stage 3: Visual Realization

The **Visual Realization** module transforms the input image based on the **Edit-Plan** generated in Stage 2. It acts as a control interface that strictly follows the plan to preserve identity/layout while applying cultural adaptations.

## Key Concepts

-   **Edit-Plan**: A structured JSON instruction sheet defining what to change (replace, edit text) and what to keep (preserve).
-   **Control Interface**: The engine ensures that changes are bounded by the plan, preventing hallucinations or drift.

## Usage

Run the realization engine as a module:

```bash
python -m src.realization.main \
  --img <path_to_input_image> \
  --plan <path_to_edit_plan.json> \
  --output <path_to_output_image>
```

### Example

```bash
python -m src.realization.main \
  --img data/input_images/picnic_usa.jpg \
  --plan outputs/plan_japan.json \
  --output outputs/final_japan.png
```

## JSON Schema

You can inspect the expected structure of the Edit-Plan by running:

```python
from src.realization.schema import get_edit_plan_schema
import json
print(json.dumps(get_edit_plan_schema(), indent=2))
```

## Structure

-   `engine.py`: Core logic for applying edits.
-   `models.py`: Pydantic definitions for the Edit-Plan methods.
-   `schema.py`: Schema validation tools.
-   `main.py`: CLI entry point.
