# Stage 2: Cultural Reasoning (The "Brain")

This module is the core decision-making engine of the Image Transcreation Pipeline. It takes the structured scene graph from **Stage 1 (Perception)** and determines how to adapt the content for a **Target Culture**, using a **Knowledge Graph** and an **LLM** for reasoning.

## Pipeline Workflow

1.  **Input**: A JSON file containing the Scene Graph (Stage 1 Output).
2.  **Context**: Target Culture (e.g., "Japan", "India") + Knowledge Graph.
3.  **Process**:
    *   **Identify**: Detects cultural elements in the input (e.g., "Hamburger").
    *   **Retrieve**: Finds relevant cultural substitutes in the Knowledge Base (e.g., "Onigiri").
    *   **Reason**: Asks an LLM to decide the best substitution strategy based on context.
4.  **Output**: A `TranscreationPlan` JSON detailing what to transform and what to preserve.

## Setup

1.  **Environment Variables**:
    Ensure your `.env` file in the project root has your LLM configuration:
    ```env
    LLM_PROVIDER=openai
    LLM_API_KEY=your-api-key
    LLM_MODEL=gpt-4o
    ```

2.  **Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

##  Usage

Run the reasoning engine via the command line:

```bash
python src/reasoning/main.py \
  --input <path_to_stage1_output.json> \
  --target "<Target_Culture>" \
  --kg <path_to_knowledge_graph.json> \
  --output <path_to_output_plan.json>
```

### Example: Hamburger to Japan

```bash
python src/reasoning/main.py \
  --input data/input_examples/picnic_usa.json \
  --target "Japan" \
  --kg data/knowledge_base/countries_graph.json \
  --output outputs/plan_japan.json
```

## Output Format

The output is a JSON file containing the **Transcreation Plan**:

```json
{
  "target_culture": "Japan",
  "transformations": [
    {
      "original_object": "Hamburger",
      "original_type": "FOOD",
      "target_object": "Onigiri",
      "rationale": "Onigiri is a culturally appropriate casual snack for a picnic setting in Japan.",
      "confidence": 0.95
    }
  ],
  "preservations": [
    {
      "original_object": "Tree",
      "rationale": "Trees are universal and fit the target context."
    }
  ]
}
```
