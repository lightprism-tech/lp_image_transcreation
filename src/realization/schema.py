import json
from src.realization.models import EditPlan

def get_edit_plan_schema() -> dict:
    """
    Returns the JSON schema for the EditPlan.
    """
    return EditPlan.model_json_schema()

def validate_edit_plan(plan_data: dict) -> EditPlan:
    """
    Validates a dictionary against the EditPlan schema.
    """
    return EditPlan(**plan_data)
