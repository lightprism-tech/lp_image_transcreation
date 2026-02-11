from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field

class PreserveConstraint(BaseModel):
    """
    Defines invariants such as layout, pose, lighting, identity anchors.
    Example: "layout", "pose", "lighting"
    """
    aspect: str = Field(..., description="The aspect to preserve, e.g., 'layout', 'pose', 'lighting'.")

class ReplaceAction(BaseModel):
    """
    Defines object substitutions referencing object IDs and optional constraints.
    """
    object_id: int = Field(..., description="ID of the object in the original scene graph.")
    original: str = Field(..., description="Original object label.")
    new: str = Field(..., description="New object label for substitution.")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Optional constraints like size, orientation.")

class EditTextAction(BaseModel):
    """
    Defines localized text edits with bounding boxes and replacement strings.
    """
    bbox: List[int] = Field(..., description="Bounding box [x_min, y_min, x_max, y_max].")
    original: str = Field(..., description="Original text content.")
    translated: str = Field(..., description="Translated or replaced text content.")

class AdjustStyleAction(BaseModel):
    """
    Defines global style harmonization signals.
    """
    palette: Optional[str] = Field(None, description="Color palette to apply.")
    motifs: List[str] = Field(default_factory=list, description="Cultural motifs to add.")
    texture: Optional[str] = Field(None, description="Texture style to apply.")

class EditPlan(BaseModel):
    """
    The Edit-Plan JSON as a Control Interface.
    """
    preserve: List[str] = Field(default_factory=list, description="List of aspects to preserve.")
    replace: List[ReplaceAction] = Field(default_factory=list, description="List of object replacement actions.")
    edit_text: List[EditTextAction] = Field(default_factory=list, description="List of text edit actions.")
    adjust_style: Optional[AdjustStyleAction] = Field(None, description="Global style adjustments.")
