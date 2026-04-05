from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator

class PreserveConstraint(BaseModel):
    """
    Defines invariants such as layout, pose, lighting, identity anchors.
    Example: "layout", "pose", "lighting"
    """
    aspect: str = Field(..., description="The aspect to preserve, e.g., 'layout', 'pose', 'lighting'.")

class ReplaceAction(BaseModel):
    """
    Defines object substitutions referencing object IDs and optional constraints.
    bbox is [x_min, y_min, x_max, y_max] for inpainting; from scene graph when available.
    Stage 2 / detection outputs often use floats; coerced to int on validation.
    """
    object_id: int = Field(..., description="ID of the object in the original scene graph.")
    original: str = Field(..., description="Original object label.")
    new: str = Field(..., description="New object label for substitution.")
    bbox: Optional[List[int]] = Field(default=None, description="Bounding box [x_min, y_min, x_max, y_max] for inpainting.")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Optional constraints like size, orientation.")

    @field_validator("bbox", mode="before")
    @classmethod
    def bbox_coerce_ints(cls, v: Optional[List]) -> Optional[List[int]]:
        if v is None:
            return None
        if not isinstance(v, list) or len(v) < 4:
            return v
        return [int(round(float(x))) for x in v[:4]]

class EditTextAction(BaseModel):
    """
    Defines localized text edits with bounding boxes and replacement strings.
    """
    bbox: List[int] = Field(..., description="Bounding box [x_min, y_min, x_max, y_max].")
    original: str = Field(..., description="Original text content.")
    translated: str = Field(..., description="Translated or replaced text content.")
    style: Optional[Dict[str, Any]] = Field(
        default=None,
        description="OCR/source-region style metadata (font/color/background/size).",
    )

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
