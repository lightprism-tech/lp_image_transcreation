from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
# from src.perception.schemas.scene_schema import ScenePerceptionSchema # Removed: file does not exist

# --- Cultural Knowledge Base K(c): grounded mappings and constraints per target culture ---

class SubstitutionEntry(BaseModel):
    """One source -> many candidate targets for a culturally salient class."""
    source: str
    targets: List[str]

class StylePriors(BaseModel):
    """Stylistic priors for visual adaptation (palette, motifs)."""
    palette: List[str] = Field(default_factory=list)
    motifs: List[str] = Field(default_factory=list)

class CulturalKBEntry(BaseModel):
    """
    Knowledge base entry for a target culture. Encodes grounded substitutions,
    negative constraints (avoid lists), stylistic priors, and sensitivity notes.
    """
    culture: str
    substitutions: Dict[str, List[SubstitutionEntry]] = Field(
        default_factory=dict,
        description="Maps culturally salient class (e.g. FOOD, CLOTHING) to list of source->targets."
    )
    avoid: List[str] = Field(
        default_factory=list,
        description="Negative constraints: stereotypical or cliched edits to disallow."
    )
    style_priors: Optional[StylePriors] = None
    sensitivity_notes: List[str] = Field(
        default_factory=list,
        description="Notes that discourage harmful or cliched edits."
    )

class CulturalNode(BaseModel):
    id: str
    label: str
    type: str
    culture: Optional[str] = None # Enriched attribute

class Transformation(BaseModel):
    original_object: str
    original_type: str
    target_object: str
    rationale: str
    confidence: float

class Preservation(BaseModel):
    original_object: str
    rationale: str

class TranscreationPlan(BaseModel):
    target_culture: str
    transformations: List[Transformation]
    preservations: List[Preservation]
    avoidance_adherence: List[str] # Notes on what was avoided

class ReasoningInput(BaseModel):
    scene_graph: Dict[str, Any] # Stage 1 output
    target_culture: str
    avoid_list: List[str] = Field(default_factory=list)
