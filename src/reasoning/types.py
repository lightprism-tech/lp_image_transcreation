from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
# from src.perception.schemas.scene_schema import ScenePerceptionSchema # Removed: file does not exist

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
