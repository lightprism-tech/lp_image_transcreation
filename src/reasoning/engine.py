import json
import logging
from typing import List, Dict, Any, Optional
from src.reasoning.types import (
    ReasoningInput, TranscreationPlan, Transformation, Preservation, CulturalNode
)
from src.reasoning.knowledge_loader import KnowledgeLoader
from src.reasoning.llm_client import LLMClient

logger = logging.getLogger(__name__)

class CulturalReasoningEngine:
    def __init__(self, knowledge_graph_path: str):
        self.kg_loader = KnowledgeLoader(knowledge_graph_path)
        self.llm_client = LLMClient()

    def analyze_image(self, input_data: ReasoningInput) -> TranscreationPlan:
        logger.info(f"Starting analysis for target culture: {input_data.target_culture}")
        scene_objects = input_data.scene_graph.get("objects", []) # Assuming Stage 1 output has "objects" list
        target_culture = input_data.target_culture
        
        transformations: List[Transformation] = []
        preservations: List[Preservation] = []
        
        for obj in scene_objects:
            obj_label = obj.get("label") or obj.get("class_name") # Handle different schema keys
            obj_id = obj.get("id")
            
            if not obj_label:
                continue

            # 1. Identify Source Culture and Type from KG
            kg_node = self.kg_loader.find_node(obj_label)
            source_culture = "Unknown"
            obj_type = "object"
            
            if kg_node:
                obj_type = kg_node.type
                source_culture = self.kg_loader.get_culture_of_node(kg_node.id) or "Unknown"
            
            # Simple logic: If source culture is known and different from target, OR if LLM decides
            # For now, we rely on the prompt to decide if adaptation is needed, fueled by KG candidates.
            
            # 2. Retrieve Candidates from KG for Target Culture
            candidates = self.kg_loader.get_nodes_by_type_and_culture(obj_type, target_culture)
            candidate_labels = [c.label for c in candidates]
            
            # 3. Construct LLM Prompt
            prompt = self._construct_prompt(
                obj_label=obj_label,
                obj_type=obj_type,
                source_culture=source_culture,
                target_culture=target_culture,
                candidates=candidate_labels,
                context=input_data.scene_graph.get("scene", {}).get("description", ""),
                avoid_list=input_data.avoid_list
            )
            
            # 4. Get LLM Decision
            reasoning_result = self.llm_client.generate_reasoning(prompt)
            
            # 5. Parse Decision
            if reasoning_result.get("action") == "transform":
                transformations.append(Transformation(
                    original_object=obj_label,
                    original_type=obj_type,
                    target_object=reasoning_result.get("target_object", "Unknown"),
                    rationale=reasoning_result.get("rationale", "No rationale provided."),
                    confidence=reasoning_result.get("confidence", 0.0)
                ))
            else:
                preservations.append(Preservation(
                    original_object=obj_label,
                    rationale=reasoning_result.get("rationale", "Preserved by default.")
                ))

        return TranscreationPlan(
            target_culture=target_culture,
            transformations=transformations,
            preservations=preservations,
            avoidance_adherence=[] 
        )

    def _construct_prompt(self, obj_label: str, obj_type: str, source_culture: str, 
                          target_culture: str, candidates: List[str], context: str, avoid_list: List[str]) -> str:
        
        return f"""
        You are a cultural adaptation expert.
        Task: Decide whether to transform the object '{obj_label}' (Type: {obj_type}) found in a '{source_culture}' context to be appropriate for a '{target_culture}' setting.
        
        Image Context: {context}
        
        Knowledge Graph Candidates for {target_culture}: {candidates}
        Avoid List: {avoid_list}
        
        Return JSON with:
        - "action": "transform" or "preserve"
        - "target_object": The chosen substitute (pick from Candidates if suitable, or suggest a better culturally relevant one).
        - "rationale": Brief explanation of why this transformation or preservation is chosen.
        - "confidence": Float between 0 and 1.
        """
