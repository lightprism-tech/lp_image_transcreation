import unittest
from unittest.mock import MagicMock, patch
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.reasoning.knowledge_loader import KnowledgeLoader
from src.reasoning.engine import CulturalReasoningEngine
from src.reasoning.types import ReasoningInput, TranscreationPlan
from src.reasoning.llm_client import LLMClient

class TestKnowledgeLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary dummy graph file
        self.test_graph_path = "test_graph.json"
        self.test_data = {
            "nodes": [
                {"id": "C_USA", "label": "USA", "type": "COUNTRY"},
                {"id": "C_JPN", "label": "Japan", "type": "COUNTRY"},
                {"id": "F_BURGER", "label": "Burger", "type": "FOOD"},
                {"id": "F_SUSHI", "label": "Sushi", "type": "FOOD"}
            ],
            "edges": [
                {"source": "C_USA", "target": "F_BURGER", "relation": "has_food"},
                {"source": "C_JPN", "target": "F_SUSHI", "relation": "has_food"}
            ]
        }
        with open(self.test_graph_path, "w", encoding="utf-8") as f:
            json.dump(self.test_data, f)
        
        self.loader = KnowledgeLoader(self.test_graph_path)

    def tearDown(self):
        if os.path.exists(self.test_graph_path):
            os.remove(self.test_graph_path)

    def test_find_node_valid(self):
        node = self.loader.find_node("Burger")
        self.assertIsNotNone(node)
        self.assertEqual(node.label, "Burger")
        self.assertEqual(node.type, "FOOD")

    def test_find_node_case_insensitive(self):
        node = self.loader.find_node("burger")
        self.assertIsNotNone(node)
        self.assertEqual(node.id, "F_BURGER")

    def test_find_node_invalid(self):
        node = self.loader.find_node("Pizza")
        self.assertIsNone(node)

    def test_get_culture_of_node(self):
        culture = self.loader.get_culture_of_node("F_BURGER")
        self.assertEqual(culture, "USA")
        
        culture = self.loader.get_culture_of_node("F_SUSHI")
        self.assertEqual(culture, "Japan")

    def test_get_nodes_by_type_and_culture(self):
        nodes = self.loader.get_nodes_by_type_and_culture("FOOD", "Japan")
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].label, "Sushi")
        
        nodes2 = self.loader.get_nodes_by_type_and_culture("FOOD", "USA")
        self.assertEqual(len(nodes2), 1)
        self.assertEqual(nodes2[0].label, "Burger")

class TestReasoningEngine(unittest.TestCase):
    def setUp(self):
        # Mock KnowledgeLoader
        self.mock_loader = MagicMock()
        # Mock LLMClient
        self.mock_llm = MagicMock()
        
        # Initialize Engine with mocks
        with patch('src.reasoning.engine.KnowledgeLoader', return_value=self.mock_loader), \
             patch('src.reasoning.engine.LLMClient', return_value=self.mock_llm):
            self.engine = CulturalReasoningEngine("dummy_path")
            # Re-assign mocks because __init__ creates new instances
            self.engine.kg_loader = self.mock_loader
            self.engine.llm_client = self.mock_llm

    def test_analyze_image_transformation(self):
        # Setup Inputs
        input_data = ReasoningInput(
            scene_graph={
                "objects": [{"id": 1, "label": "Burger", "type": "FOOD"}],
                "scene": {"description": "Eating lunch"}
            },
            target_culture="Japan"
        )

        # Setup Mock Behaviors
        self.mock_loader.find_node.return_value.type = "FOOD"
        self.mock_loader.get_culture_of_node.return_value = "USA"
        self.mock_loader.get_nodes_by_type_and_culture.return_value = [] # Candidates irrelevant to mock LLM

        # Mock LLM Response
        self.mock_llm.generate_reasoning.return_value = {
            "action": "transform",
            "target_object": "Sushi",
            "rationale": "Better fit",
            "confidence": 0.9
        }

        # Run Analysis
        plan = self.engine.analyze_image(input_data)

        # Assertions
        self.assertEqual(plan.target_culture, "Japan")
        self.assertEqual(len(plan.transformations), 1)
        self.assertEqual(plan.transformations[0].original_object, "Burger")
        self.assertEqual(plan.transformations[0].target_object, "Sushi")
        self.assertEqual(len(plan.preservations), 0)

    def test_analyze_image_preservation(self):
        # Setup Inputs
        input_data = ReasoningInput(
            scene_graph={
                "objects": [{"id": 1, "label": "Tree", "type": "PLANT"}]
            },
            target_culture="Japan"
        )

        # Mock LLM Response
        self.mock_llm.generate_reasoning.return_value = {
            "action": "preserve",
            "rationale": "Universal object"
        }
        
        # Mocks for KG
        self.mock_loader.find_node.return_value = None # Obj not in KG

        # Run Analysis
        plan = self.engine.analyze_image(input_data)

        # Assertions
        self.assertEqual(len(plan.preservations), 1)
        self.assertEqual(plan.preservations[0].original_object, "Tree")
        self.assertEqual(len(plan.transformations), 0)

if __name__ == "__main__":
    unittest.main()
