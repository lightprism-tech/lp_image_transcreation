"""Pytest for reasoning knowledge_loader: one test per method."""
import json
import os
import pytest
from src.reasoning.knowledge_loader import KnowledgeLoader
from src.reasoning.schemas import CulturalNode


@pytest.fixture
def temp_kg_path(tmp_path):
    data = {
        "nodes": [
            {"id": "C_USA", "label": "USA", "type": "COUNTRY"},
            {"id": "C_IND", "label": "India", "type": "COUNTRY"},
            {"id": "F_BURGER", "label": "Burger", "type": "FOOD"},
            {"id": "F_SUSHI", "label": "Sushi", "type": "FOOD"},
            {"id": "F_CURRY", "label": "Curry", "type": "FOOD"},
        ],
        "edges": [
            {"source": "C_USA", "target": "F_BURGER"},
            {"source": "C_IND", "target": "F_CURRY"},
            {"source": "C_IND", "target": "F_SUSHI"},
        ],
    }
    path = tmp_path / "kg.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_knowledge_loader_init_loads_graph(temp_kg_path):
    loader = KnowledgeLoader(temp_kg_path)
    assert len(loader.nodes) == 5


def test_knowledge_loader_init_raises_on_missing_file():
    import pytest
    with pytest.raises(FileNotFoundError):
        KnowledgeLoader("nonexistent_file_12345.json")


def test_knowledge_loader_find_node_by_label(temp_kg_path):
    loader = KnowledgeLoader(temp_kg_path)
    node = loader.find_node("Burger")
    assert node is not None
    assert node.label == "Burger"
    assert node.type == "FOOD"
    assert node.id == "F_BURGER"


def test_knowledge_loader_find_node_case_insensitive(temp_kg_path):
    loader = KnowledgeLoader(temp_kg_path)
    node = loader.find_node("burger")
    assert node is not None
    assert node.id == "F_BURGER"


def test_knowledge_loader_find_node_missing_returns_none(temp_kg_path):
    loader = KnowledgeLoader(temp_kg_path)
    assert loader.find_node("Pizza") is None


def test_knowledge_loader_get_culture_of_node(temp_kg_path):
    loader = KnowledgeLoader(temp_kg_path)
    assert loader.get_culture_of_node("F_BURGER") == "USA"
    assert loader.get_culture_of_node("F_CURRY") == "India"


def test_knowledge_loader_get_culture_of_node_unknown_returns_none(temp_kg_path):
    loader = KnowledgeLoader(temp_kg_path)
    assert loader.get_culture_of_node("unknown_id") is None


def test_knowledge_loader_get_nodes_by_type_and_culture(temp_kg_path):
    loader = KnowledgeLoader(temp_kg_path)
    nodes = loader.get_nodes_by_type_and_culture("FOOD", "India")
    labels = [n.label for n in nodes]
    assert "Curry" in labels
    assert "Sushi" in labels
    assert len(nodes) == 2


def test_knowledge_loader_get_nodes_by_type_and_culture_case_insensitive(temp_kg_path):
    loader = KnowledgeLoader(temp_kg_path)
    nodes = loader.get_nodes_by_type_and_culture("FOOD", "india")
    assert len(nodes) == 2


def test_knowledge_loader_get_nodes_by_type_and_culture_unknown_culture(temp_kg_path):
    loader = KnowledgeLoader(temp_kg_path)
    nodes = loader.get_nodes_by_type_and_culture("FOOD", "Mars")
    assert nodes == []


def test_knowledge_loader_get_nodes_by_type_and_culture_unknown_type(temp_kg_path):
    loader = KnowledgeLoader(temp_kg_path)
    nodes = loader.get_nodes_by_type_and_culture("VEHICLE", "India")
    assert nodes == []


def test_knowledge_loader_links_uses_links_key(tmp_path):
    """Graph with 'links' instead of 'edges'."""
    data = {
        "nodes": [
            {"id": "C_X", "label": "X", "type": "COUNTRY"},
            {"id": "O1", "label": "Obj", "type": "OBJECT"},
        ],
        "links": [{"source": "C_X", "target": "O1"}],
    }
    path = tmp_path / "kg.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    loader = KnowledgeLoader(str(path))
    assert loader.get_culture_of_node("O1") == "X"
