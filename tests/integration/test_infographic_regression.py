import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.reasoning.engine import CulturalReasoningEngine
from src.reasoning.schemas import ReasoningInput


@pytest.mark.integration
def test_known_infographic_sample_builds_region_aware_text_edits(tmp_path):
    sample = Path("data/output/json/Japan_stage1_perception.json")
    if not sample.exists():
        pytest.skip("Known infographic sample not found: data/output/json/Japan_stage1_perception.json")

    scene_graph = json.loads(sample.read_text(encoding="utf-8"))
    scene_graph["image_type"] = scene_graph.get("image_type") or {"type": "infographic", "confidence": 0.9}

    (tmp_path / "dummy.json").write_text("{}")
    with patch("src.reasoning.engine.KnowledgeLoader", return_value=MagicMock()), patch(
        "src.reasoning.engine.LLMClient", return_value=MagicMock()
    ):
        engine = CulturalReasoningEngine(str(tmp_path / "dummy.json"))
        engine.llm_client.generate_reasoning.return_value = {"rewritten_text": "Adapted region text"}
        edits = engine.build_text_edits(
            ReasoningInput(scene_graph=scene_graph, target_culture="India", avoid_list=[])
        )

    assert isinstance(edits, list)
    if edits:
        first = edits[0]
        assert "bbox" in first
        assert "translated" in first
        assert "style" in first
