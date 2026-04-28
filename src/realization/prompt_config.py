from pathlib import Path
from typing import Any, Dict

import yaml

_PROMPT_FILE = Path(__file__).resolve().parent / "config" / "realization.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_PROMPTS = _load_yaml(_PROMPT_FILE).get("prompts", {})


def get_prompt(path: str, default: str = "") -> str:
    node: Any = _PROMPTS
    for key in (path or "").split("."):
        if not isinstance(node, dict):
            return default
        node = node.get(key)
    return str(node) if isinstance(node, str) else default


def get_prompt_list(path: str, default: list | None = None) -> list:
    default = default or []
    node: Any = _PROMPTS
    for key in (path or "").split("."):
        if not isinstance(node, dict):
            return list(default)
        node = node.get(key)
    return list(node) if isinstance(node, list) else list(default)
