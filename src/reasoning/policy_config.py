"""
Reasoning policy settings loaded from reasoning.yaml (and optional env overrides).
"""
import os
from pathlib import Path
from typing import Any, List, Set

import yaml

_CONFIG_FILE = Path(__file__).resolve().parent / "config" / "reasoning.yaml"
_ENV_PREFIX = "REASONING_POLICY_"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _policy_root() -> dict:
    return _load_yaml(_CONFIG_FILE).get("policy") or {}


def _walk(path: str) -> Any:
    node: Any = _policy_root()
    for key in (path or "").split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _env_override(path: str) -> Any:
    env_name = _ENV_PREFIX + path.replace(".", "_").upper()
    raw = os.getenv(env_name)
    if raw is None:
        return None
    text = raw.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def get_policy(path: str) -> Any:
    env_val = _env_override(path)
    if env_val is not None:
        return env_val
    val = _walk(path)
    if val is None:
        raise KeyError(f"Missing reasoning policy key: policy.{path}")
    return val


def get_policy_list(path: str) -> List[str]:
    val = get_policy(path)
    if not isinstance(val, list):
        raise TypeError(f"policy.{path} must be a list")
    return [str(item).strip() for item in val if str(item).strip()]


def get_policy_int(path: str) -> int:
    return int(get_policy(path))


def get_policy_float(path: str) -> float:
    return float(get_policy(path))


def get_policy_set(path: str) -> Set[str]:
    return {str(item).strip().upper() for item in get_policy_list(path) if str(item).strip()}


def get_policy_dict(path: str) -> dict:
    val = get_policy(path)
    if not isinstance(val, dict):
        raise TypeError(f"policy.{path} must be a mapping")
    parsed: dict = {}
    for key, value in val.items():
        key_text = str(key or "").strip().lower()
        value_text = str(value or "").strip().upper()
        if key_text and value_text:
            parsed[key_text] = value_text
    return parsed
