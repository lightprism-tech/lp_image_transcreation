"""
Load and merge realization configuration from YAML defaults, JSON overrides, and env.
"""
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULTS_PATH = Path(__file__).resolve().parent / "config" / "defaults.yaml"

_ENV_PREFIX = "REALIZATION_"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (overlay or {}).items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _env_key_to_path(env_name: str) -> Optional[str]:
    """
    Map REALIZATION_ARTIFACT_GATE__MIN_MEAN_ABS_CHANGE -> artifact_gate.min_mean_abs_change
    (double underscore separates nesting levels).
    """
    if not env_name.startswith(_ENV_PREFIX):
        return None
    tail = env_name[len(_ENV_PREFIX) :]
    if not tail:
        return None
    parts = [p.lower() for p in tail.split("__") if p]
    if not parts:
        return None
    return ".".join(parts)


def _coerce_env_value(raw: str) -> Any:
    text = (raw or "").strip()
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _set_nested(config: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    node = config
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def _env_overrides() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for env_name, raw in os.environ.items():
        path = _env_key_to_path(env_name)
        if not path:
            continue
        _set_nested(out, path, _coerce_env_value(raw))
    return out


def load_realization_config(user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge YAML defaults, user JSON config, and REALIZATION_* environment overrides.

    Precedence (lowest to highest): defaults.yaml < user_config < environment variables.
    """
    defaults = _load_yaml(_DEFAULTS_PATH)
    merged = _deep_merge(defaults, user_config or {})
    return _deep_merge(merged, _env_overrides())


def section_value(config: Dict[str, Any], section: str, key: str) -> Any:
    """Read a required value from a config section (no code-level fallback)."""
    block = config.get(section)
    if not isinstance(block, dict):
        raise KeyError(f"Missing config section: {section}")
    if key not in block:
        raise KeyError(f"Missing config key: {section}.{key}")
    return block[key]
