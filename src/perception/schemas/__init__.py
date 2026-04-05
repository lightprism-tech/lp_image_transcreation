"""
JSON Schema Definitions for Perception Pipeline

This module contains JSON schemas for validating:
- object_schema.json: Individual object detection results
- scene_schema.json: Complete scene perception output

Schemas can be accessed via settings:
    from perception.config import settings
    
    object_schema_path = settings.OBJECT_SCHEMA_PATH
    scene_schema_path = settings.SCENE_SCHEMA_PATH
"""

import json
from pathlib import Path

# Schema file paths
SCHEMA_DIR = Path(__file__).parent
OBJECT_SCHEMA_PATH = SCHEMA_DIR / "object_schema.json"
SCENE_SCHEMA_PATH = SCHEMA_DIR / "scene_schema.json"


def load_schema(schema_name: str) -> dict:
    """
    Load a JSON schema by name.
    
    Args:
        schema_name: Name of schema ('object' or 'scene')
    
    Returns:
        Loaded JSON schema as dictionary
    
    Raises:
        FileNotFoundError: If schema file doesn't exist
        ValueError: If schema_name is invalid
    """
    if schema_name == "object":
        schema_path = OBJECT_SCHEMA_PATH
    elif schema_name == "scene":
        schema_path = SCENE_SCHEMA_PATH
    else:
        raise ValueError(f"Invalid schema name: {schema_name}. Use 'object' or 'scene'")
    
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)


__all__ = [
    'SCHEMA_DIR',
    'OBJECT_SCHEMA_PATH',
    'SCENE_SCHEMA_PATH',
    'load_schema'
]
