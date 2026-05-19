import json
import logging
import os
import pickle
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

REL_ASSOCIATED_WITH = "ASSOCIATED_WITH"
REL_PART_OF = "PART_OF"

# Limit attribute edges per culture scope (override with KG_MAX_ATTRS_PER_CULTURE).
_MAX_ATTR_EDGES_PER_CULTURE_NODE = int(
    os.environ.get("KG_MAX_ATTRS_PER_CULTURE", "75")
)


def load_countries_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "countries.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _category_prefixes() -> Dict[str, str]:
    return {
        "food": "F",
        "sport": "S",
        "clothing": "C",
        "architecture": "A",
        "landmark": "L",
        "currency": "Y",
        "language": "G",
        "animal": "N",
        "festival": "E",
        "dance": "D",
        "history": "H",
        "geography": "R",
        "music": "M",
        "art": "T",
        "technology": "X",
        "religion": "B",
        "mythology": "O",
        "literature": "W",
        "philosophy": "P",
        "climate": "K",
        "living_style": "V",
        "home_feature": "I",
        "outdoor_activity": "Z",
    }


def _clean_token(value: str) -> str:
    return "".join(c for c in value if c.isalnum() or c.isspace()).strip().upper().replace(" ", "_")


def _add_country_node(nodes_dict: Dict[str, Dict], country: Dict[str, Any]) -> str:
    country_id = f"C_{country['id']}"
    if country_id not in nodes_dict:
        nodes_dict[country_id] = {
            "id": country_id,
            "label": country["label"],
            "type": "COUNTRY",
        }
    return country_id


def _add_culture_node(
    nodes_dict: Dict[str, Dict],
    culture_id: str,
    label: str,
    country_code: str,
    country_label: str,
) -> None:
    if culture_id in nodes_dict:
        return
    nodes_dict[culture_id] = {
        "id": culture_id,
        "label": label,
        "type": "CULTURE",
        "country_id": country_code,
        "country_label": country_label,
    }


def _flatten_attributes(attrs: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for category, values in (attrs or {}).items():
        if not isinstance(values, list):
            continue
        for value in values:
            if value is None:
                continue
            items.append((category, str(value).strip()))
    return items


def _emit_associated_edges(
    nodes_dict: Dict[str, Dict],
    edges: List[Dict[str, Any]],
    category_prefixes: Dict[str, str],
    scope_id: str,
    attrs: Dict[str, List[str]],
    max_edges: int,
) -> None:
    all_attributes = _flatten_attributes(attrs)
    selected = all_attributes[:max_edges]
    created = 0
    for category, value in selected:
        if not value:
            continue
        clean_value = _clean_token(value)
        prefix = category_prefixes.get(category, "U")
        node_id = f"{prefix}_{clean_value}"
        if node_id not in nodes_dict:
            nodes_dict[node_id] = {
                "id": node_id,
                "label": value,
                "type": category.upper(),
            }
        edges.append(
            {
                "source": scope_id,
                "target": node_id,
                "relation": REL_ASSOCIATED_WITH,
                "domain": category,
            }
        )
        created += 1
    while created < max_edges:
        dummy_val = f"Feature {created + 1}"
        dummy_id = f"U_{scope_id}_{created}"
        if dummy_id not in nodes_dict:
            nodes_dict[dummy_id] = {
                "id": dummy_id,
                "label": dummy_val,
                "type": "UNKNOWN",
            }
        edges.append(
            {
                "source": scope_id,
                "target": dummy_id,
                "relation": REL_ASSOCIATED_WITH,
                "domain": "generic",
            }
        )
        created += 1


def _traverse_cultures(
    nodes_dict: Dict[str, Dict],
    edges: List[Dict[str, Any]],
    category_prefixes: Dict[str, str],
    country: Dict[str, Any],
    country_id: str,
    spec: Dict[str, Any],
    parent_scope_id: str,
    path_prefix: str,
) -> None:
    raw_id = str(spec.get("id") or "CULT").strip().upper()
    cult_id = f"CU_{path_prefix}_{raw_id}"
    label = str(spec.get("label") or raw_id)
    _add_culture_node(nodes_dict, cult_id, label, country["id"], country["label"])
    edges.append(
        {
            "source": cult_id,
            "target": parent_scope_id,
            "relation": REL_PART_OF,
        }
    )
    nested_attrs = spec.get("attributes") or {}
    _emit_associated_edges(
        nodes_dict,
        edges,
        category_prefixes,
        cult_id,
        nested_attrs,
        _MAX_ATTR_EDGES_PER_CULTURE_NODE,
    )
    for child in spec.get("cultures") or []:
        if isinstance(child, dict):
            _traverse_cultures(
                nodes_dict,
                edges,
                category_prefixes,
                country,
                country_id,
                child,
                parent_scope_id=cult_id,
                path_prefix=f"{path_prefix}_{raw_id}",
            )


def generate_knowledge_graph():
    countries = load_countries_data()
    category_prefixes = _category_prefixes()

    nodes_dict: Dict[str, Dict] = {}
    edges: List[Dict[str, Any]] = []

    for country in countries:
        country_id = _add_country_node(nodes_dict, country)
        code = str(country["id"]).strip().upper()
        root_culture_id = f"CU_{code}_ROOT"
        _add_culture_node(
            nodes_dict,
            root_culture_id,
            f"{country['label']} (national)",
            country["id"],
            country["label"],
        )
        edges.append(
            {
                "source": root_culture_id,
                "target": country_id,
                "relation": REL_PART_OF,
            }
        )
        attrs = country.get("attributes") or {}
        _emit_associated_edges(
            nodes_dict,
            edges,
            category_prefixes,
            root_culture_id,
            attrs,
            _MAX_ATTR_EDGES_PER_CULTURE_NODE,
        )
        for child in country.get("cultures") or []:
            if isinstance(child, dict):
                _traverse_cultures(
                    nodes_dict,
                    edges,
                    category_prefixes,
                    country,
                    country_id,
                    child,
                    parent_scope_id=root_culture_id,
                    path_prefix=code,
                )

    nodes = list(nodes_dict.values())
    output_data = {
        "nodes": nodes,
        "edges": edges,
        "schema_version": "2",
        "relation_types": [REL_ASSOCIATED_WITH, REL_PART_OF],
    }

    output_dir = "data/knowledge_base"
    os.makedirs(output_dir, exist_ok=True)

    output_file_json = os.path.join(output_dir, "countries_graph.json")
    with open(output_file_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    output_file_pkl = os.path.join(output_dir, "countries_graph.pkl")
    with open(output_file_pkl, "wb") as f:
        pickle.dump(output_data, f)

    output_file_jsonl = os.path.join(output_dir, "countries_graph.jsonl")
    with open(output_file_jsonl, "w", encoding="utf-8") as f:
        for node in nodes:
            f.write(json.dumps(node) + "\n")
        for edge in edges:
            f.write(json.dumps(edge) + "\n")

    logger.info("Successfully generated knowledge graph data:")
    logger.info("  - JSON: %s", output_file_json)
    logger.info("  - Pickle: %s", output_file_pkl)
    logger.info("  - JSONL: %s", output_file_jsonl)
    logger.info("Total Nodes: %s", len(nodes))
    logger.info("Total Edges: %s", len(edges))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    generate_knowledge_graph()
