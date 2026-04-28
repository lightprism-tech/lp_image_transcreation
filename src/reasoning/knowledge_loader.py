import json
import logging
import os
import re
from typing import List, Dict, Optional, Union, Set, Any, Tuple
from src.reasoning.schemas import CulturalNode, CulturalKBEntry, StylePriors, SubstitutionEntry

logger = logging.getLogger(__name__)

CULTURAL_MAPPINGS_FILENAME = "cultural_mappings.json"


def _is_cultural_kb_format(data: dict) -> bool:
    """Detect if the loaded JSON is a cultural KB (culture-keyed entries with substitutions/avoid)."""
    if not data or not isinstance(data, dict):
        return False
    for v in data.values():
        if isinstance(v, dict) and ("substitutions" in v or "avoid" in v):
            return True
    return False


class KnowledgeLoader:
    """
    Loads either (1) a Cultural Knowledge Base K(c) with per-culture entries
    (substitutions, avoid, style_priors, sensitivity_notes) or (2) a legacy
    knowledge graph (nodes/edges). Provides a unified interface for the
    reasoning engine.
    """
    def __init__(self, graph_path: str):
        self.graph_path = graph_path
        self.nodes: Dict[str, Dict] = {}
        self._node_to_country: Dict[str, str] = {}
        self._country_type_index: Dict[str, Dict[str, List[Dict]]] = {}
        self._cultural_kb: Dict[str, CulturalKBEntry] = {}
        self._is_kb_format = False
        self._label_to_type: Dict[str, str] = {}
        self._preferred_substitutions: List[Dict[str, str]] = []
        self._cultural_types: Set[str] = set()
        self._embedding_components = None

        self._load(graph_path)
        self._load_cultural_mappings(graph_path)

    def _load(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error("Knowledge file not found: %s", path)
            raise
        except json.JSONDecodeError:
            logger.error("Invalid JSON in knowledge file: %s", path)
            raise

        if _is_cultural_kb_format(data):
            self._is_kb_format = True
            for culture_key, entry in data.items():
                if isinstance(entry, dict):
                    subs_raw = entry.get("substitutions", {})
                    substitutions = {}
                    for cls, lst in subs_raw.items():
                        if isinstance(lst, list):
                            substitutions[cls] = [SubstitutionEntry(**x) for x in lst]
                        else:
                            substitutions[cls] = []
                    sp = entry.get("style_priors")
                    style_priors = StylePriors(**sp) if isinstance(sp, dict) else None
                    self._cultural_kb[culture_key] = CulturalKBEntry(
                        culture=entry.get("culture", culture_key),
                        substitutions=substitutions,
                        avoid=entry.get("avoid", []),
                        style_priors=style_priors,
                        sensitivity_notes=entry.get("sensitivity_notes", []),
                    )
            logger.info("Loaded Cultural KB: %d culture(s).", len(self._cultural_kb))
        else:
            self._load_graph_data(data)

    def _load_cultural_mappings(self, graph_path: str) -> None:
        """Load label_to_type and preferred_substitutions from KB. Graph file is used first if it has them; else cultural_mappings.json in same dir."""
        if self._label_to_type or self._preferred_substitutions:
            logger.debug("Cultural mappings already loaded from graph file.")
            return
        mappings_path = os.path.join(os.path.dirname(graph_path), CULTURAL_MAPPINGS_FILENAME)
        if not os.path.isfile(mappings_path):
            logger.debug("No cultural mappings file at %s; using empty mappings.", mappings_path)
            return
        try:
            with open(mappings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._label_to_type = {k.lower(): v for k, v in (data.get("label_to_type") or {}).items()}
            self._preferred_substitutions = data.get("preferred_substitutions") or []
            logger.info("Loaded cultural mappings from file: %d label_to_type, %d preferred_substitutions.",
                        len(self._label_to_type), len(self._preferred_substitutions))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load cultural mappings from %s: %s", mappings_path, e)

    def get_cultural_types(self) -> Set[str]:
        """Return set of types in the KG that are culture-related (have per-country attributes). Excludes COUNTRY."""
        return set(self._cultural_types)

    def get_label_to_type(self) -> Dict[str, str]:
        """Return label -> cultural type mapping from KB (e.g. bicycle -> SPORT)."""
        return dict(self._label_to_type)

    def get_preferred_substitution(self, object_label: str, target_culture: str) -> Optional[str]:
        """Return preferred target_object for (object_label, target_culture) from KB, or None."""
        obj_lower = (object_label or "").lower()
        culture_lower = (target_culture or "").lower()
        for entry in self._preferred_substitutions:
            if (entry.get("object_label") or "").lower() == obj_lower and (
                entry.get("target_culture") or ""
            ).lower() == culture_lower:
                return entry.get("target_object")
        return None

    def _load_graph_data(self, data: dict) -> None:
        # 1. Load Nodes
        if "nodes" in data:
            for n in data["nodes"]:
                self.nodes[n["id"]] = n

        # 2. Load Edges and Build Indexes
        edges = data.get("edges") or data.get("links") or []
        country_nodes = {nid: n["label"] for nid, n in self.nodes.items() if n.get("type") == "COUNTRY"}

        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source in country_nodes:
                self._node_to_country[target] = country_nodes[source]
                country_label = country_nodes[source]
                target_node = self.nodes.get(target)
                if target_node:
                    t_type = target_node.get("type", "UNKNOWN")
                    if country_label not in self._country_type_index:
                        self._country_type_index[country_label] = {}
                    if t_type not in self._country_type_index[country_label]:
                        self._country_type_index[country_label][t_type] = []
                    self._country_type_index[country_label][t_type].append(target_node)
                    if t_type != "COUNTRY":
                        self._cultural_types.add(t_type)

        # Mappings may be embedded in the graph file (same as countries_graph.json schema)
        if data.get("label_to_type"):
            self._label_to_type = {k.lower(): v for k, v in data["label_to_type"].items()}
        if data.get("preferred_substitutions"):
            self._preferred_substitutions = data["preferred_substitutions"]

        logger.info("Loaded Knowledge Graph: %d nodes, %d edges.", len(self.nodes), len(edges))

    def find_node(self, label: str) -> Optional[CulturalNode]:
        """Find a node by label (case-insensitive)."""
        label_lower = label.lower()
        for node in self.nodes.values():
            if node.get("label", "").lower() == label_lower:
                return CulturalNode(**node)
        return None

    def get_culture_of_node(self, node_id: str) -> Optional[str]:
        """
        Finds the country/culture associated with a node using pre-built index.
        """
        return self._node_to_country.get(node_id)

    def get_nodes_by_type_and_culture(self, node_type: str, culture_name: str) -> List[CulturalNode]:
        """
        Returns all nodes of a specific type associated with a specific culture using pre-built index.
        """
        culture_key = None
        for k in self._country_type_index.keys():
            if k.lower() == culture_name.lower():
                culture_key = k
                break
        if not culture_key:
            return []
        nodes = self._country_type_index[culture_key].get(node_type, [])
        return [CulturalNode(**n) for n in nodes]

    # --- Cultural KB API (for K(c) format) ---

    def _culture_key(self, culture_name: str) -> Optional[str]:
        for k in self._cultural_kb.keys():
            if k.lower() == culture_name.lower():
                return k
        return None

    def get_kb_entry(self, culture_name: str) -> Optional[CulturalKBEntry]:
        """Return the full KB entry for a target culture, if using KB format."""
        key = self._culture_key(culture_name)
        return self._cultural_kb.get(key) if key else None

    def get_candidates_from_kb(self, culture_name: str, object_label: str, obj_type: str) -> List[str]:
        """
        Retrieve candidate substitutes from the KB for a given object and type.
        Returns list of target labels (empty if KB format not used or no match).
        """
        entry = self.get_kb_entry(culture_name)
        if not entry:
            return []
        subs = entry.substitutions.get(obj_type, [])
        label_lower = object_label.lower()
        for se in subs:
            if se.source.lower() == label_lower:
                return list(se.targets)
        return []

    def get_avoid_list(self, culture_name: str) -> List[str]:
        """Negative constraints (avoid list) for the target culture."""
        entry = self.get_kb_entry(culture_name)
        return list(entry.avoid) if entry else []

    def get_style_priors(self, culture_name: str) -> Optional[StylePriors]:
        """Stylistic priors (palette, motifs) for the target culture."""
        entry = self.get_kb_entry(culture_name)
        return entry.style_priors if entry else None

    def get_sensitivity_notes(self, culture_name: str) -> List[str]:
        """Sensitivity notes that discourage harmful or cliched edits."""
        entry = self.get_kb_entry(culture_name)
        return list(entry.sensitivity_notes) if entry else []

    def get_item_by_label(self, label: str, culture_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        key = (label or "").strip().lower()
        if not key:
            return None
        for node in self.nodes.values():
            node_label = str(node.get("label") or "").strip().lower()
            if node_label != key:
                continue
            node_culture = self._node_to_country.get(str(node.get("id")))
            if culture_name and node_culture and node_culture.lower() != culture_name.lower():
                continue
            return dict(node)
        return None

    def get_visual_attributes(self, label: str, obj_type: str, culture_name: str = "") -> Dict[str, str]:
        item = self.get_item_by_label(label, culture_name=culture_name) or {}
        existing = item.get("visual_attributes")
        if isinstance(existing, dict):
            attrs = {
                "shape": str(existing.get("shape") or "").strip(),
                "color": str(existing.get("color") or "").strip(),
                "texture": str(existing.get("texture") or "").strip(),
                "context": str(existing.get("context") or "").strip(),
            }
            if all(attrs.values()):
                return attrs
        return self._infer_visual_attributes(label, obj_type, culture_name)

    def _infer_visual_attributes(self, label: str, obj_type: str, culture_name: str = "") -> Dict[str, str]:
        text = re.sub(r"[^a-z0-9 ]+", " ", (label or "").lower()).strip()
        tokens = [t for t in text.split() if t]
        base_color = "multicolor"
        if any(t in text for t in ("rice", "bread", "naan", "bun", "chapati", "idli")):
            base_color = "warm beige"
        elif any(t in text for t in ("soup", "curry", "stew", "sauce", "dal")):
            base_color = "golden yellow"
        elif any(t in text for t in ("tea", "coffee")):
            base_color = "deep brown"
        shape = "recognizable local form"
        if any(t in text for t in ("ball", "dumpling", "jamun")):
            shape = "rounded"
        elif any(t in text for t in ("roll", "wrap", "pav", "sandwich")):
            shape = "elongated handheld"
        elif any(t in text for t in ("plate", "bowl", "cup")):
            shape = "container-centered"
        texture = "natural material texture"
        if (obj_type or "").upper() == "FOOD":
            texture = "fresh cooked texture"
        elif (obj_type or "").upper() in {"CLOTHING", "ART"}:
            texture = "woven textile texture"
        context = "placed naturally in the scene"
        if culture_name:
            context = f"placed naturally in a {culture_name} cultural setting"
        if tokens:
            context = f"{context}, featuring {tokens[0]}"
        return {
            "shape": shape,
            "color": base_color,
            "texture": texture,
            "context": context,
        }

    def get_scene_candidates(self, culture_name: str) -> List[Dict[str, Any]]:
        scenes: List[Dict[str, Any]] = []
        for node in self.nodes.values():
            node_type = str(node.get("type") or "").upper()
            if node_type not in {"SCENE", "PLACE", "EVENT", "ARCHITECTURE"}:
                continue
            node_id = str(node.get("id") or "")
            country = self._node_to_country.get(node_id, "")
            if culture_name and country and country.lower() != culture_name.lower():
                continue
            name = str(node.get("label") or "").strip()
            if not name:
                continue
            scenes.append(
                {
                    "name": name,
                    "tags": [node_type.lower(), culture_name.lower() if culture_name else "culture"],
                    "elements": [f"{name} details", f"{culture_name} local context".strip()],
                    "lighting": "natural",
                    "style": "vibrant",
                }
            )
        if scenes:
            return scenes
        fallback_name = f"{culture_name} street market".strip() if culture_name else "local street market"
        return [
            {
                "name": fallback_name,
                "tags": ["food", "crowded", "local"],
                "elements": ["vendors", "signboards", "people"],
                "lighting": "natural",
                "style": "vibrant",
            }
        ]

    def rank_candidates_by_embedding(self, query: str, candidates: List[str]) -> List[str]:
        if not query or not candidates:
            return candidates
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
        except Exception:
            return candidates
        try:
            if self._embedding_components is None:
                self._embedding_components = SentenceTransformer("all-MiniLM-L6-v2")
            model = self._embedding_components
            q_emb = model.encode([query])
            c_emb = model.encode(candidates)
            sims = cosine_similarity(q_emb, c_emb)[0]
            pairs: List[Tuple[float, str]] = list(zip(sims.tolist(), candidates))
            pairs.sort(key=lambda x: x[0], reverse=True)
            return [p[1] for p in pairs]
        except Exception:
            return candidates
