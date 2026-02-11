import json
import logging
from typing import List, Dict, Optional
from src.reasoning.types import CulturalNode

logger = logging.getLogger(__name__)

class KnowledgeLoader:
    def __init__(self, graph_path: str):
        self.graph_path = graph_path
        self.nodes: Dict[str, Dict] = {}
        # Indexes for O(1) lookup
        self._node_to_country: Dict[str, str] = {} # node_id -> country_label
        self._country_type_index: Dict[str, Dict[str, List[Dict]]] = {} # country_label -> {type -> [nodes]}
        
        self._load_graph()

    def _load_graph(self):
        try:
            with open(self.graph_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 1. Load Nodes
                if "nodes" in data:
                    for n in data["nodes"]:
                        self.nodes[n["id"]] = n
                
                # 2. Load Edges and Build Indexes
                edges = []
                if "edges" in data:
                    edges = data["edges"]
                elif "links" in data:
                    edges = data["links"]
                
                # First pass: Identify all Country nodes to be sure
                country_nodes = {nid: n["label"] for nid, n in self.nodes.items() if n.get("type") == "COUNTRY"}
                
                # Build Indexes
                for edge in edges:
                    source = edge.get("source")
                    target = edge.get("target")
                    
                    # Index: Node -> Culture (parent country)
                    if source in country_nodes:
                        self._node_to_country[target] = country_nodes[source]
                        
                        # Index: Culture -> Type -> [Nodes]
                        country_label = country_nodes[source]
                        target_node = self.nodes.get(target)
                        
                        if target_node:
                            t_type = target_node.get("type", "UNKNOWN")
                            
                            if country_label not in self._country_type_index:
                                self._country_type_index[country_label] = {}
                            if t_type not in self._country_type_index[country_label]:
                                self._country_type_index[country_label][t_type] = []
                            
                            self._country_type_index[country_label][t_type].append(target_node)

            logger.info(f"Loaded Knowledge Graph: {len(self.nodes)} nodes, {len(edges)} edges.")
            
        except FileNotFoundError:
            logger.error(f"Knowledge Graph file not found: {self.graph_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in Knowledge Graph: {self.graph_path}")
            raise

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
        # Iterate to find exact case match for culture key if needed, or normalize keys
        # For efficiency, let's assume culture_name might not match case perfectly, 
        # but our index keys come from the graph labels. 
        # We can try direct lookup first.
        
        # Helper to find case-insensitive key
        culture_key = None
        for k in self._country_type_index.keys():
            if k.lower() == culture_name.lower():
                culture_key = k
                break
        
        if not culture_key:
            return []
            
        nodes = self._country_type_index[culture_key].get(node_type, [])
        return [CulturalNode(**n) for n in nodes]
