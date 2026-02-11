
import json
import os
import pickle

def load_countries_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "countries.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_knowledge_graph():
    countries = load_countries_data()

    # Valid categories and their prefixes for ID generation
    category_prefixes = {
        "food": "F",
        "sport": "S",
        "clothing": "C",
        "architecture": "A",
        "landmark": "L",
        "currency": "Y",  # C is taken by Clothing, Y for CurrencY
        "language": "G",  # L is taken by Landmark, G for LanGuage
        "animal": "N",    # A is taken by Architecture, N for aNimal
        "festival": "E",  # F is taken by Food, E for fEstival
        "dance": "D",
        "history": "H",
        "geography": "R", # G is taken, R for geogRaphy
        "music": "M",
        "art": "T",       # A is taken, T for arT
        "technology": "X",
        "religion": "B",
        "mythology": "O",
        "literature": "W", # Write
        "philosophy": "P",
        "climate": "K",
        "living_style": "V",
        "home_feature": "I",
        "outdoor_activity": "Z"
    }

    # Dictionary to store unique nodes (keyed by ID) to avoid duplicates
    nodes_dict = {}
    edges = []

    for country in countries:
        # Create Country Node
        country_id = f"C_{country['id']}"  # e.g., C_IND
        
        if country_id not in nodes_dict:
            nodes_dict[country_id] = {
                "id": country_id,
                "label": country["label"],
                "type": "COUNTRY"
            }

        # Process Attributes
        attrs = country["attributes"]
        
        # We target 50 edges per country
        edges_created_count = 0
        
        # Flatten attributes to list of (category, value)
        all_attributes = []
        for category, values in attrs.items():
            for value in values:
                all_attributes.append((category, value))
        
        # Take first 75 attributes to create edges for
        # With new categories (climate, living_style, etc.), we have more data.
        # Increasing limit toensure we capture these new diverse attributes.
        
        selected_attributes = all_attributes[:75]
        
        # If less than 75, pad with generic features? 
        # The prompt implies "add in doc folder", "add more edges", "node formte".
        # I will just process up to 75 unique attributes.
        
        for category, value in selected_attributes:
            # Generate Attribute Node ID
            # Clean value for ID: UPPERCASE, spaces to underscores, remove special chars
            clean_value = "".join(c for c in value if c.isalnum() or c.isspace()).strip().upper().replace(" ", "_")
            
            prefix = category_prefixes.get(category, "U") # U for Unknown
            node_id = f"{prefix}_{clean_value}"
            
            # Create Attribute Node if not exists
            if node_id not in nodes_dict:
                nodes_dict[node_id] = {
                    "id": node_id,
                    "label": value,
                    "type": category.upper()
                }
            
            # Create Edge
            edges.append({
                "source": country_id,
                "target": node_id,
                "relation": category
            })
            edges_created_count += 1
            
        # Pad if needed (though with updated lists, likely not needed)
        while edges_created_count < 75:
             # Create a generic dummy node
             dummy_val = f"Feature {edges_created_count + 1}"
             dummy_id = f"U_{country['id']}_{edges_created_count}"
             
             if dummy_id not in nodes_dict:
                 nodes_dict[dummy_id] = {
                     "id": dummy_id,
                     "label": dummy_val,
                     "type": "UNKNOWN"
                 }
             
             edges.append({
                 "source": country_id,
                 "target": dummy_id,
                 "relation": "generic"
             })
             edges_created_count += 1

    # Convert nodes dict back to list
    nodes = list(nodes_dict.values())

    output_data = {
        "nodes": nodes,
        "edges": edges
    }

    output_dir = "data/knowledge_base"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    output_file_json = os.path.join(output_dir, "countries_graph.json")
    with open(output_file_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Save as Pickle
    output_file_pkl = os.path.join(output_dir, "countries_graph.pkl")
    with open(output_file_pkl, 'wb') as f:
        pickle.dump(output_data, f)

    # Save as JSONL
    output_file_jsonl = os.path.join(output_dir, "countries_graph.jsonl")
    with open(output_file_jsonl, 'w') as f:
        for node in nodes:
            f.write(json.dumps(node) + '\n')
        # Edges in JSONL might need to be clearly distinguished or just mixed in?
        # The user example showed nodes. Edges usually link nodes. 
        # I will dump edges as well.
        for edge in edges:
            f.write(json.dumps(edge) + '\n')
    
    print(f"Successfully generated knowledge graph data:")
    print(f"  - JSON: {output_file_json}")
    print(f"  - Pickle: {output_file_pkl}")
    print(f"  - JSONL: {output_file_jsonl}")
    print(f"Total Nodes: {len(nodes)}")
    print(f"Total Edges: {len(edges)}")

if __name__ == "__main__":
    generate_knowledge_graph()
