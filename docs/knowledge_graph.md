# Knowledge Graph Data

This document contains documentation for the Knowledge Graph component of the Image Transcreation Pipeline.

## Dataset Overview

The knowledge graph dataset consists of cultural attributes for 20 countries.

- **Nodes**: 1339 (20 Countries + 1319 Cultural Attributes)
- **Edges**: 1500 (75 attributes connected per country)
- **Node Types**: `COUNTRY`, `FOOD`, `SPORT`, `CLOTHING`, `ARCHITECTURE`, `CLIMATE`, `LIVING_STYLE`, `HOME_FEATURE`, `OUTDOOR_ACTIVITY`, etc.

### Schema Change
The graph now uses a **Node-based architecture** for attributes.
- **Country Nodes**: `id: "C_IND"`, `type: "COUNTRY"`
- **Attribute Nodes**: `id: "F_SUSHI"`, `type: "FOOD"`, `id: "K_TROPICAL"`, `type: "CLIMATE"`
- **Edges**: Connect Country Nodes to Attribute Nodes.

### Data Source
Country data is now stored in `scripts/knowledge_graph/countries.json` for easier editing.

### Countries Included
1.  **India** (IND)
2.  **United States** (USA)
3.  **Japan** (JPN)
4.  **France** (FRA)
5.  **Italy** (ITA)
6.  **Brazil** (BRA)
7.  **China** (CHN)
8.  **Germany** (DEU)
9.  **Australia** (AUS)
10. **Egypt** (EGY)
11. **Canada** (CAN)
12. **Russia** (RUS)
13. **South Africa** (ZAF)
14. **Mexico** (MEX)
15. **Argentina** (ARG)
16. **Spain** (ESP)
17. **Turkey** (TUR)
18. **Thailand** (THA)
19. **South Korea** (KOR)
20. **Greece** (GRC)

## Data Generation

The data is generated using a Python script that contains hardcoded cultural attributes for each country.

### Script Location
`scripts/knowledge_graph/generator.py`

### Usage
To regenerate the knowledge graph data, run the following command from the project root:

```bash
python scripts/knowledge_graph/generator.py
```

## Output Formats

The generation script produces the knowledge graph in three formats in the `data/knowledge_base/` directory:

1.  **JSON** (`countries_graph.json`)
    - Standard JSON format with `nodes` and `edges` arrays.
    
    ```json
    {
      "nodes": [{"id": "IND", "label": "India", "type": "COUNTRY"}, ...],
      "edges": [{"source": "IND", "relation": "food", "value": "Chapati"}, ...]
    }
    ```

2.  **Pickle** (`countries_graph.pkl`)
    - Python pickle serialization of the same dictionary structure as the JSON file. Useful for quick loading in Python environments.

3.  **JSONL** (`countries_graph.jsonl`)
    - Line-delimited JSON. Each line is a valid JSON object representing either a node or an edge.
    
    ```jsonl
    {"id": "IND", "label": "India", "type": "COUNTRY"}
    ...
    {"source": "IND", "relation": "food", "value": "Chapati"}
    ...
    ```
