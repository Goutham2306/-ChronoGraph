"""
Builds the ChronoGraph graph JSON ({"nodes": [], "edges": []}) from the flat
list of extracted relationship triples produced by pipeline.extract.

Nodes are deduplicated by (name, type). Edges preserve every field required
for temporal retrieval and citations: timestamp, source, source_id, evidence,
confidence.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_PATH = OUTPUT_DIR / "graph.json"


def build_graph(triples: List[dict]) -> dict:
    node_index: Dict[Tuple[str, str], dict] = {}
    edges: List[dict] = []

    def _get_or_create_node(name: str, node_type: str) -> str:
        key = (name.strip(), node_type)
        if key not in node_index:
            node_id = f"{node_type}:{name.strip()}"
            node_index[key] = {"id": node_id, "name": name.strip(), "type": node_type}
        return node_index[key]["id"]

    for i, t in enumerate(triples):
        subject_id = _get_or_create_node(t["subject"], t["subject_type"])
        object_id = _get_or_create_node(t["object"], t["object_type"])

        edges.append(
            {
                "id": f"edge:{i}",
                "subject": subject_id,
                "predicate": t["predicate"],
                "object": object_id,
                "timestamp": t["timestamp"],
                "source": t["source"],
                "source_id": t["source_id"],
                "evidence": t["evidence"],
                "confidence": t["confidence"],
            }
        )

    graph = {
        "nodes": list(node_index.values()),
        "edges": edges,
    }
    return graph


def save_graph(graph: dict, path: Path = OUTPUT_PATH) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(graph, f, indent=2)
    return path


if __name__ == "__main__":
    # Quick manual test with a couple of hand-built triples.
    sample_triples = [
        {
            "subject": "Rohit Nair",
            "subject_type": "Person",
            "predicate": "ADVOCATED_FOR",
            "object": "GCP",
            "object_type": "Technology",
            "evidence": "advocating we move forward: migrate analytics workloads",
            "confidence": 0.95,
            "timestamp": "2023-03-02",
            "source": "slack",
            "source_id": "SLACK-008",
        }
    ]
    g = build_graph(sample_triples)
    print(json.dumps(g, indent=2))
