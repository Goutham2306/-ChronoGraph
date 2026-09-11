"""
Loads output/graph.json into Neo4j.

Run:
    python -m pipeline.neo4j_loader

CRITICAL DESIGN NOTE (see Day 2 spec, Section 1):
A naive `MERGE (s)-[r:PREDICATE]->(o)` would collapse every historical event
between the same two entities into a single relationship — e.g. all 3+
separate ADVOCATED_FOR events between "Person:Rohit Nair" and
"Technology:GCP" would become one edge, destroying the temporal history
that's the entire point of ChronoGraph.

Instead, each relationship is MERGEd on an `event_key` built from
`source_id + "|" + timestamp`, which is unique per historical event in the
source data. This means:
  - Running the loader twice does NOT duplicate events (idempotent).
  - Multiple distinct events between the same two entities remain
    separately retrievable relationships.

Nodes are MERGEd on their existing `id` field from graph.json
(e.g. "Person:Rohit Nair"), which is already stable and unique per the
Day 1 graph_builder.py logic (verified during inspection).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

GRAPH_JSON_PATH = Path(__file__).resolve().parent.parent / "output" / "graph.json"

_CYPHER_MERGE_NODE_TEMPLATE = """
MERGE (n {{id: $id}})
SET n.name = $name
SET n:{label}
"""

# We can't parametrize a relationship TYPE in Cypher, so predicate must be
# validated against the schema allow-list before this template is filled in.
_CYPHER_MERGE_EDGE_TEMPLATE = """
MATCH (s {{id: $subject_id}})
MATCH (o {{id: $object_id}})
MERGE (s)-[r:{predicate} {{event_key: $event_key}}]->(o)
SET r.timestamp = $timestamp,
    r.source = $source,
    r.source_id = $source_id,
    r.evidence = $evidence,
    r.confidence = $confidence
"""

ALLOWED_LABELS = {"Person", "Technology", "Project"}
ALLOWED_PREDICATES = {
    "ADVOCATED_FOR",
    "COMMITTED_CODE",
    "ARGUED_AGAINST",
    "USED",
    "REPORTED",
    "REVIEWED",
}


class Neo4jConfigError(Exception):
    """Raised when required Neo4j env vars are missing."""


def _load_graph_json() -> dict:
    if not GRAPH_JSON_PATH.exists():
        raise FileNotFoundError(
            f"{GRAPH_JSON_PATH} not found. Run `python run_day1.py` first "
            "to generate it."
        )
    with open(GRAPH_JSON_PATH) as f:
        return json.load(f)


def _get_driver():
    uri = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")

    missing = [
        name
        for name, val in [
            ("NEO4J_URI", uri),
            ("NEO4J_USERNAME", username),
            ("NEO4J_PASSWORD", password),
        ]
        if not val
    ]
    if missing:
        raise Neo4jConfigError(
            f"Missing required Neo4j env var(s): {', '.join(missing)}. "
            "Set them in .env (see .env.example)."
        )

    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        raise Neo4jConfigError(
            f"Could not connect to Neo4j at {uri}: {exc}\n"
            "Check that Neo4j is running and NEO4J_URI/USERNAME/PASSWORD "
            "are correct."
        ) from exc
    return driver


def load_graph_into_neo4j(graph: dict | None = None, verbose: bool = True) -> dict:
    """
    Loads nodes then edges into Neo4j. Idempotent: safe to re-run.

    Returns a summary dict: {"nodes_loaded": int, "edges_loaded": int}.
    """
    graph = graph if graph is not None else _load_graph_json()
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = _get_driver()
    nodes_loaded = 0
    edges_loaded = 0

    try:
        with driver.session(database=database) as session:
            for node in graph["nodes"]:
                if node["type"] not in ALLOWED_LABELS:
                    if verbose:
                        print(f"  ! skipping node with unknown label: {node}")
                    continue
                cypher = _CYPHER_MERGE_NODE_TEMPLATE.format(label=node["type"])
                session.run(cypher, id=node["id"], name=node["name"])
                nodes_loaded += 1

            for edge in graph["edges"]:
                if edge["predicate"] not in ALLOWED_PREDICATES:
                    if verbose:
                        print(f"  ! skipping edge with unknown predicate: {edge}")
                    continue
                event_key = f"{edge['source_id']}|{edge['timestamp']}|{edge['id']}"
                cypher = _CYPHER_MERGE_EDGE_TEMPLATE.format(
                    predicate=edge["predicate"]
                )
                session.run(
                    cypher,
                    subject_id=edge["subject"],
                    object_id=edge["object"],
                    event_key=event_key,
                    timestamp=edge["timestamp"],
                    source=edge["source"],
                    source_id=edge["source_id"],
                    evidence=edge["evidence"],
                    confidence=edge["confidence"],
                )
                edges_loaded += 1
    finally:
        driver.close()

    if verbose:
        print(f"Loaded {nodes_loaded} nodes and {edges_loaded} edges into Neo4j.")

    return {"nodes_loaded": nodes_loaded, "edges_loaded": edges_loaded}


if __name__ == "__main__":
    print("=" * 60)
    print("Loading output/graph.json into Neo4j")
    print("=" * 60)
    try:
        result = load_graph_into_neo4j()
        print(f"\nDone. {result}")
    except Neo4jConfigError as exc:
        print(f"\nNeo4j configuration error:\n{exc}")
        raise SystemExit(1)
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        raise SystemExit(1)
