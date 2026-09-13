"""
Graph validation.

Checks output/graph.json (or any graph dict of the same shape) against
every requirement in the Week 1 spec:
  - all nodes have valid types
  - all nodes have IDs
  - all edges have valid predicates
  - subject exists (as a real node id)
  - object exists (as a real node id)
  - timestamp exists
  - source exists
  - source_id exists
  - evidence exists
  - confidence is between 0 and 1

This is intentionally separate from pipeline/neo4j_loader.py, which
currently just skips and warns on unknown labels/predicates when loading
into Neo4j. This module is the hard-fail gate: it's meant to run right
after graph_builder.py produces a graph, before anything is trusted
downstream, and it collects every problem it finds rather than stopping at
the first one, so a caller gets the full picture in one pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ALLOWED_NODE_TYPES = {"Person", "Technology", "Project"}
ALLOWED_PREDICATES = {
    "ADVOCATED_FOR",
    "COMMITTED_CODE",
    "ARGUED_AGAINST",
    "USED",
    "REPORTED",
    "REVIEWED",
}

_REQUIRED_EDGE_FIELDS = (
    "subject", "predicate", "object", "timestamp",
    "source", "source_id", "evidence", "confidence",
)


class GraphValidationError(Exception):
    """
    Raised by validate_graph_or_raise() when the graph is invalid. Carries
    the full list of problems found, not just the first one.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        message = f"Graph validation failed with {len(errors)} error(s):\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        super().__init__(message)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0

    def __bool__(self) -> bool:
        return self.is_valid


def validate_graph(graph: dict) -> ValidationResult:
    """
    Validates a graph dict shaped like {"nodes": [...], "edges": [...]}.
    Never raises - collects every problem found and returns them all in
    ValidationResult.errors. Use validate_graph_or_raise() for a hard-fail
    entrypoint (e.g. in run_day1.py).
    """
    errors: list[str] = []

    nodes = graph.get("nodes")
    edges = graph.get("edges")

    if nodes is None:
        errors.append("Graph is missing a 'nodes' key.")
        nodes = []
    if edges is None:
        errors.append("Graph is missing an 'edges' key.")
        edges = []

    node_ids: set[str] = set()
    for i, node in enumerate(nodes):
        node_id = node.get("id")
        node_type = node.get("type")

        if not node_id:
            errors.append(f"Node at index {i} is missing an 'id'.")
        else:
            if node_id in node_ids:
                errors.append(f"Duplicate node id: {node_id!r}.")
            node_ids.add(node_id)

        if node_type not in ALLOWED_NODE_TYPES:
            errors.append(
                f"Node {node_id!r} has invalid type {node_type!r}; "
                f"must be one of {sorted(ALLOWED_NODE_TYPES)}."
            )

    for i, edge in enumerate(edges):
        label = edge.get("id", f"index {i}")

        missing_fields = [f for f in _REQUIRED_EDGE_FIELDS if not edge.get(f) and edge.get(f) != 0]
        # note: confidence=0.0 is a legitimate value and must not be treated
        # as "missing" by a truthiness check, hence the "!= 0" carve-out
        # above only affects the missing-field check, not the range check
        # below, which validates the actual number.
        if missing_fields:
            errors.append(f"Edge {label} is missing required field(s): {missing_fields}.")
            continue  # skip further checks on this edge - fields aren't there to check

        if edge["predicate"] not in ALLOWED_PREDICATES:
            errors.append(
                f"Edge {label} has invalid predicate {edge['predicate']!r}; "
                f"must be one of {sorted(ALLOWED_PREDICATES)}."
            )

        if edge["subject"] not in node_ids:
            errors.append(
                f"Edge {label} references unknown subject node {edge['subject']!r}."
            )
        if edge["object"] not in node_ids:
            errors.append(
                f"Edge {label} references unknown object node {edge['object']!r}."
            )

        confidence = edge["confidence"]
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            errors.append(
                f"Edge {label} has invalid confidence {confidence!r}; "
                "must be a number between 0 and 1."
            )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        node_count=len(nodes),
        edge_count=len(edges),
    )


def validate_graph_or_raise(graph: dict) -> ValidationResult:
    """Same as validate_graph(), but raises GraphValidationError if invalid."""
    result = validate_graph(graph)
    if not result.is_valid:
        raise GraphValidationError(result.errors)
    return result
