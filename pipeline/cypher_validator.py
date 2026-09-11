"""
Cypher validation gate for Day 2.

LLM-generated Cypher is NEVER executed without passing through this module
first. Pure Python, no LLM calls, no network calls — deterministic and
independently testable. This is layer one of two defenses; layer two is
Neo4j's read-only transaction mode in pipeline/neo4j_loader.py's sibling
query-execution code (pipeline/rag_pipeline.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Schema actually verified against output/graph.json (see Phase 0 inspection).
ALLOWED_LABELS = {"Person", "Technology", "Project"}
ALLOWED_RELATIONSHIP_TYPES = {
    "ADVOCATED_FOR",
    "COMMITTED_CODE",
    "ARGUED_AGAINST",
    "USED",
    "REPORTED",
    "REVIEWED",
}

_DENIED_KEYWORDS = [
    "CREATE",
    "DELETE",
    "DETACH",
    "SET",
    "MERGE",
    "DROP",
    "REMOVE",
    "ALTER",
    "LOAD CSV",
    "FOREACH",
    "CALL",  # ChronoGraph's read queries never need procedure calls
]

_ALLOWED_LEADING_CLAUSES = ("MATCH", "OPTIONAL MATCH", "WITH", "UNWIND")

_SEMICOLON_RE = re.compile(r";")
_VAR_LENGTH_PATH_RE = re.compile(r"\*(?:\s*\.\.\s*(\d+))?")

# Relationship types appear inside [ ... ], node labels inside ( ... ).
_REL_TYPE_RE = re.compile(r"\[\s*\w*\s*:(\w+)")
_LABEL_RE = re.compile(r"\(\s*\w*((?:\s*:\s*\w+)+)\s*[\{\)]")
_LABEL_SPLIT_RE = re.compile(r":\s*(\w+)")


@dataclass
class ValidationResult:
    is_valid: bool
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_valid


def validate_cypher(cypher: str, max_traversal_depth: int = 4) -> ValidationResult:
    reasons: list[str] = []

    if not cypher or not cypher.strip():
        return ValidationResult(is_valid=False, reasons=["Empty query."])

    stripped = cypher.strip()
    upper = stripped.upper()

    for keyword in _DENIED_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", upper):
            reasons.append(f"Contains disallowed keyword: {keyword}")

    if not upper.startswith(_ALLOWED_LEADING_CLAUSES):
        reasons.append(
            "Query must begin with MATCH, OPTIONAL MATCH, WITH, or UNWIND."
        )

    if _SEMICOLON_RE.search(stripped):
        reasons.append("Multiple chained statements are not allowed.")

    referenced_labels: set[str] = set()
    for label_group in _LABEL_RE.findall(stripped):
        referenced_labels.update(_LABEL_SPLIT_RE.findall(label_group))
    unknown_labels = referenced_labels - ALLOWED_LABELS
    if unknown_labels:
        reasons.append(f"Unknown label(s) not in schema: {sorted(unknown_labels)}")

    referenced_rel_types = set(_REL_TYPE_RE.findall(stripped))
    unknown_rel_types = referenced_rel_types - ALLOWED_RELATIONSHIP_TYPES
    if unknown_rel_types:
        reasons.append(
            f"Unknown relationship type(s) not in schema: {sorted(unknown_rel_types)}"
        )

    for match in _VAR_LENGTH_PATH_RE.finditer(stripped):
        depth_str = match.group(1)
        if depth_str is None:
            reasons.append("Unbounded variable-length path (e.g. -[*]-) not allowed.")
        elif int(depth_str) > max_traversal_depth:
            reasons.append(
                f"Path depth {depth_str} exceeds max allowed depth {max_traversal_depth}."
            )

    return ValidationResult(is_valid=len(reasons) == 0, reasons=reasons)
