import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.graph_validator import (
    GraphValidationError,
    validate_graph,
    validate_graph_or_raise,
)

GRAPH_PATH = Path(__file__).resolve().parent.parent / "output" / "graph.json"


def _real_graph():
    return json.load(open(GRAPH_PATH))


def test_real_graph_json_validates_cleanly():
    result = validate_graph(_real_graph())
    assert result.is_valid, result.errors
    assert result.node_count == len(_real_graph()["nodes"])
    assert result.edge_count == len(_real_graph()["edges"])


def test_catches_invalid_node_type():
    graph = copy.deepcopy(_real_graph())
    graph["nodes"][0]["type"] = "Robot"
    result = validate_graph(graph)
    assert not result.is_valid
    assert any("Robot" in e for e in result.errors)


def test_catches_missing_node_id():
    graph = copy.deepcopy(_real_graph())
    del graph["nodes"][0]["id"]
    result = validate_graph(graph)
    assert not result.is_valid
    assert any("missing an 'id'" in e for e in result.errors)


def test_catches_duplicate_node_id():
    graph = copy.deepcopy(_real_graph())
    graph["nodes"].append(dict(graph["nodes"][0]))  # exact duplicate
    result = validate_graph(graph)
    assert not result.is_valid
    assert any("Duplicate node id" in e for e in result.errors)


def test_catches_invalid_predicate():
    graph = copy.deepcopy(_real_graph())
    graph["edges"][0]["predicate"] = "SABOTAGED"
    result = validate_graph(graph)
    assert not result.is_valid
    assert any("SABOTAGED" in e for e in result.errors)


def test_catches_dangling_subject_reference():
    graph = copy.deepcopy(_real_graph())
    graph["edges"][0]["subject"] = "Person:DoesNotExist"
    result = validate_graph(graph)
    assert not result.is_valid
    assert any("DoesNotExist" in e for e in result.errors)


def test_catches_dangling_object_reference():
    graph = copy.deepcopy(_real_graph())
    graph["edges"][0]["object"] = "Technology:DoesNotExist"
    result = validate_graph(graph)
    assert not result.is_valid
    assert any("DoesNotExist" in e for e in result.errors)


def test_catches_out_of_range_confidence():
    graph = copy.deepcopy(_real_graph())
    graph["edges"][0]["confidence"] = 1.5
    result = validate_graph(graph)
    assert not result.is_valid
    assert any("confidence" in e.lower() for e in result.errors)


def test_confidence_of_exactly_zero_is_not_treated_as_missing():
    """
    A truthiness-based missing-field check would incorrectly flag
    confidence=0.0 as absent. It's a legitimate value and must pass.
    """
    graph = copy.deepcopy(_real_graph())
    graph["edges"][0]["confidence"] = 0.0
    result = validate_graph(graph)
    missing_confidence_errors = [
        e for e in result.errors
        if "missing required field" in e and "confidence" in e.lower()
    ]
    assert missing_confidence_errors == []


def test_catches_missing_evidence_field():
    graph = copy.deepcopy(_real_graph())
    del graph["edges"][0]["evidence"]
    result = validate_graph(graph)
    assert not result.is_valid
    assert any("missing required field" in e for e in result.errors)


def test_catches_missing_timestamp_source_source_id():
    for field_name in ("timestamp", "source", "source_id"):
        graph = copy.deepcopy(_real_graph())
        del graph["edges"][0][field_name]
        result = validate_graph(graph)
        assert not result.is_valid, f"should have failed on missing {field_name}"


def test_validate_graph_or_raise_raises_on_invalid_graph():
    graph = copy.deepcopy(_real_graph())
    graph["edges"][0]["predicate"] = "SABOTAGED"
    try:
        validate_graph_or_raise(graph)
        assert False, "should have raised GraphValidationError"
    except GraphValidationError as exc:
        assert len(exc.errors) > 0


def test_validate_graph_or_raise_does_not_raise_on_valid_graph():
    validate_graph_or_raise(_real_graph())  # should not raise


def test_missing_nodes_or_edges_key_reported_not_crashed():
    result = validate_graph({"nodes": []})
    assert not result.is_valid
    assert any("'edges'" in e for e in result.errors)
