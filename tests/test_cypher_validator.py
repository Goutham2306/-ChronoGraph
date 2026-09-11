import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.cypher_validator import validate_cypher


def test_valid_read_query_passes():
    cypher = (
        "MATCH (p:Person)-[r:ADVOCATED_FOR]->(t:Technology) "
        "WHERE toLower(p.name) CONTAINS toLower('rohit') "
        "RETURN p.name, type(r), t.name, r.timestamp "
        "ORDER BY r.timestamp LIMIT 50"
    )
    result = validate_cypher(cypher)
    assert result.is_valid, result.reasons


def test_rejects_create():
    result = validate_cypher("CREATE (n:Person {name: 'Evil'}) RETURN n")
    assert not result.is_valid


def test_rejects_delete():
    result = validate_cypher("MATCH (n:Person) DETACH DELETE n")
    assert not result.is_valid


def test_rejects_set():
    result = validate_cypher("MATCH (n:Person) SET n.hacked = true RETURN n")
    assert not result.is_valid


def test_rejects_merge():
    result = validate_cypher("MERGE (n:Person {name: 'Fake'}) RETURN n")
    assert not result.is_valid


def test_rejects_call_procedures():
    result = validate_cypher("MATCH (n) CALL dbms.killQuery('123') RETURN n")
    assert not result.is_valid


def test_rejects_unknown_label():
    result = validate_cypher("MATCH (n:AdminUser) RETURN n")
    assert not result.is_valid
    assert any("Unknown label" in r for r in result.reasons)


def test_rejects_unknown_relationship_type():
    result = validate_cypher(
        "MATCH (a:Person)-[:DROP_EVERYTHING]->(b:Technology) RETURN a, b"
    )
    assert not result.is_valid
    assert any("Unknown relationship type" in r for r in result.reasons)


def test_rejects_unbounded_variable_length_path():
    result = validate_cypher("MATCH (a:Person)-[*]-(b:Technology) RETURN a, b")
    assert not result.is_valid


def test_rejects_chained_statements():
    result = validate_cypher(
        "MATCH (n:Person) RETURN n; MATCH (m:Technology) DETACH DELETE m"
    )
    assert not result.is_valid


def test_rejects_empty_query():
    result = validate_cypher("")
    assert not result.is_valid


def test_all_real_predicates_pass_when_used_correctly():
    predicates = [
        "ADVOCATED_FOR", "COMMITTED_CODE", "ARGUED_AGAINST",
        "USED", "REPORTED", "REVIEWED",
    ]
    for pred in predicates:
        cypher = f"MATCH (a:Person)-[r:{pred}]->(b:Technology) RETURN a, r, b LIMIT 10"
        result = validate_cypher(cypher)
        assert result.is_valid, f"{pred}: {result.reasons}"


def test_all_real_labels_pass_when_used_correctly():
    for label in ["Person", "Technology", "Project"]:
        cypher = f"MATCH (n:{label}) RETURN n LIMIT 10"
        result = validate_cypher(cypher)
        assert result.is_valid, f"{label}: {result.reasons}"
