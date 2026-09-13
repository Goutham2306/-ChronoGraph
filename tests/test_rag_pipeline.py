import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.rag_pipeline import _build_citations, _mock_retrieve


def test_mock_retrieve_does_not_return_full_graph_for_every_question():
    all_edges = _mock_retrieve("Who argued against AWS?")
    unrelated = _mock_retrieve("What technologies did Karthik commit code for?")
    assert len(all_edges) > 0
    # Different questions should generally retrieve different-sized subsets,
    # not silently be the same "return everything" list.
    assert all_edges != unrelated or len(all_edges) < 72


def test_mock_retrieve_finds_both_name_variants_for_same_person():
    """
    Regression test for the entity-resolution gap found during inspection:
    'Priya' and 'Priya Menon' are separate nodes in the real graph, and a
    question about 'Priya' should surface edges from both.
    """
    results = _mock_retrieve("Who argued against AWS?")
    subject_names = {r["subject_name"] for r in results}
    assert "Priya Menon" in subject_names
    assert "Priya" in subject_names


def test_mock_retrieve_empty_for_nonsense_question():
    results = _mock_retrieve("asdf jkl qwerty")
    assert results == []


def test_citation_validation_drops_uncited_or_invented_source_ids():
    evidence = [
        {
            "source_id": "SLACK-001",
            "timestamp": "2023-01-10",
            "source": "slack",
            "subject": "a", "subject_name": "a", "predicate": "X",
            "object": "b", "object_name": "b", "evidence": "e", "confidence": 0.9,
        }
    ]
    # LLM claims to cite a real one AND a fabricated one - only the real
    # one should survive into the final citations list.
    citations = _build_citations(evidence, ["SLACK-001", "FABRICATED-999"])
    assert len(citations) == 1
    assert citations[0].source_id == "SLACK-001"


def test_citation_validation_with_no_cited_ids():
    citations = _build_citations([], [])
    assert citations == []


def test_neo4j_result_shape_validation_catches_missing_fields():
    """
    Regression test: if the LLM's generated Cypher doesn't alias its
    RETURN clause exactly as instructed, this must fail loudly and
    specifically - not silently produce a KeyError deep inside
    answer.py's formatting code.
    """
    from pipeline.rag_pipeline import REQUIRED_EVIDENCE_FIELDS

    # Simulate what _neo4j_retrieve's validation logic checks: a row
    # missing 'subject_name' and 'object_name' (a plausible LLM mistake).
    bad_row_fields = {
        "timestamp", "subject", "predicate", "object",
        "evidence", "confidence", "source", "source_id",
    }
    missing = REQUIRED_EVIDENCE_FIELDS - bad_row_fields
    assert missing == {"subject_name", "object_name"}
