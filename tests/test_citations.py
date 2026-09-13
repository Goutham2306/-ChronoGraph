import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.citations import Citation, build_citations, format_citation


def _sample_evidence():
    return [
        {
            "source_id": "SLACK-001",
            "timestamp": "2023-01-10",
            "source": "slack",
            "subject": "a", "subject_name": "a", "predicate": "X",
            "object": "b", "object_name": "b",
            "evidence": "Rohit raised concerns about Redshift.",
            "confidence": 0.9,
        },
        {
            "source_id": "GITHUB-004",
            "timestamp": "2023-03-12",
            "source": "github",
            "subject": "c", "subject_name": "c", "predicate": "Y",
            "object": "d", "object_name": "d",
            "evidence": "Merged the GKE cutover PR.",
            "confidence": 0.85,
        },
    ]


def test_build_citations_keeps_only_real_cited_sources():
    evidence = _sample_evidence()
    citations = build_citations(evidence, ["SLACK-001"])
    assert len(citations) == 1
    assert citations[0].source_id == "SLACK-001"
    assert citations[0].timestamp == "2023-01-10"
    assert citations[0].source == "slack"


def test_build_citations_drops_fabricated_source_id():
    """
    The core anti-hallucination guarantee: a source_id the model claims to
    have cited, but that isn't in the actually-retrieved evidence, must
    never become a Citation.
    """
    evidence = _sample_evidence()
    citations = build_citations(evidence, ["SLACK-001", "FABRICATED-999"])
    source_ids = {c.source_id for c in citations}
    assert source_ids == {"SLACK-001"}
    assert "FABRICATED-999" not in source_ids


def test_build_citations_preserves_multiple_real_sources():
    evidence = _sample_evidence()
    citations = build_citations(evidence, ["SLACK-001", "GITHUB-004"])
    assert len(citations) == 2
    assert {c.source_id for c in citations} == {"SLACK-001", "GITHUB-004"}


def test_build_citations_empty_inputs():
    assert build_citations([], []) == []
    assert build_citations(_sample_evidence(), []) == []


def test_build_citations_carries_evidence_text():
    evidence = _sample_evidence()
    citations = build_citations(evidence, ["SLACK-001"])
    assert citations[0].evidence == "Rohit raised concerns about Redshift."


def test_format_citation_matches_spec_shape():
    citation = Citation(source_id="SLACK-001", timestamp="2023-01-10", source="slack")
    assert format_citation(citation) == "[SLACK-001, 2023-01-10]"
