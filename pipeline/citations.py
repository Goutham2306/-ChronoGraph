"""
Citation construction and validation.

This logic previously lived inline inside pipeline/rag_pipeline.py as a
private helper. It's pulled out here as a standalone, reusable module per
the Week 2 project structure - pipeline/rag_pipeline.py now imports from
here rather than defining its own copy, so there is exactly one
implementation of citation validation, not two.

Core rule: a citation is only ever built from a source_id that (a) the
answer-generation model claimed to rely on AND (b) actually exists in the
evidence that was really retrieved. Anything else is dropped silently
rather than trusted - this is what prevents an invented citation from
reaching the user even if the LLM hallucinated one.
"""

from __future__ import annotations

from pydantic import BaseModel

from pipeline.temporal_retrieval import EvidenceRecord


class Citation(BaseModel):
    source_id: str
    timestamp: str
    source: str
    evidence: str | None = None


def build_citations(
    evidence: list[EvidenceRecord], cited_source_ids: list[str]
) -> list[Citation]:
    """
    Builds validated Citation objects.

    Args:
        evidence: the actual retrieved evidence records for this question
            (already temporally sorted or not - order doesn't matter here).
        cited_source_ids: source_id values the answer-generation model
            claimed to have relied on.

    Returns:
        One Citation per cited_source_id that was found in `evidence`.
        Any cited_source_id NOT present in evidence is dropped - never
        fabricated, never passed through on trust.
    """
    by_source_id = {e["source_id"]: e for e in evidence}
    citations: list[Citation] = []
    for source_id in cited_source_ids:
        record = by_source_id.get(source_id)
        if record is None:
            continue
        citations.append(
            Citation(
                source_id=source_id,
                timestamp=record["timestamp"],
                source=record["source"],
                evidence=record.get("evidence"),
            )
        )
    return citations


def format_citation(citation: Citation) -> str:
    """Renders a citation as "[SOURCE_ID, timestamp]", e.g. "[SLACK-001, 2023-01-10]"."""
    return f"[{citation.source_id}, {citation.timestamp}]"
