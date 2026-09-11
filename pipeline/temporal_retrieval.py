"""
Temporal normalization + chronological sorting of retrieved evidence.

Works on plain dicts shaped like graph.json edges (whether they came from a
real Neo4j query result or mock-mode filtering of output/graph.json) — see
pipeline/rag_pipeline.py for where each mode's raw rows get converted into
this common shape before reaching here.

CRITICAL (per spec): sorting must use the actual timestamp field, parsed as
a real date, not a string-search hack.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TypedDict


class EvidenceRecord(TypedDict):
    timestamp: str  # ISO date string, e.g. "2023-01-10"
    subject: str
    subject_name: str
    predicate: str
    object: str
    object_name: str
    evidence: str
    confidence: float
    source: str
    source_id: str


def _parse_timestamp(ts: str) -> date:
    """
    Parses the ISO date strings used throughout graph.json
    (e.g. "2023-01-10"). Raises ValueError on malformed input rather than
    silently sorting it to the wrong place — a missing/bad timestamp should
    be visible, not hidden.
    """
    return datetime.strptime(ts, "%Y-%m-%d").date()


def sort_chronologically(records: list[EvidenceRecord]) -> list[EvidenceRecord]:
    """
    Sorts evidence records by their real timestamp field, ascending
    (earliest first). Records with unparseable timestamps are moved to the
    end and flagged, rather than crashing the whole retrieval or being
    silently misplaced.
    """
    valid: list[tuple[date, EvidenceRecord]] = []
    invalid: list[EvidenceRecord] = []

    for record in records:
        try:
            valid.append((_parse_timestamp(record["timestamp"]), record))
        except (ValueError, KeyError):
            invalid.append(record)

    valid.sort(key=lambda pair: pair[0])
    return [r for _, r in valid] + invalid


def filter_by_date_range(
    records: list[EvidenceRecord],
    start: str | None = None,
    end: str | None = None,
) -> list[EvidenceRecord]:
    """Filters records to those with timestamp in [start, end] (inclusive)."""
    start_date = _parse_timestamp(start) if start else None
    end_date = _parse_timestamp(end) if end else None

    result = []
    for record in records:
        try:
            ts = _parse_timestamp(record["timestamp"])
        except (ValueError, KeyError):
            continue
        if start_date and ts < start_date:
            continue
        if end_date and ts > end_date:
            continue
        result.append(record)
    return result
