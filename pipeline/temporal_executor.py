from __future__ import annotations

from pipeline.temporal_reasoning import (
    EvidenceRecord,
    find_advocacy_then_usage,
    find_argument_then_usage,
    get_entity_history,
    get_events_after,
    get_events_before,
    get_events_between,
    get_project_history,
)
from pipeline.temporal_router import TemporalRoute, classify_temporal_question
from pipeline.temporal_retrieval import sort_chronologically


def apply_temporal_reasoning(
    question: str,
    records: list[EvidenceRecord],
) -> tuple[list[EvidenceRecord], TemporalRoute]:
    """
    Apply the temporal reasoning strategy selected from the question.

    This function never creates evidence. Every returned event comes
    from the supplied records.
    """

    route = classify_temporal_question(question)

    if route.intent == "BEFORE":
        return (
            get_events_before(records, route.date),
            route,
        )

    if route.intent == "AFTER":
        return (
            get_events_after(records, route.date),
            route,
        )

    if route.intent == "BETWEEN":
        return (
            get_events_between(
                records,
                route.start_date,
                route.end_date,
            ),
            route,
        )

    if route.intent == "ENTITY_HISTORY":
        return (
            get_entity_history(
                records,
                route.entity,
            ),
            route,
        )

    if route.intent == "PROJECT_HISTORY":
        return (
            get_project_history(
                records,
                route.entity,
            ),
            route,
        )

    if route.intent == "ADVOCACY_THEN_USAGE":
        pairs = find_advocacy_then_usage(records)

        flattened: list[EvidenceRecord] = []

        for pair in pairs:
            flattened.append(pair["advocated"])
            flattened.append(pair["used"])

        return _deduplicate_and_sort(flattened), route

    if route.intent == "ARGUMENT_THEN_USAGE":
        pairs = find_argument_then_usage(records)

        flattened = []

        for pair in pairs:
            flattened.append(pair["argued_against"])
            flattened.append(pair["used_anyway"])

        return _deduplicate_and_sort(flattened), route

    return sort_chronologically(records), route


def _deduplicate_and_sort(
    records: list[EvidenceRecord],
) -> list[EvidenceRecord]:
    """
    Remove duplicate events while preserving only evidence supplied
    to this function, then return the events chronologically sorted.
    """

    seen: set[tuple] = set()
    result: list[EvidenceRecord] = []

    for record in records:
        key = (
            record.get("source_id"),
            record.get("timestamp"),
            record.get("subject"),
            record.get("predicate"),
            record.get("object"),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(record)

    return sort_chronologically(result)