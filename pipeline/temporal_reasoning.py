"""
Temporal reasoning layer.

Builds on pipeline/temporal_retrieval.py's sorting/filtering primitives to
answer questions about SEQUENCE and CHANGE, not just "what happened" -
e.g. "what happened before/after X", "what is this entity's full history",
"was there advocacy followed later by implementation".

Everything here operates on already-retrieved EvidenceRecord lists (real
graph data, or a filtered subset of it) - this module does no retrieval
of its own and never invents an event that isn't in its input. All reasoning
is grounded in the timestamp field, parsed as a real date via
pipeline.temporal_retrieval, never a string-position hack.
"""

from __future__ import annotations

from pipeline.temporal_retrieval import (
    EvidenceRecord,
    _parse_timestamp,
    filter_by_date_range,
    sort_chronologically,
)


def get_events_before(
    records: list[EvidenceRecord],
    date: str,
) -> list[EvidenceRecord]:
    """All records with timestamp strictly before `date`, chronologically sorted."""
    cutoff = _parse_timestamp(date)

    filtered = [
        r
        for r in records
        if _safe_parse(r) is not None
        and _safe_parse(r) < cutoff
    ]

    return sort_chronologically(filtered)


def get_events_after(
    records: list[EvidenceRecord],
    date: str,
) -> list[EvidenceRecord]:
    """All records with timestamp strictly after `date`, chronologically sorted."""
    cutoff = _parse_timestamp(date)

    filtered = [
        r
        for r in records
        if _safe_parse(r) is not None
        and _safe_parse(r) > cutoff
    ]

    return sort_chronologically(filtered)


def get_events_between(
    records: list[EvidenceRecord],
    start_date: str,
    end_date: str,
) -> list[EvidenceRecord]:
    """
    Thin wrapper over filter_by_date_range + sort, for naming symmetry
    with the other get_events_* functions.
    """
    filtered = filter_by_date_range(
        records,
        start=start_date,
        end=end_date,
    )

    return sort_chronologically(filtered)


def get_entity_history(
    records: list[EvidenceRecord],
    entity_name: str,
) -> list[EvidenceRecord]:
    """
    Chronological history of every event where `entity_name` appears as
    either subject or object.

    Uses a case-insensitive substring match to work around the graph's
    known name-variant issue, e.g. "Rohit" vs "Rohit Nair".
    """
    needle = entity_name.lower()

    matched = [
        r
        for r in records
        if needle in r.get("subject_name", "").lower()
        or needle in r.get("object_name", "").lower()
    ]

    return sort_chronologically(matched)


def get_project_history(
    records: list[EvidenceRecord],
    project_name: str,
) -> list[EvidenceRecord]:
    """
    Same as get_entity_history, named separately for readability when
    the caller knows they mean a Project.
    """
    return get_entity_history(records, project_name)


def _safe_parse(record: EvidenceRecord):
    """
    Safely parse a record timestamp.

    Returns None if the timestamp is missing or invalid.
    """
    try:
        return _parse_timestamp(record["timestamp"])
    except (ValueError, KeyError):
        return None


# ----------------------------------------------------------------------
# Pattern detection
# ----------------------------------------------------------------------
#
# These functions look for specific, named sequences across an entity's
# sorted history. They report only patterns that are actually present in
# the given records.
#
# They do not infer intent or causation beyond what the sequence of
# predicates and timestamps directly shows.


def find_advocacy_then_usage(
    records: list[EvidenceRecord],
) -> list[dict]:
    """
    Finds cases where a subject ADVOCATED_FOR some object, and later a
    USED event referencing the same object appears.

    Returns:
        A list of dictionaries containing:
        {
            "advocated": <record>,
            "used": <record>
        }

    For each advocacy event, only the earliest matching later usage
    event is selected.
    """

    timeline = sort_chronologically(records)

    advocacy_events = [
        r
        for r in timeline
        if r.get("predicate") == "ADVOCATED_FOR"
    ]

    usage_events = [
        r
        for r in timeline
        if r.get("predicate") == "USED"
    ]

    pairs = []

    for adv in advocacy_events:
        adv_ts = _safe_parse(adv)

        if adv_ts is None:
            continue

        for use in usage_events:
            use_ts = _safe_parse(use)

            if use_ts is None:
                continue

            if (
                use.get("object") == adv.get("object")
                and use_ts > adv_ts
            ):
                pairs.append(
                    {
                        "advocated": adv,
                        "used": use,
                    }
                )

                # Select only the earliest matching usage event
                # for this advocacy event.
                break

    return pairs


def find_argument_then_usage(
    records: list[EvidenceRecord],
) -> list[dict]:
    """
    Finds cases where someone ARGUED_AGAINST an object, but a later USED
    event for that same object exists anyway.

    This represents a temporal sequence:

        ARGUED_AGAINST
              ↓
          later
              ↓
             USED

    Returns:
        A list of dictionaries containing:
        {
            "argued_against": <record>,
            "used_anyway": <record>
        }

    For each argument event, only the earliest matching later usage
    event is selected.
    """

    timeline = sort_chronologically(records)

    argument_events = [
        r
        for r in timeline
        if r.get("predicate") == "ARGUED_AGAINST"
    ]

    usage_events = [
        r
        for r in timeline
        if r.get("predicate") == "USED"
    ]

    pairs = []

    for arg in argument_events:
        arg_ts = _safe_parse(arg)

        if arg_ts is None:
            continue

        for use in usage_events:
            use_ts = _safe_parse(use)

            if use_ts is None:
                continue

            # Same technology/object AND usage happened later.
            if (
                use.get("object") == arg.get("object")
                and use_ts > arg_ts
            ):
                pairs.append(
                    {
                        "argued_against": arg,
                        "used_anyway": use,
                    }
                )

                # Important:
                # Only keep the earliest later usage for this
                # argument event.
                break

    return pairs


def summarize_entity_timeline(
    records: list[EvidenceRecord],
    entity_name: str,
) -> dict:
    """
    Groups an entity's chronological history into:

    - first event
    - latest event
    - complete ordered history
    - event count

    This is a convenience shape for CLI/API display and does not add
    a new reasoning capability beyond get_entity_history().
    """

    history = get_entity_history(
        records,
        entity_name,
    )

    if not history:
        return {
            "entity": entity_name,
            "event_count": 0,
            "first": None,
            "latest": None,
            "history": [],
        }

    return {
        "entity": entity_name,
        "event_count": len(history),
        "first": history[0],
        "latest": history[-1],
        "history": history,
    }