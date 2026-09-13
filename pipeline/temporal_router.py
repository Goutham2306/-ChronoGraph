from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


TemporalIntent = Literal[
    "BEFORE",
    "AFTER",
    "BETWEEN",
    "ENTITY_HISTORY",
    "PROJECT_HISTORY",
    "ADVOCACY_THEN_USAGE",
    "ARGUMENT_THEN_USAGE",
    "GENERAL",
]


@dataclass(frozen=True)
class TemporalRoute:
    intent: TemporalIntent
    entity: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    date: str | None = None


_ISO_DATE = r"\d{4}-\d{2}-\d{2}"


def classify_temporal_question(question: str) -> TemporalRoute:
    """
    Classify a user question into a deterministic temporal reasoning route.

    The router does not retrieve data and does not generate facts.
    It only identifies which existing temporal reasoning operation
    should be applied to retrieved evidence.
    """

    q = question.strip()
    lower = q.lower()

    # ---------------------------------------------------------
    # Advocacy -> later usage
    # ---------------------------------------------------------

    if (
        ("advocat" in lower or "advocated for" in lower)
        and (
            "then used" in lower
            or "later used" in lower
            or "eventually used" in lower
            or ("after" in lower and "used" in lower)
        )
    ):
        return TemporalRoute(
            intent="ADVOCACY_THEN_USAGE"
        )

    # ---------------------------------------------------------
    # Argument against -> later usage
    # ---------------------------------------------------------

    if (
        (
            "argued against" in lower
            or "argument against" in lower
        )
        and (
            "then used" in lower
            or "later used" in lower
            or "anyway" in lower
            or "eventually used" in lower
        )
    ):
        return TemporalRoute(
            intent="ARGUMENT_THEN_USAGE"
        )

    # ---------------------------------------------------------
    # Between two dates
    # ---------------------------------------------------------

    between = re.search(
        rf"(?:between|from)\s+({_ISO_DATE})\s+"
        rf"(?:and|to)\s+({_ISO_DATE})",
        lower,
    )

    if between:
        return TemporalRoute(
            intent="BETWEEN",
            start_date=between.group(1),
            end_date=between.group(2),
        )

    # ---------------------------------------------------------
    # After a date
    # ---------------------------------------------------------

    after = re.search(
        rf"(?:after|following|since)\s+({_ISO_DATE})",
        lower,
    )

    if after:
        return TemporalRoute(
            intent="AFTER",
            date=after.group(1),
        )

    # ---------------------------------------------------------
    # Before a date
    # ---------------------------------------------------------

    before = re.search(
        rf"(?:before|prior to|until)\s+({_ISO_DATE})",
        lower,
    )

    if before:
        return TemporalRoute(
            intent="BEFORE",
            date=before.group(1),
        )

    # ---------------------------------------------------------
    # Entity / project history
    # ---------------------------------------------------------

    history_match = re.search(
        r"(?:history|historical timeline|timeline)"
        r"\s+(?:of|for)\s+(.+?)(?:\?|$)",
        q,
        re.IGNORECASE,
    )

    if history_match:
        entity = history_match.group(1).strip()

        if "project" in lower:
            return TemporalRoute(
                intent="PROJECT_HISTORY",
                entity=entity,
            )

        return TemporalRoute(
            intent="ENTITY_HISTORY",
            entity=entity,
        )

    # ---------------------------------------------------------
    # Generic entity history
    # ---------------------------------------------------------

    possessive = re.search(
        r"\b(?:what happened to|show me|give me)\s+"
        r"(.+?)(?:'s)?\s+(?:history|timeline)",
        q,
        re.IGNORECASE,
    )

    if possessive:
        return TemporalRoute(
            intent="ENTITY_HISTORY",
            entity=possessive.group(1).strip(),
        )

    # ---------------------------------------------------------
    # No temporal pattern
    # ---------------------------------------------------------

    return TemporalRoute(
        intent="GENERAL"
    )