"""
Natural language -> Cypher generation for ChronoGraph.

Week 3 additions:
- Deterministic Cypher templates for temporal reasoning patterns.
- AFTER
- BEFORE
- BETWEEN
- ENTITY_HISTORY
- PROJECT_HISTORY
- ADVOCACY_THEN_USAGE
- ARGUMENT_THEN_USAGE

Normal questions are still handled by Gemini using structured output.

The generated query is read-only and is validated separately by
pipeline.cypher_validator.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from pipeline.cypher_validator import (
    ALLOWED_LABELS,
    ALLOWED_RELATIONSHIP_TYPES,
)
from pipeline.llm import get_qa_llm


SCHEMA_DESCRIPTION = f"""
This is a forensic historical knowledge graph called ChronoGraph.

Node labels:
{sorted(ALLOWED_LABELS)}

Relationship types:
{sorted(ALLOWED_RELATIONSHIP_TYPES)}

Every node has:
- id
- name

Every relationship has:
- timestamp
- source
- source_id
- evidence
- confidence

Person names may have variants such as:
- Rohit Nair
- Rohit

Therefore, Person name matching should use case-insensitive CONTAINS.

Technology and Project names may use exact or CONTAINS matching.

Every normal query must return exactly these aliases:

subject_name
subject
predicate
object_name
object
timestamp
source
source_id
evidence
confidence

Use:

type(r) AS predicate

because the predicate is represented by the relationship type.

Queries must be read-only.

Allowed Cypher clauses:
MATCH
OPTIONAL MATCH
WHERE
WITH
RETURN
ORDER BY
LIMIT
UNWIND

Never use:
CREATE
MERGE
SET
DELETE
DETACH DELETE
REMOVE
DROP
ALTER
LOAD CSV
FOREACH
CALL
"""


class GeneratedCypher(BaseModel):
    cypher: str = Field(
        description="A single read-only Cypher query."
    )

    reasoning: str = Field(
        description="One sentence explaining why the query answers the question."
    )


_PROMPT_TEMPLATE = """{schema}

User question:
"{question}"

Generate a single read-only Cypher query that retrieves the graph evidence
needed to answer this question.

Important:
- Respect the allowed node labels and relationship types.
- Use case-insensitive CONTAINS matching for Person names.
- Return the required aliases.
- Order temporal results by timestamp.
- Use LIMIT 50.
- Do not invent graph properties.
- Do not modify the database.

Return only the structured output fields.
"""


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _required_return(prefix: str = "r") -> str:
    """
    Standard return projection expected by the downstream pipeline.
    """
    return f"""
RETURN
    {prefix}_subject.name AS subject_name,
    {prefix}_subject.id AS subject,
    type({prefix}_rel) AS predicate,
    {prefix}_object.name AS object_name,
    {prefix}_object.id AS object,
    {prefix}_rel.timestamp AS timestamp,
    {prefix}_rel.source AS source,
    {prefix}_rel.source_id AS source_id,
    {prefix}_rel.evidence AS evidence,
    {prefix}_rel.confidence AS confidence
ORDER BY timestamp
LIMIT 50
""".strip()


def _extract_date(question: str) -> str | None:
    match = re.search(
        r"\b(\d{4}-\d{2}-\d{2})\b",
        question,
    )
    return match.group(1) if match else None


def _extract_between_dates(
    question: str,
) -> tuple[str, str] | None:
    dates = re.findall(
        r"\b\d{4}-\d{2}-\d{2}\b",
        question,
    )

    if len(dates) >= 2:
        return dates[0], dates[1]

    return None


def _extract_entity_after_phrase(
    question: str,
    phrases: tuple[str, ...],
) -> str | None:
    lower_question = question.lower()

    for phrase in phrases:
        index = lower_question.find(phrase.lower())

        if index != -1:
            entity = question[index + len(phrase):].strip()

            entity = entity.rstrip("?. ")

            if entity:
                return entity

    return None


# ---------------------------------------------------------------------
# Temporal deterministic queries
# ---------------------------------------------------------------------


def _after_query(date: str) -> GeneratedCypher:
    cypher = f"""
MATCH (subject_node)-[rel]->(object_node)
WHERE rel.timestamp > '{date}'
RETURN
    subject_node.name AS subject_name,
    subject_node.id AS subject,
    type(rel) AS predicate,
    object_node.name AS object_name,
    object_node.id AS object,
    rel.timestamp AS timestamp,
    rel.source AS source,
    rel.source_id AS source_id,
    rel.evidence AS evidence,
    rel.confidence AS confidence
ORDER BY timestamp
LIMIT 50
""".strip()

    return GeneratedCypher(
        cypher=cypher,
        reasoning=(
            f"Retrieves all graph events occurring after {date} "
            "in chronological order."
        ),
    )


def _before_query(date: str) -> GeneratedCypher:
    cypher = f"""
MATCH (subject_node)-[rel]->(object_node)
WHERE rel.timestamp < '{date}'
RETURN
    subject_node.name AS subject_name,
    subject_node.id AS subject,
    type(rel) AS predicate,
    object_node.name AS object_name,
    object_node.id AS object,
    rel.timestamp AS timestamp,
    rel.source AS source,
    rel.source_id AS source_id,
    rel.evidence AS evidence,
    rel.confidence AS confidence
ORDER BY timestamp
LIMIT 50
""".strip()

    return GeneratedCypher(
        cypher=cypher,
        reasoning=(
            f"Retrieves all graph events occurring before {date} "
            "in chronological order."
        ),
    )


def _between_query(
    start_date: str,
    end_date: str,
) -> GeneratedCypher:
    cypher = f"""
MATCH (subject_node)-[rel]->(object_node)
WHERE rel.timestamp >= '{start_date}'
  AND rel.timestamp <= '{end_date}'
RETURN
    subject_node.name AS subject_name,
    subject_node.id AS subject,
    type(rel) AS predicate,
    object_node.name AS object_name,
    object_node.id AS object,
    rel.timestamp AS timestamp,
    rel.source AS source,
    rel.source_id AS source_id,
    rel.evidence AS evidence,
    rel.confidence AS confidence
ORDER BY timestamp
LIMIT 50
""".strip()

    return GeneratedCypher(
        cypher=cypher,
        reasoning=(
            f"Retrieves graph events between {start_date} and {end_date}, "
            "inclusive, in chronological order."
        ),
    )


def _entity_history_query(
    entity: str,
) -> GeneratedCypher:
    safe_entity = entity.replace("'", "\\'")

    cypher = f"""
MATCH (subject_node:Person)-[rel]-(object_node)
WHERE toLower(subject_node.name) CONTAINS toLower('{safe_entity}')
RETURN
    subject_node.name AS subject_name,
    subject_node.id AS subject,
    type(rel) AS predicate,
    object_node.name AS object_name,
    object_node.id AS object,
    rel.timestamp AS timestamp,
    rel.source AS source,
    rel.source_id AS source_id,
    rel.evidence AS evidence,
    rel.confidence AS confidence
ORDER BY timestamp
LIMIT 50
""".strip()

    return GeneratedCypher(
        cypher=cypher,
        reasoning=(
            f"Retrieves the chronological relationship history of "
            f"Person '{entity}', including name variants."
        ),
    )


def _project_history_query(
    entity: str,
) -> GeneratedCypher:
    safe_entity = entity.replace("'", "\\'")

    cypher = f"""
MATCH (subject_node)-[rel]-(object_node:Project)
WHERE toLower(object_node.name) CONTAINS toLower('{safe_entity}')
RETURN
    subject_node.name AS subject_name,
    subject_node.id AS subject,
    type(rel) AS predicate,
    object_node.name AS object_name,
    object_node.id AS object,
    rel.timestamp AS timestamp,
    rel.source AS source,
    rel.source_id AS source_id,
    rel.evidence AS evidence,
    rel.confidence AS confidence
ORDER BY timestamp
LIMIT 50
""".strip()

    return GeneratedCypher(
        cypher=cypher,
        reasoning=(
            f"Retrieves the chronological relationship history of "
            f"Project '{entity}'."
        ),
    )


# ---------------------------------------------------------------------
# Advocacy -> later usage
# ---------------------------------------------------------------------


def _advocacy_then_usage_query() -> GeneratedCypher:
    """
    Find a technology that was advocated for and later used.

    We intentionally use two MATCH patterns separated by WITH rather than
    relying on a single chained relationship pattern. This makes the
    temporal comparison explicit and allows the graph to contain different
    people for the advocacy and later implementation.
    """

    cypher = """
MATCH (advocate:Person)-[adv_rel:ADVOCATED_FOR]->(technology:Technology)
MATCH (user:Person)-[use_rel:USED]->(technology)
WHERE adv_rel.timestamp < use_rel.timestamp
WITH
    advocate,
    adv_rel,
    technology,
    user,
    use_rel
UNWIND [
    {
        event_person: advocate,
        event_rel: adv_rel,
        event_object: technology
    },
    {
        event_person: user,
        event_rel: use_rel,
        event_object: technology
    }
] AS event
RETURN
    event.event_person.name AS subject_name,
    event.event_person.id AS subject,
    type(event.event_rel) AS predicate,
    event.event_object.name AS object_name,
    event.event_object.id AS object,
    event.event_rel.timestamp AS timestamp,
    event.event_rel.source AS source,
    event.event_rel.source_id AS source_id,
    event.event_rel.evidence AS evidence,
    event.event_rel.confidence AS confidence
ORDER BY timestamp
LIMIT 50
""".strip()

    return GeneratedCypher(
        cypher=cypher,
        reasoning=(
            "Finds technologies that were advocated for first and used "
            "later, returning both temporal events so the answer can "
            "identify the advocacy and subsequent usage."
        ),
    )


# ---------------------------------------------------------------------
# Argument -> later usage
# ---------------------------------------------------------------------


def _argument_then_usage_query() -> GeneratedCypher:
    """
    Find a technology that someone argued against and was later used.

    For each argument event, select the earliest later usage event for
    the same technology. Return both events exactly once.
    """

    cypher = """
MATCH (opponent:Person)-[arg_rel:ARGUED_AGAINST]->(technology:Technology)
MATCH (user:Person)-[use_rel:USED]->(technology)
WHERE arg_rel.timestamp < use_rel.timestamp
WITH
    opponent,
    arg_rel,
    technology,
    user,
    use_rel
ORDER BY arg_rel.timestamp, use_rel.timestamp
WITH
    opponent,
    arg_rel,
    technology,
    collect({
        user: user,
        use_rel: use_rel
    })[0] AS earliest_usage
WITH
    opponent,
    arg_rel,
    technology,
    earliest_usage
UNWIND [
    {
        event_person: opponent,
        event_rel: arg_rel,
        event_object: technology
    },
    {
        event_person: earliest_usage.user,
        event_rel: earliest_usage.use_rel,
        event_object: technology
    }
] AS event
RETURN DISTINCT
    event.event_person.name AS subject_name,
    event.event_person.id AS subject,
    type(event.event_rel) AS predicate,
    event.event_object.name AS object_name,
    event.event_object.id AS object,
    event.event_rel.timestamp AS timestamp,
    event.event_rel.source AS source,
    event.event_rel.source_id AS source_id,
    event.event_rel.evidence AS evidence,
    event.event_rel.confidence AS confidence
ORDER BY timestamp
LIMIT 50
""".strip()

    return GeneratedCypher(
        cypher=cypher,
        reasoning=(
            "Finds technologies that were argued against first and then "
            "used later, selecting the earliest later usage for each "
            "argument event and returning each event only once."
        ),
    )


# ---------------------------------------------------------------------
# Temporal question detection
# ---------------------------------------------------------------------


def _generate_temporal_query(
    question: str,
) -> GeneratedCypher | None:
    """
    Detect well-defined Week 3 temporal patterns.

    These patterns are deterministic because their graph semantics are
    known and should not depend on Gemini producing exactly the right
    Cypher structure.
    """

    q = question.lower().strip()

    # Advocacy -> usage
    if (
        "advocated for" in q
        and "later used" in q
    ) or (
        "advocated" in q
        and "later" in q
        and "used" in q
    ):
        return _advocacy_then_usage_query()

    # Argument -> usage
    # Supports "argue against", "argued against",
    # and "arguing against".
    if (
        (
            "argue against" in q
            or "argued against" in q
            or "arguing against" in q
        )
        and "used" in q
        and "later" in q
    ) or (
        (
            "argue against" in q
            or "argued against" in q
            or "arguing against" in q
        )
        and "used anyway" in q
    ):
        return _argument_then_usage_query()

    # Between
    if "between" in q:
        dates = _extract_between_dates(question)

        if dates:
            return _between_query(
                dates[0],
                dates[1],
            )

    # After
    if "after" in q:
        date = _extract_date(question)

        if date:
            return _after_query(date)

    # Before
    if "before" in q:
        date = _extract_date(question)

        if date:
            return _before_query(date)

    # Project history
    if "project history" in q:
        entity = _extract_entity_after_phrase(
            question,
            (
                "project history of ",
                "project history for ",
            ),
        )

        if entity:
            return _project_history_query(entity)

    # Project IDs such as CLOUD-52 can be requested with
    # "history of CLOUD-52" without explicitly saying "project".
    if "history of" in q:
        entity = _extract_entity_after_phrase(
            question,
            (
                "history of ",
                "history for ",
            ),
        )

        if entity and re.fullmatch(r"[A-Za-z]+-\d+", entity):
            return _project_history_query(entity)

    # Person/entity history
    if "history of" in q:
        entity = _extract_entity_after_phrase(
            question,
            (
                "history of ",
                "history for ",
            ),
        )

        if entity:
            return _entity_history_query(entity)

    return None


# ---------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------


def generate_cypher(
    question: str,
) -> GeneratedCypher:
    """
    Generate read-only Cypher.

    Week 3 temporal patterns use deterministic templates.

    Other questions are generated by Gemini using structured output.
    """

    temporal_query = _generate_temporal_query(question)

    if temporal_query is not None:
        return temporal_query

    llm = get_qa_llm()

    structured_llm = llm.with_structured_output(
        GeneratedCypher
    )

    prompt = _PROMPT_TEMPLATE.format(
        schema=SCHEMA_DESCRIPTION,
        question=question,
    )

    result = structured_llm.invoke(prompt)

    return result