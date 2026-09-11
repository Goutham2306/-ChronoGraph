"""
Natural language -> Cypher generation for Day 2.

Uses LangChain's `.with_structured_output()` against Gemini to force the
model into a Pydantic schema (GeneratedCypher) rather than parsing Cypher
out of free text. The generated query is NOT executed here — validation
(pipeline/cypher_validator.py) and execution (pipeline/rag_pipeline.py) are
separate stages, per the architecture's separation of concerns.

ENTITY RESOLUTION NOTE (found during Phase 0 inspection): the same person
sometimes appears as two distinct nodes in the graph — e.g.
"Person:Rohit Nair" and "Person:Rohit" — because Day 1 extraction doesn't
canonicalize names across entries. Rather than silently under-retrieving
when a user asks about "Rohit", the generated Cypher uses `CONTAINS`
matching on person names (case-insensitive) instead of exact equality, so
both node variants are captured. This is a real, deliberate workaround for
a real data quality gap — not a hidden assumption.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from pipeline.cypher_validator import ALLOWED_LABELS, ALLOWED_RELATIONSHIP_TYPES
from pipeline.llm import get_qa_llm

SCHEMA_DESCRIPTION = f"""
This is a forensic historical knowledge graph called ChronoGraph, tracking
real engineering decisions, discussions, and code changes over time at a
company migrating infrastructure from AWS to GCP.

Node labels (exactly these, no others): {sorted(ALLOWED_LABELS)}
Relationship types (exactly these, no others): {sorted(ALLOWED_RELATIONSHIP_TYPES)}

Every node has: id (e.g. "Person:Rohit Nair"), name.
Every relationship has: timestamp (ISO date string, e.g. "2023-01-10"),
source (e.g. "slack", "github", "jira"), source_id, evidence (text snippet),
confidence (float 0-1).

IMPORTANT: person names in this graph are inconsistently extracted — the
same person may appear as e.g. both "Rohit Nair" and "Rohit". When matching
a Person node by name, use a case-insensitive CONTAINS match on n.name
rather than exact equality, e.g.:
    MATCH (p:Person) WHERE toLower(p.name) CONTAINS toLower('rohit')
For Technology/Project nodes, exact or CONTAINS matching on name is fine
depending on how the question names the entity.

Always RETURN exactly these aliases, so results can be processed
downstream (use AS to rename): subject_name, subject, predicate,
object_name, object, timestamp, source, source_id, evidence, confidence.
For example:
    RETURN p.name AS subject_name, p.id AS subject, r.predicate AS predicate,
           t.name AS object_name, t.id AS object, r.timestamp AS timestamp,
           r.source AS source, r.source_id AS source_id,
           r.evidence AS evidence, r.confidence AS confidence
Note: r.predicate is not a stored property (the predicate IS the
relationship type) - use type(r) AS predicate instead.
Always include ORDER BY timestamp and a reasonable LIMIT (e.g. 50) unless
the question clearly needs fewer results.

The query MUST be read-only: only MATCH, OPTIONAL MATCH, WHERE, WITH,
RETURN, ORDER BY, LIMIT, UNWIND. Never CREATE, MERGE, SET, DELETE, DETACH,
REMOVE, DROP, ALTER, LOAD CSV, FOREACH, or CALL.
"""


class GeneratedCypher(BaseModel):
    cypher: str = Field(description="A single read-only Cypher query.")
    reasoning: str = Field(
        description="One sentence on why this query answers the question."
    )


_PROMPT_TEMPLATE = """{schema}

User question: "{question}"

Generate a single read-only Cypher query to retrieve the graph evidence
needed to answer this question. Return ONLY the structured output fields.
"""


def generate_cypher(question: str) -> GeneratedCypher:
    """
    Calls Gemini via LangChain structured output to produce a Cypher query
    for the given natural-language question. Does NOT validate or execute
    the query - see pipeline.cypher_validator and pipeline.rag_pipeline.
    """
    llm = get_qa_llm()
    structured_llm = llm.with_structured_output(GeneratedCypher)
    prompt = _PROMPT_TEMPLATE.format(schema=SCHEMA_DESCRIPTION, question=question)
    result = structured_llm.invoke(prompt)
    return result
