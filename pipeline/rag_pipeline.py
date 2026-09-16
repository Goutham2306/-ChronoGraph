"""
Day 2 orchestrator.

question -> Cypher generation -> validation -> retrieval (Neo4j or mock) ->
temporal sorting -> Week 3 temporal reasoning -> answer generation ->
citation validation -> result

Mode is explicit and always reported (MODE: NEO4J or MODE: MOCK) - mock
results are never presented as if they came from Neo4j, per spec.

Mock mode does NOT execute the generated Cypher. It filters output/graph.json
directly by entities mentioned in the question.

For temporal-only questions, mock mode provides the graph evidence to the
Week 3 temporal reasoning layer so that date filtering happens there.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from pipeline.answer import INSUFFICIENT_EVIDENCE_MESSAGE, generate_answer
from pipeline.cypher_generator import generate_cypher
from pipeline.cypher_validator import validate_cypher
from pipeline.temporal_executor import apply_temporal_reasoning
from pipeline.temporal_retrieval import EvidenceRecord, sort_chronologically


load_dotenv()


GRAPH_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "graph.json"
)


class Citation(BaseModel):
    source_id: str
    timestamp: str
    source: str


class PipelineResult(BaseModel):
    mode: Literal["neo4j", "mock"]
    question: str
    cypher: str | None
    cypher_reasoning: str | None
    cypher_valid: bool
    cypher_validation_reasons: list[str]
    evidence: list[dict]
    timeline: list[dict]
    answer: str
    citations: list[Citation]
    latency_seconds: float
    error: str | None = None


def _load_graph_json() -> dict:
    if not GRAPH_JSON_PATH.exists():
        raise FileNotFoundError(
            f"{GRAPH_JSON_PATH} not found. "
            "Run `python run_day1.py` first."
        )

    with open(GRAPH_JSON_PATH) as f:
        return json.load(f)


def _extract_mentioned_terms(question: str) -> list[str]:
    """
    Simple term extraction for mock-mode filtering.
    """

    words = (
        question
        .replace("?", "")
        .replace(",", "")
        .split()
    )

    candidates = [
        w
        for w in words
        if w[:1].isupper() and len(w) > 2
    ]

    return candidates


def _mock_retrieve(
    question: str,
) -> list[EvidenceRecord]:
    """
    Retrieve evidence from graph.json in mock mode.

    Entity-specific questions are filtered using names mentioned
    in the question.

    Temporal-only questions contain dates such as:
        - What happened after 2023-03-10?
        - What happened before 2023-03-10?
        - What happened between 2023-03-01 and 2023-03-20?

    These questions should not be filtered using capitalized words such
    as "What". Instead, the complete graph evidence is passed to the
    Week 3 temporal reasoning layer.

    Questions with neither an entity nor a temporal condition return
    no evidence.

    This function never creates new evidence.
    """

    graph = _load_graph_json()

    node_names = {
        n["id"]: n["name"]
        for n in graph["nodes"]
    }

    terms = [
        term.lower()
        for term in _extract_mentioned_terms(question)
    ]

    # Detect temporal-only questions.
    #
    # A date plus a temporal keyword identifies a temporal query.
    # This prevents words such as "What" from being treated as entities.
    temporal_only = bool(
        re.search(
            r"\b\d{4}-\d{2}-\d{2}\b",
            question,
        )
    ) and any(
        keyword in question.lower()
        for keyword in (
            "before",
            "after",
            "between",
            "later",
            "earlier",
            "history",
        )
    )

    # Temporal-only questions need all graph events so that
    # Week 3 temporal reasoning can perform the actual filtering.
    if temporal_only:
        return [
            {
                **edge,
                "subject_name": node_names.get(
                    edge["subject"],
                    "",
                ),
                "object_name": node_names.get(
                    edge["object"],
                    "",
                ),
            }
            for edge in graph["edges"]
        ]

    # If there is no recognizable entity and the question is not
    # temporal, there is no evidence to retrieve.
    if not terms:
        return []

    matched: list[EvidenceRecord] = []

    for edge in graph["edges"]:
        subject_name = node_names.get(
            edge["subject"],
            "",
        )

        object_name = node_names.get(
            edge["object"],
            "",
        )

        haystack = (
            f"{subject_name} {object_name}"
        ).lower()

        if any(
            term in haystack
            for term in terms
        ):
            record: EvidenceRecord = {
                **edge,
                "subject_name": subject_name,
                "object_name": object_name,
            }

            matched.append(record)

    return matched


REQUIRED_EVIDENCE_FIELDS = {
    "timestamp",
    "subject",
    "subject_name",
    "predicate",
    "object",
    "object_name",
    "evidence",
    "confidence",
    "source",
    "source_id",
}


class Neo4jResultShapeError(Exception):
    """
    Raised when the generated Cypher RETURN clause does not
    produce the fields required by the pipeline.
    """


def _neo4j_retrieve(
    cypher: str,
) -> list[EvidenceRecord]:
    """
    Executes validated Cypher against Neo4j.
    """

    from neo4j import GraphDatabase

    uri = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    database = os.environ.get(
        "NEO4J_DATABASE",
        "neo4j",
    )

    if not all(
        [
            uri,
            username,
            password,
        ]
    ):
        raise RuntimeError(
            "Neo4j mode requires NEO4J_URI, "
            "NEO4J_USERNAME, NEO4J_PASSWORD "
            "to be set in .env."
        )

    driver = GraphDatabase.driver(
        uri,
        auth=(
            username,
            password,
        ),
    )

    try:
        driver.verify_connectivity()

        with driver.session(
            database=database
        ) as session:

            def _work(tx):
                result = tx.run(cypher)

                return [
                    record.data()
                    for record in result
                ]

            rows = session.execute_read(_work)

    finally:
        driver.close()

    if rows:
        actual_fields = set(
            rows[0].keys()
        )

        missing = (
            REQUIRED_EVIDENCE_FIELDS
            - actual_fields
        )

        if missing:
            raise Neo4jResultShapeError(
                "Generated Cypher's RETURN clause "
                f"is missing required field(s): "
                f"{sorted(missing)}. "
                f"Got fields: "
                f"{sorted(actual_fields)}. "
                f"Cypher was:\n{cypher}"
            )

    normalized: list[EvidenceRecord] = []

    for row in rows:
        normalized.append(
            row
        )  # type: ignore[arg-type]

    return normalized


def _build_citations(
    evidence: list[EvidenceRecord],
    cited_ids: list[str],
) -> list[Citation]:
    """
    Build citations only from source IDs that actually
    exist in retrieved evidence.
    """

    by_source_id = {
        e["source_id"]: e
        for e in evidence
        if e.get("source_id")
    }

    citations: list[Citation] = []

    seen: set[str] = set()

    for source_id in cited_ids:

        if source_id in seen:
            continue

        record = by_source_id.get(
            source_id
        )

        if record is None:
            continue

        citations.append(
            Citation(
                source_id=source_id,
                timestamp=record["timestamp"],
                source=record["source"],
            )
        )

        seen.add(source_id)

    return citations


def answer_question(
    question: str,
) -> PipelineResult:

    start = time.time()

    use_mock = (
        os.environ.get(
            "CHRONOGRAPH_MOCK_MODE",
            "false",
        ).lower()
        == "true"
    )

    mode: Literal["neo4j", "mock"] = (
        "mock"
        if use_mock
        else "neo4j"
    )

    # =========================================================
    # 1. Generate Cypher
    # =========================================================

    try:
        generated = generate_cypher(
            question
        )

    except Exception as exc:  # noqa: BLE001

        return PipelineResult(
            mode=mode,
            question=question,
            cypher=None,
            cypher_reasoning=None,
            cypher_valid=False,
            cypher_validation_reasons=[
                f"Cypher generation failed: {exc}"
            ],
            evidence=[],
            timeline=[],
            answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            citations=[],
            latency_seconds=(
                time.time() - start
            ),
            error="cypher_generation_failed",
        )

    # =========================================================
    # 2. Validate Cypher
    # =========================================================

    validation = validate_cypher(
        generated.cypher
    )

    evidence: list[EvidenceRecord] = []
    error = None

    # =========================================================
    # 3. Retrieve evidence
    # =========================================================

    if mode == "mock":

        evidence = _mock_retrieve(
            question
        )

    else:

        if not validation.is_valid:

            error = (
                "cypher_validation_failed"
            )

        else:

            try:

                evidence = _neo4j_retrieve(
                    generated.cypher
                )

            except Neo4jResultShapeError as exc:

                error = (
                    "cypher_result_shape_invalid: "
                    f"{exc}"
                )

                evidence = []

            except Exception as exc:  # noqa: BLE001

                error = (
                    "neo4j_unavailable"
                )

                evidence = []

    # =========================================================
    # 4. Chronological sorting
    # =========================================================

    timeline = sort_chronologically(
        evidence
    )

    # =========================================================
    # 5. WEEK 3 TEMPORAL REASONING
    # =========================================================
    #
    # Question
    #    ↓
    # Temporal Router
    #    ↓
    # Temporal Reasoning
    #    ↓
    # Filtered / related evidence
    #
    # The executor never creates new evidence.
    # =========================================================

    timeline, temporal_route = (
        apply_temporal_reasoning(
            question,
            timeline,
        )
    )

    # =========================================================
    # 6. Generate grounded answer
    # =========================================================

    if not timeline:

        answer_text = (
            INSUFFICIENT_EVIDENCE_MESSAGE
        )

        citations: list[Citation] = []

    else:

        try:

            generated_answer = (
                generate_answer(
                    question,
                    timeline,
                )
            )

            answer_text = (
                generated_answer.answer
            )

            # =================================================
            # 7. Evidence-first citation validation
            # =================================================

            evidence_ids = list(
                dict.fromkeys(
                    e["source_id"]
                    for e in timeline
                    if e.get("source_id")
                )
            )

            llm_cited_ids = [
                source_id
                for source_id
                in generated_answer.cited_source_ids
                if source_id in evidence_ids
            ]

            # If Gemini returns no valid IDs,
            # use actual evidence IDs.
            cited_ids = (
                llm_cited_ids
                or evidence_ids
            )

            citations = _build_citations(
                timeline,
                cited_ids,
            )

        except Exception as exc:  # noqa: BLE001

            answer_text = (
                INSUFFICIENT_EVIDENCE_MESSAGE
            )

            citations = []

            error = (
                error
                or f"answer_generation_failed: {exc}"
            )

    # =========================================================
    # 8. Final result
    # =========================================================

    return PipelineResult(
        mode=mode,
        question=question,
        cypher=generated.cypher,
        cypher_reasoning=generated.reasoning,
        cypher_valid=validation.is_valid,
        cypher_validation_reasons=validation.reasons,
        evidence=[
            dict(e)
            for e in timeline
        ],
        timeline=[
            dict(e)
            for e in timeline
        ],
        answer=answer_text,
        citations=citations,
        latency_seconds=(
            time.time() - start
        ),
        error=error,
    )