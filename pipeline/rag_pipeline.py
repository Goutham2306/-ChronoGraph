"""
Day 2 orchestrator.

question -> Cypher generation -> validation -> retrieval (Neo4j or mock) ->
temporal sorting -> answer generation -> citation validation -> result

Mode is explicit and always reported (MODE: NEO4J or MODE: MOCK) - mock
results are never presented as if they came from Neo4j, per spec.

Mock mode does NOT execute the generated Cypher (there's no in-memory
Cypher engine here) - it instead filters output/graph.json directly by the
entities mentioned in the question, using the same entity-matching logic
described to the LLM in the Cypher prompt (case-insensitive substring match
on names). This is a clearly separate code path from Neo4j execution, not a
disguised version of it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from pipeline.answer import INSUFFICIENT_EVIDENCE_MESSAGE, generate_answer
from pipeline.cypher_generator import generate_cypher
from pipeline.cypher_validator import validate_cypher
from pipeline.temporal_retrieval import EvidenceRecord, sort_chronologically

load_dotenv()

GRAPH_JSON_PATH = Path(__file__).resolve().parent.parent / "output" / "graph.json"


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
            f"{GRAPH_JSON_PATH} not found. Run `python run_day1.py` first."
        )
    with open(GRAPH_JSON_PATH) as f:
        return json.load(f)


def _extract_mentioned_terms(question: str) -> list[str]:
    """
    Very simple term extraction for mock-mode filtering: pulls out
    capitalized words/phrases and known technology-ish tokens from the
    question. This is intentionally simple - mock mode exists as a fallback
    demo path, not a reimplementation of NL understanding, and is clearly
    documented as such (Section 6 of the spec).
    """
    words = question.replace("?", "").replace(",", "").split()
    candidates = [w for w in words if w[:1].isupper() and len(w) > 2]
    return candidates


def _mock_retrieve(question: str) -> list[EvidenceRecord]:
    """
    Filters output/graph.json edges by whether any capitalized term from
    the question appears in the subject/object node names. Does NOT return
    the entire graph for every question (explicit spec requirement).
    """
    graph = _load_graph_json()
    node_names = {n["id"]: n["name"] for n in graph["nodes"]}
    terms = [t.lower() for t in _extract_mentioned_terms(question)]

    if not terms:
        return []

    matched: list[EvidenceRecord] = []
    for edge in graph["edges"]:
        subject_name = node_names.get(edge["subject"], "")
        object_name = node_names.get(edge["object"], "")
        haystack = f"{subject_name} {object_name}".lower()
        if any(term in haystack for term in terms):
            record: EvidenceRecord = {
                **edge,
                "subject_name": subject_name,
                "object_name": object_name,
            }
            matched.append(record)
    return matched


def _neo4j_retrieve(cypher: str) -> list[EvidenceRecord]:
    """
    Executes validated Cypher against a real Neo4j instance in a read-only
    transaction (second defense layer beneath validation).
    """
    from neo4j import GraphDatabase

    uri = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    if not all([uri, username, password]):
        raise RuntimeError(
            "Neo4j mode requires NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD "
            "to be set in .env."
        )

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:

            def _work(tx):
                result = tx.run(cypher)
                return [record.data() for record in result]

            rows = session.execute_read(_work)
    finally:
        driver.close()

    # Normalize whatever shape the generated RETURN clause produced into
    # EvidenceRecord-shaped dicts. This assumes the LLM followed the prompt's
    # instruction to return r.timestamp/r.source/etc. - if it used different
    # aliases, this normalization step is where that would need adjusting.
    normalized: list[EvidenceRecord] = []
    for row in rows:
        normalized.append(row)  # type: ignore[arg-type]
    return normalized


def _build_citations(evidence: list[EvidenceRecord], cited_ids: list[str]) -> list[Citation]:
    """
    Only builds a Citation for a source_id that both (a) the LLM claimed to
    cite AND (b) actually exists in the retrieved evidence. This is the
    citation validation step - prevents invented citations from reaching
    the user even if the LLM hallucinated one.
    """
    by_source_id = {e["source_id"]: e for e in evidence}
    citations = []
    for source_id in cited_ids:
        record = by_source_id.get(source_id)
        if record is None:
            continue  # LLM cited something not in evidence - drop it, don't trust it
        citations.append(
            Citation(
                source_id=source_id,
                timestamp=record["timestamp"],
                source=record["source"],
            )
        )
    return citations


def answer_question(question: str) -> PipelineResult:
    start = time.time()
    use_mock = os.environ.get("CHRONOGRAPH_MOCK_MODE", "false").lower() == "true"
    mode: Literal["neo4j", "mock"] = "mock" if use_mock else "neo4j"

    # --- Cypher generation (always happens, even in mock mode, so the
    # user/demo can see what a real query would look like) ---
    try:
        generated = generate_cypher(question)
    except Exception as exc:  # noqa: BLE001
        return PipelineResult(
            mode=mode,
            question=question,
            cypher=None,
            cypher_reasoning=None,
            cypher_valid=False,
            cypher_validation_reasons=[f"Cypher generation failed: {exc}"],
            evidence=[],
            timeline=[],
            answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            citations=[],
            latency_seconds=time.time() - start,
            error="cypher_generation_failed",
        )

    validation = validate_cypher(generated.cypher)

    evidence: list[EvidenceRecord] = []
    error = None

    if mode == "mock":
        evidence = _mock_retrieve(question)
    else:
        if not validation.is_valid:
            error = "cypher_validation_failed"
        else:
            try:
                evidence = _neo4j_retrieve(generated.cypher)
            except Exception as exc:  # noqa: BLE001
                error = "neo4j_unavailable"
                evidence = []

    timeline = sort_chronologically(evidence)

    if not timeline:
        answer_text = INSUFFICIENT_EVIDENCE_MESSAGE
        citations: list[Citation] = []
    else:
        try:
            generated_answer = generate_answer(question, timeline)
            answer_text = generated_answer.answer
            citations = _build_citations(timeline, generated_answer.cited_source_ids)
        except Exception as exc:  # noqa: BLE001
            answer_text = INSUFFICIENT_EVIDENCE_MESSAGE
            citations = []
            error = error or f"answer_generation_failed: {exc}"

    return PipelineResult(
        mode=mode,
        question=question,
        cypher=generated.cypher,
        cypher_reasoning=generated.reasoning,
        cypher_valid=validation.is_valid,
        cypher_validation_reasons=validation.reasons,
        evidence=[dict(e) for e in timeline],
        timeline=[dict(e) for e in timeline],
        answer=answer_text,
        citations=citations,
        latency_seconds=time.time() - start,
        error=error,
    )
