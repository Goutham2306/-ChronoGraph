"""
ChronoGraph - Day 2 CLI demo.

Run:
    python run_day2.py

Prompts for a question, runs the full Day 2 pipeline (Cypher generation ->
validation -> retrieval -> temporal sorting -> answer generation ->
citations), and prints each stage clearly.
"""

import json
import sys

from pipeline.rag_pipeline import answer_question


def _section(title: str) -> None:
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)


def run_once(question: str) -> None:
    result = answer_question(question)

    _section("MODE")
    print(f"MODE: {result.mode.upper()}")
    if result.mode == "mock":
        print("(mock mode: retrieving directly from output/graph.json, "
              "NOT Neo4j - see .env CHRONOGRAPH_MOCK_MODE)")

    _section("QUESTION")
    print(result.question)

    _section("GENERATED CYPHER")
    print(result.cypher or "(generation failed)")
    if result.cypher_reasoning:
        print(f"\nReasoning: {result.cypher_reasoning}")
    print(f"\nValid: {result.cypher_valid}")
    if result.cypher_validation_reasons:
        print("Validation issues:")
        for reason in result.cypher_validation_reasons:
            print(f"  - {reason}")

    _section("RETRIEVED EVIDENCE")
    if not result.evidence:
        print("(no evidence retrieved)")
    else:
        for e in result.evidence:
            print(
                f"[{e['timestamp']}] {e.get('subject_name', e.get('subject'))} "
                f"--{e.get('predicate', e.get('type', '?'))}--> "
                f"{e.get('object_name', e.get('object'))} "
                f"(source: {e['source']}/{e['source_id']}, "
                f"confidence: {e['confidence']})"
            )

    _section("CHRONOLOGICAL TIMELINE")
    if not result.timeline:
        print("(empty)")
    else:
        for e in result.timeline:
            print(f"{e['timestamp']}: {e.get('evidence', '')}")

    _section("FINAL ANSWER")
    print(result.answer)

    _section("CITATIONS")
    if not result.citations:
        print("(no citations)")
    else:
        for c in result.citations:
            print(f"[{c.source_id}, {c.timestamp}] (source: {c.source})")

    if result.error:
        print(f"\n[!] Pipeline reported an error: {result.error}")

    print(f"\nLatency: {result.latency_seconds:.2f}s")


def main() -> None:
    question = input("Enter your question: ").strip()
    if not question:
        print("No question entered.")
        sys.exit(1)
    run_once(question)


if __name__ == "__main__":
    main()
