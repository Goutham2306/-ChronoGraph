"""
Answer generation for Day 2.

Receives the original question plus already-retrieved, already-sorted
evidence and asks Gemini (via LangChain structured output) to synthesize a
grounded narrative answer. The model is explicitly told to answer ONLY from
the evidence it's given and to say so plainly if evidence is insufficient -
it is never shown the whole graph, only the retrieved+filtered subset.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from pipeline.llm import get_qa_llm
from pipeline.temporal_retrieval import EvidenceRecord

INSUFFICIENT_EVIDENCE_MESSAGE = "Insufficient evidence in the available graph."


class GeneratedAnswer(BaseModel):
    answer: str = Field(
        description="The narrative answer, grounded only in the provided "
        f"evidence. If evidence is insufficient, this must be exactly: "
        f'"{INSUFFICIENT_EVIDENCE_MESSAGE}"'
    )
    cited_source_ids: list[str] = Field(
        description="source_id values (from the provided evidence only) "
        "that support claims made in the answer."
    )


def _format_evidence_for_prompt(evidence: list[EvidenceRecord]) -> str:
    if not evidence:
        return "(no evidence retrieved)"
    lines = []
    for i, record in enumerate(evidence, start=1):
        lines.append(
            f"{i}. [{record['timestamp']}] {record.get('subject_name', record['subject'])} "
            f"--{record['predicate']}--> {record.get('object_name', record['object'])} "
            f"| evidence: \"{record['evidence']}\" "
            f"| confidence: {record['confidence']} "
            f"| source: {record['source']} ({record['source_id']})"
        )
    return "\n".join(lines)


_PROMPT_TEMPLATE = """You are answering a question about ChronoGraph, a forensic \
historical engineering knowledge graph. Answer ONLY using the evidence listed \
below, in chronological order. Do not invent facts, people, dates, or \
relationships not present in this evidence.

Rules:
- If the evidence is empty or clearly insufficient to answer the question, \
your answer must be exactly: "{insufficient_message}"
- Explain events in chronological order when the question is historical.
- Every factual claim you make must be traceable to one of the numbered \
evidence items below. List the source_id of every evidence item you actually \
relied on in cited_source_ids.
- Do not cite a source_id that isn't in the evidence list below.

Question: "{question}"

Evidence (chronological order):
{evidence}
"""


def generate_answer(question: str, evidence: list[EvidenceRecord]) -> GeneratedAnswer:
    llm = get_qa_llm()
    structured_llm = llm.with_structured_output(GeneratedAnswer)
    prompt = _PROMPT_TEMPLATE.format(
        insufficient_message=INSUFFICIENT_EVIDENCE_MESSAGE,
        question=question,
        evidence=_format_evidence_for_prompt(evidence),
    )
    return structured_llm.invoke(prompt)
