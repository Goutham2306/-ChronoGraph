"""
Entity + relationship extraction for ChronoGraph.

For each SourceEntry (one Slack message / PR / Jira ticket), we call an LLM
through LlamaIndex's structured-output interface and force it to return a
Pydantic-typed EntryExtraction: only entities/relationships that match the
ChronoGraph schema are allowed. This is the same schema the Day 2 Cypher
generator will be constrained to, so extraction and retrieval never drift
apart.

Timestamp handling: every SourceEntry already carries a ground-truth DATE
from its source system (Slack message timestamp, PR date, Jira ticket date).
We attach that as the timestamp on every relationship extracted from that
entry, rather than asking the LLM to re-derive a date from prose - that
would be strictly less reliable for an MVP and isn't necessary since our
mock data (like real Slack/Git/Jira exports) always has real timestamps.
"""

import json
import os
from typing import List, Literal

from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from pydantic import BaseModel, Field

from pipeline.loader import SourceEntry

load_dotenv()

EntityType = Literal["Person", "Technology", "Project"]
RelationType = Literal[
    "ADVOCATED_FOR",
    "COMMITTED_CODE",
    "ARGUED_AGAINST",
    "USED",
    "REPORTED",
    "REVIEWED",
]


class ExtractedEntity(BaseModel):
    name: str = Field(description="Canonical name, e.g. 'Rohit Nair' or 'GCP'")
    type: EntityType


class ExtractedRelationship(BaseModel):
    subject: str = Field(description="Name of the source entity, usually a Person")
    subject_type: EntityType
    predicate: RelationType
    object: str = Field(description="Name of the target entity")
    object_type: EntityType
    evidence: str = Field(
        description="Short verbatim-or-near-verbatim snippet from the entry text "
        "that justifies this relationship"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model's confidence this relationship is correct"
    )


class EntryExtraction(BaseModel):
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]


SYSTEM_PROMPT = """You are an information extraction engine for ChronoGraph, a temporal \
graph system tracking engineering decisions.

You will be given one source entry (a Slack message, a GitHub PR/commit, or a Jira \
ticket). Extract ONLY entities and relationships that are explicitly stated or clearly \
implied by the text. Do not invent facts.

Allowed entity types: Person, Technology, Project
Allowed relationship types (predicate): ADVOCATED_FOR, COMMITTED_CODE, ARGUED_AGAINST, \
USED, REPORTED, REVIEWED

Rules:
- A relationship's subject is almost always a Person; object is usually a Technology \
or Project.
- ADVOCATED_FOR: subject argues in favor of adopting/using something.
- ARGUED_AGAINST: subject argues against adopting/using something.
- COMMITTED_CODE: subject wrote/pushed a code change.
- USED: subject used a technology in their work (e.g. "used Terraform").
- REPORTED: subject filed/reported an issue, bug, or ticket.
- REVIEWED: subject reviewed someone else's work (a PR, a proposal).
- If the entry has no clear relationships, return an empty relationships list.
- evidence must be a short quote or close paraphrase (under ~20 words) from the entry \
text, not the whole entry.
- confidence should reflect how explicit the statement is (0.9+ for direct statements, \
0.5-0.7 for implied ones).
"""


def _build_llm() -> GoogleGenAI:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add your Gemini key to .env."
        )

    model = os.environ.get(
        "CHRONOGRAPH_EXTRACTION_MODEL",
        "gemini-2.5-flash"
    )

    return GoogleGenAI(
        model=model,
        api_key=api_key,
        temperature=0,
    )


def extract_from_entry(entry: SourceEntry, llm: GoogleGenAI) -> EntryExtraction:
    """Runs schema-constrained extraction on a single SourceEntry."""
    structured_llm = llm.as_structured_llm(output_cls=EntryExtraction)
    prompt = f"{SYSTEM_PROMPT}\n\nSource entry:\n{entry.as_llm_input()}"
    response = structured_llm.complete(prompt)
    return response.raw  # parsed EntryExtraction instance


def extract_all(entries: List[SourceEntry], verbose: bool = True) -> List[dict]:
    """
    Runs extraction over every entry and returns a flat list of dicts, each
    a fully-resolved relationship record ready for graph_builder:

        {
            "subject": ..., "subject_type": ...,
            "predicate": ..., "object": ..., "object_type": ...,
            "evidence": ..., "confidence": ...,
            "timestamp": entry.date, "source": entry.source, "source_id": entry.source_id,
        }
    """
    llm = _build_llm()
    all_relationships: List[dict] = []

    for i, entry in enumerate(entries, start=1):
        if verbose:
            print(f"[{i}/{len(entries)}] extracting {entry.source_id} ({entry.date})...")
        try:
            result = extract_from_entry(entry, llm)
        except Exception as exc:  # keep the pipeline running even if one call fails
            print(f"  ! extraction failed for {entry.source_id}: {exc}")
            continue

        for rel in result.relationships:
            all_relationships.append(
                {
                    "subject": rel.subject,
                    "subject_type": rel.subject_type,
                    "predicate": rel.predicate,
                    "object": rel.object,
                    "object_type": rel.object_type,
                    "evidence": rel.evidence,
                    "confidence": rel.confidence,
                    "timestamp": entry.date,
                    "source": entry.source,
                    "source_id": entry.source_id,
                }
            )
        if verbose and result.relationships:
            for rel in result.relationships:
                print(
                    f"    {rel.subject} --{rel.predicate}--> {rel.object}"
                    f"  (timestamp={entry.date}, confidence={rel.confidence:.2f})"
                )

    return all_relationships


if __name__ == "__main__":
    from pipeline.loader import load_all_entries

    entries = load_all_entries()
    triples = extract_all(entries)
    print(f"\nExtracted {len(triples)} relationships total.")
    print(json.dumps(triples[:3], indent=2))
