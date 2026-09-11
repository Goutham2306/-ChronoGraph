"""
Document loading + entry parsing for ChronoGraph.

Uses LlamaIndex's SimpleDirectoryReader to ingest the raw mock data files
(data/slack.txt, data/github.txt, data/jira.txt) as Documents. Each file is
then split into discrete "entries" (one Slack message / one PR / one Jira
ticket) using the '---' delimited block format the mock data is written in.

Why not just chunk with a generic LlamaIndex node parser (e.g. sentence
splitter)? Because our source data has natural, semantically meaningful
boundaries (one message = one unit of evidence), and each entry already
carries a stable ID and DATE in its metadata block. Splitting on those
boundaries instead of arbitrary token windows gives us clean, ground-truth
source_id/timestamp values to attach to every extracted relationship later,
rather than trying to regex a date out of unstructured prose.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Matches one '---' delimited block, capturing its raw field lines and free text.
_ENTRY_PATTERN = re.compile(r"---\s*\n(.*?)(?=\n---|\Z)", re.DOTALL)
_FIELD_PATTERN = re.compile(r"^([A-Z_]+):\s?(.*)$")


@dataclass
class SourceEntry:
    """One discrete unit of evidence (a Slack message, a PR, a Jira ticket)."""

    source_id: str          # e.g. "SLACK-001", "PR-118", "CLOUD-52"
    source: str              # e.g. "slack", "github", "jira"
    date: str                 # ISO date string, e.g. "2023-01-10"
    author: str                # AUTHOR / REPORTER field
    entry_type: str             # TYPE field (e.g. "Pull Request", "Bug")
    title: str                   # TITLE field, if present
    text: str                     # concatenation of TEXT/DESCRIPTION fields
    raw_fields: dict = field(default_factory=dict)  # all fields, for debugging

    def as_llm_input(self) -> str:
        """Compact representation fed to the extraction LLM for this entry."""
        parts = [f"Date: {self.date}", f"Author: {self.author}"]
        if self.title:
            parts.append(f"Title: {self.title}")
        parts.append(f"Text: {self.text}")
        return "\n".join(parts)


def _parse_entries(raw_text: str, source: str) -> List[SourceEntry]:
    entries = []
    for block in _ENTRY_PATTERN.findall(raw_text):
        fields = {}
        text_buffer = []
        capturing_field = None
        for line in block.strip().splitlines():
            match = _FIELD_PATTERN.match(line)
            if match:
                key, value = match.group(1), match.group(2)
                fields[key] = value
                capturing_field = key
            elif capturing_field:
                # continuation of a multi-line field (rare in our mock data,
                # but handled so extraction never silently truncates text)
                fields[capturing_field] += " " + line.strip()

        if "ID" not in fields:
            continue  # skip malformed/empty blocks

        text = fields.get("TEXT", "") or fields.get("DESCRIPTION", "")
        entries.append(
            SourceEntry(
                source_id=fields["ID"],
                source=source,
                date=fields.get("DATE", ""),
                author=fields.get("AUTHOR") or fields.get("REPORTER", ""),
                entry_type=fields.get("TYPE", ""),
                title=fields.get("TITLE", ""),
                text=text,
                raw_fields=fields,
            )
        )
    return entries


def load_all_entries() -> List[SourceEntry]:
    """
    Loads data/*.txt via LlamaIndex's SimpleDirectoryReader (document
    processing layer), then parses each Document's text into individual
    SourceEntry records.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Expected mock data at {DATA_DIR}")

    documents: List[Document] = SimpleDirectoryReader(str(DATA_DIR)).load_data()

    all_entries: List[SourceEntry] = []
    for doc in documents:
        file_name = doc.metadata.get("file_name", "")
        source = Path(file_name).stem  # "slack", "github", or "jira"
        all_entries.extend(_parse_entries(doc.text, source=source))

    # Chronological order makes downstream debugging/printing easier to read.
    all_entries.sort(key=lambda e: (e.date, e.source_id))
    return all_entries


if __name__ == "__main__":
    entries = load_all_entries()
    print(f"Loaded {len(entries)} entries across data/*.txt\n")
    for e in entries[:3]:
        print(e)
