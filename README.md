# ChronoGraph AI Pipeline — Goutham's Component

Temporal GraphRAG extraction + retrieval pipeline for ChronoGraph. This repo owns
everything upstream of the Neo4j/FastAPI backend (Ullas) and the Next.js frontend
(Ayush): document extraction, graph triple generation, and (Day 2) natural-language
question answering over the graph.

## Status

- **Day 1: done.** Mock data + LlamaIndex-based extraction pipeline producing
  `output/graph.json`.
- **Day 2: done (see below).** LangChain + Gemini Cypher generation, Neo4j
  retrieval (with a clearly-separated mock fallback), temporal ordering,
  grounded answer generation, and citation validation.

## Setup

```bash
pip install -r requirements.txt --break-system-packages   # or use a venv
cp .env.example .env
# edit .env: add your GEMINI_API_KEY, and NEO4J_URI/USERNAME/PASSWORD
```

## Day 1: Run the extraction pipeline

```bash
python run_day1.py
```

This runs, in one command:

```
data/*.txt (raw Slack/Git/Jira mock data)
    -> LlamaIndex SimpleDirectoryReader (document loading)
    -> parsed source entries (pipeline/loader.py)
    -> LLM entity + relationship extraction (pipeline/extract.py)
    -> timestamped relationship triples
    -> output/graph.json  (pipeline/graph_builder.py)
```

Expected output (abridged):

```
============================================================
STAGE 1: Loading raw documents via LlamaIndex
============================================================
Loaded 32 source entries from data/slack.txt, data/github.txt, data/jira.txt

============================================================
STAGE 2 + 3: Extracting entities, relationships, and timestamps
============================================================
[1/32] extracting SLACK-001 (2023-01-10)...
    Rohit Nair --ADVOCATED_FOR--> GCP  (timestamp=2023-01-10, confidence=0.85)
...
============================================================
STAGE 4: Building graph triples -> JSON
============================================================
Nodes: ~15
Edges: ~30+
Saved graph to: output/graph.json
```

`output/graph.json` has the shape:

```json
{
  "nodes": [{"id": "Person:Rohit Nair", "name": "Rohit Nair", "type": "Person"}],
  "edges": [{
    "id": "edge:0",
    "subject": "Person:Rohit Nair",
    "predicate": "ADVOCATED_FOR",
    "object": "Technology:GCP",
    "timestamp": "2023-03-02",
    "source": "slack",
    "source_id": "SLACK-008",
    "evidence": "advocating we move forward: migrate analytics workloads",
    "confidence": 0.95
  }]
}
```

This is the exact shape Ullas's Neo4j loader can consume: `nodes` become `MERGE`d
graph nodes by `(id)`, `edges` become relationships of type `predicate` between
`subject` and `object`, carrying `timestamp`/`source`/`source_id`/`evidence`/
`confidence` as relationship properties.

### Mock data

`data/slack.txt`, `data/github.txt`, `data/jira.txt` simulate a real 2023 AWS→GCP
migration: Rohit advocating for GCP, Priya raising integration risk and arguing
against a full migration, Karthik committing infra/pipeline code, Suresh reviewing
PRs, and Meera reporting/fixing a real schema-mismatch bug (CLOUD-52) mid-migration.
Each entry is a `---`-delimited block with `ID`, `DATE`, `AUTHOR`/`REPORTER`, `TYPE`,
`TITLE`, and `TEXT`/`DESCRIPTION` fields — this mirrors how real Slack/Git/Jira
exports carry stable IDs and timestamps, which we rely on directly for the
`timestamp`/`source_id` fields on every edge rather than trying to parse dates out
of free text.

## Day 2: Load the graph into Neo4j

```bash
python -m pipeline.neo4j_loader
```

Requires `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` in `.env` (clear error
if any are missing — never silently fakes success). Reads `output/graph.json`
and loads it into Neo4j.

**Idempotent** — safe to re-run without duplicating anything. Nodes are
`MERGE`d on their existing `id` field (e.g. `"Person:Rohit Nair"`). Relationships
are **not** `MERGE`d on just `(subject)-[:PREDICATE]->(object)` — that would
collapse every historical event between the same two entities into a single
edge and destroy the temporal history that's the point of this project.
Instead each relationship is keyed by `event_key` (`source_id + timestamp +
edge id`), so distinct events between the same two entities stay separately
retrievable, while re-running the loader on the same `graph.json` doesn't
create duplicates.

## Day 2: Ask a question (CLI)

```bash
python run_day2.py
```

Prompts `Enter your question:`, then prints, in order: mode (`NEO4J` or
`MOCK`), the generated Cypher and whether it passed validation, retrieved
evidence, the chronological timeline, the final grounded answer, and
citations.

**Mock mode**: set `CHRONOGRAPH_MOCK_MODE=true` in `.env` to answer questions
directly from `output/graph.json` instead of Neo4j — useful if Neo4j setup
becomes a blocker before a demo. Mock mode filters the real graph by entities
named in the question (case-insensitive substring match); it does **not**
return the whole graph for every question, and the CLI/API always report
which mode actually produced the answer — mock results are never presented as
if they came from Neo4j.

### Example questions

```
Why did the team migrate from AWS to GCP?
What did Rohit advocate for, and when?
Who argued against AWS?
What technologies did the team use during the migration?
Show the chronological history of the AWS to GCP migration.
```

## Day 2: Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

```
GET  /health                 -> {"status": "ok"}
POST /query
     {"question": "Why did the team migrate from AWS to GCP?"}
     -> {"mode", "question", "cypher", "evidence", "timeline", "answer", "citations", ...}
```

Both endpoints have been started and hit live in development (not just
import-checked) — see the testing note below for what "tested" means for the
Gemini-calling parts specifically.

## Running tests

```bash
pytest tests/ -v
```

23 deterministic tests covering Cypher validation (including all 6 real
predicates and all 3 real labels), temporal sorting/filtering on the real
72-edge graph, mock retrieval (including a regression test for the
`Rohit`/`Rohit Nair` duplicate-node issue), and citation validation
(confirming a fabricated/uncited source_id is dropped, not trusted). No live
Gemini calls in the test suite, per the "avoid unnecessary API calls" guidance
— run the CLI yourself with a real key to exercise those paths.

## Known issues / remaining work

- **Not yet live-tested against a real Neo4j instance or the live Gemini
  API** in the environment this was built in (no network access to either
  from that sandbox). The Cypher templates, validator, and prompts are
  verified correct and schema-consistent, but you should run
  `python -m pipeline.neo4j_loader` and `python run_day2.py` yourself first,
  before a live demo, to confirm.
- `pipeline/rag_pipeline.py::_neo4j_retrieve` trusts the LLM to alias its
  `RETURN` clause exactly as instructed in the Cypher-generation prompt
  (`subject_name`, `predicate`, `timestamp`, etc.). The prompt is explicit
  about this, but it hasn't been confirmed against live model output — if
  answers look empty/malformed in Neo4j mode, check this first.
- Mock-mode retrieval uses simple capitalized-word matching and returns no
  results for questions that don't name a specific entity by a proper noun
  (e.g. "what technologies did the team use" — matches nothing, since
  "technologies"/"team" aren't capitalized). Real Neo4j mode doesn't have
  this limitation, since the LLM understands the question semantically.
  Prefer Neo4j mode for the full MVP question set.
- Entity resolution (duplicate person nodes) is worked around at query time
  via substring matching, not fixed at the source.



ChronoGraph deliberately splits responsibilities between the two frameworks instead
of using either one for everything:

| Layer | Framework | Responsibility |
|---|---|---|
| Ingestion & extraction | **LlamaIndex** | Loading raw documents (`SimpleDirectoryReader`), structured entity/relationship extraction, graph-oriented information extraction into triples |
| Question answering | **LangChain** | Question routing/classification, natural-language → Cypher generation, backend retrieval, chronological reasoning, answer generation, citations, conversation flow |

LlamaIndex is used where the job is "turn unstructured documents into structured
graph data." LangChain is used where the job is "turn a user's question into a
graph query, then turn graph results into an answer." Keeping this split explicit
means the extraction pipeline (Day 1) and the retrieval/reasoning pipeline (Day 2)
can be built, tested, and debugged independently — and it matches how each
framework's tooling is actually strongest.

## Repo layout

```
chronograph-ai-pipeline/
├── data/                     # mock Slack/GitHub/Jira source data
│   ├── slack.txt
│   ├── github.txt
│   └── jira.txt
├── pipeline/
│   ├── loader.py             # Day 1: LlamaIndex document loading + entry parsing
│   ├── extract.py            # Day 1: schema-constrained entity/relationship extraction
│   ├── graph_builder.py      # Day 1: triples -> nodes/edges -> output/graph.json
│   ├── llm.py                # Day 2: centralized LangChain + Gemini client
│   ├── cypher_validator.py   # Day 2: read-only Cypher safety gate (no LLM)
│   ├── cypher_generator.py   # Day 2: NL -> Cypher (LangChain structured output)
│   ├── temporal_retrieval.py # Day 2: chronological sorting + date-range filtering
│   ├── answer.py             # Day 2: grounded answer synthesis + citation claims
│   ├── neo4j_loader.py       # Day 2: idempotent graph.json -> Neo4j loader
│   └── rag_pipeline.py       # Day 2: orchestrator (question -> answer + citations)
├── output/
│   └── graph.json            # generated by run_day1.py
├── tests/
│   ├── test_cypher_validator.py
│   ├── test_temporal_retrieval.py
│   └── test_rag_pipeline.py
├── run_day1.py                # single-command Day 1 demo
├── run_day2.py                # single-command Day 2 CLI demo
├── api.py                     # minimal FastAPI: GET /health, POST /query
├── requirements.txt
└── .env.example
```

## Schema (shared by extraction and Day 2 Cypher generation)

**Node labels:** `Person`, `Technology`, `Project`
**Relationship types:** `ADVOCATED_FOR`, `COMMITTED_CODE`, `ARGUED_AGAINST`, `USED`,
`REPORTED`, `REVIEWED`

Verified against the actual generated `output/graph.json` (32 nodes, 72 edges) —
not assumed. Every edge carries the same key-set:
`subject, subject_type* (implicit via node id prefix), predicate, object, timestamp,
source, source_id, evidence, confidence`.

The extraction LLM is constrained to this schema via a Pydantic model with
`Literal` types — it cannot emit an entity type or relationship type outside this
list. Day 2's Cypher generator is constrained to the same schema (see
`pipeline/cypher_validator.py`'s `ALLOWED_LABELS`/`ALLOWED_RELATIONSHIP_TYPES`) so
retrieval never asks the graph about something extraction could never have
produced.

**Known data quality gap:** node dedup in `graph_builder.py` keys on exact
`(name, type)`, and extraction doesn't canonicalize names per-entry, so the same
real person sometimes ends up as two separate nodes — e.g. `Person:Rohit Nair`
and `Person:Rohit` are distinct nodes in the current graph. Day 2 works around
this at query time (case-insensitive substring matching on person names) rather
than fixing it at the source; a proper fix would canonicalize names during
extraction.

## Integration with the backend

Day 1 produces `output/graph.json`. Day 2 (`pipeline/neo4j_loader.py`) loads it
directly into Neo4j — this repo does talk to Neo4j directly as of Day 2, which
supersedes the original plan of a separate team member owning that load step.
The loader is idempotent and designed to preserve distinct historical events
between the same two entities (see Day 2 section below) rather than collapsing
them via a naive `MERGE`.
