"""
ChronoGraph - Day 1 entrypoint.

Run:
    python run_day1.py

Demonstrates the full Day 1 pipeline in one command:
    raw documents (data/*.txt)
        -> LlamaIndex document loading
        -> parsed source entries
        -> LLM entity + relationship extraction (schema-constrained)
        -> timestamped relationship triples
        -> output/graph.json  ({"nodes": [...], "edges": [...]})
"""

import json
import sys

from pipeline.extract import extract_all
from pipeline.graph_builder import build_graph, save_graph
from pipeline.loader import load_all_entries


def main() -> None:
    print("=" * 60)
    print("STAGE 1: Loading raw documents via LlamaIndex")
    print("=" * 60)
    entries = load_all_entries()
    print(f"Loaded {len(entries)} source entries from data/slack.txt, "
          f"data/github.txt, data/jira.txt\n")
    print("Example entry:")
    print(f"  {entries[0].source_id} | {entries[0].date} | {entries[0].source}")
    print(f"  {entries[0].text[:120]}...\n")

    print("=" * 60)
    print("STAGE 2 + 3: Extracting entities, relationships, and timestamps")
    print("=" * 60)
    triples = extract_all(entries)
    if not triples:
        print("\nNo relationships were extracted. Check GEMINI_API_KEY and try again.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("STAGE 4: Building graph triples -> JSON")
    print("=" * 60)
    graph = build_graph(triples)
    out_path = save_graph(graph)
    print(f"Nodes: {len(graph['nodes'])}")
    print(f"Edges: {len(graph['edges'])}")
    print(f"Saved graph to: {out_path}\n")

    print("Sample nodes:")
    print(json.dumps(graph["nodes"][:5], indent=2))
    print("\nSample edges:")
    print(json.dumps(graph["edges"][:3], indent=2))

    print("\nDay 1 pipeline complete: raw docs -> entities -> relationships "
          "-> timestamps -> triples -> JSON. See output/graph.json.")


if __name__ == "__main__":
    main()
