import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.temporal_retrieval import filter_by_date_range, sort_chronologically

GRAPH_PATH = Path(__file__).resolve().parent.parent / "output" / "graph.json"


def _real_edges():
    graph = json.load(open(GRAPH_PATH))
    return graph["edges"]


def test_sort_produces_ascending_order_on_real_data():
    edges = _real_edges()
    shuffled = edges.copy()
    random.seed(1)
    random.shuffle(shuffled)

    sorted_edges = sort_chronologically(shuffled)
    timestamps = [e["timestamp"] for e in sorted_edges]
    assert timestamps == sorted(timestamps)


def test_sort_preserves_all_records():
    edges = _real_edges()
    sorted_edges = sort_chronologically(edges)
    assert len(sorted_edges) == len(edges)


def test_malformed_timestamp_does_not_crash_and_is_isolated():
    edges = _real_edges()[:3]
    bad_record = dict(edges[0])
    bad_record["timestamp"] = "not-a-real-date"
    bad_record["source_id"] = "BAD-RECORD"

    result = sort_chronologically(edges + [bad_record])
    assert len(result) == 4
    assert result[-1]["source_id"] == "BAD-RECORD"


def test_date_range_filter_on_real_data():
    edges = _real_edges()
    q1 = filter_by_date_range(edges, start="2023-01-01", end="2023-03-31")
    assert len(q1) > 0
    assert all("2023-01-01" <= e["timestamp"] <= "2023-03-31" for e in q1)


def test_date_range_filter_excludes_out_of_range():
    edges = _real_edges()
    future = filter_by_date_range(edges, start="2030-01-01", end="2030-12-31")
    assert future == []
