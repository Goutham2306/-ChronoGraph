import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.temporal_reasoning import (
    find_advocacy_then_usage,
    find_argument_then_usage,
    get_entity_history,
    get_events_after,
    get_events_before,
    get_events_between,
    summarize_entity_timeline,
)

GRAPH_PATH = Path(__file__).resolve().parent.parent / "output" / "graph.json"


def _real_records():
    graph = json.load(open(GRAPH_PATH))
    node_names = {n["id"]: n["name"] for n in graph["nodes"]}
    records = []
    for e in graph["edges"]:
        r = dict(e)
        r["subject_name"] = node_names[e["subject"]]
        r["object_name"] = node_names[e["object"]]
        records.append(r)
    return records


def test_get_entity_history_captures_both_name_variants():
    """
    Regression test for the entity-resolution gap: 'Person:Rohit' and
    'Person:Rohit Nair' are separate nodes in the real graph, but a
    history query for 'Rohit' must surface events touching either.
    """
    records = _real_records()
    history = get_entity_history(records, "Rohit")
    names_seen = {h["subject_name"] for h in history} | {h["object_name"] for h in history}
    matching = {n for n in names_seen if "rohit" in n.lower()}
    assert "Rohit Nair" in matching
    assert "Rohit" in matching


def test_get_entity_history_is_chronologically_sorted():
    records = _real_records()
    history = get_entity_history(records, "Rohit")
    timestamps = [h["timestamp"] for h in history]
    assert timestamps == sorted(timestamps)


def test_get_entity_history_checks_both_subject_and_object_position():
    records = _real_records()
    history = get_entity_history(records, "Rohit")
    # The one real edge where bare "Rohit" is the OBJECT, not subject.
    assert any(h["source_id"] == "SLACK-004" for h in history)


def test_get_events_before_excludes_cutoff_and_later():
    records = _real_records()
    before = get_events_before(records, "2023-03-01")
    assert len(before) > 0
    assert all(e["timestamp"] < "2023-03-01" for e in before)


def test_get_events_after_excludes_cutoff_and_earlier():
    records = _real_records()
    after = get_events_after(records, "2023-06-01")
    assert len(after) > 0
    assert all(e["timestamp"] > "2023-06-01" for e in after)


def test_get_events_between_is_inclusive_and_sorted():
    records = _real_records()
    between = get_events_between(records, "2023-01-01", "2023-03-31")
    assert len(between) > 0
    assert all("2023-01-01" <= e["timestamp"] <= "2023-03-31" for e in between)
    timestamps = [e["timestamp"] for e in between]
    assert timestamps == sorted(timestamps)


def test_find_advocacy_then_usage_finds_real_sequences():
    """
    The real data has Rohit advocating for GCP on 2023-01-10, and Karthik
    using GCP on 2023-02-14 - a genuine advocacy-then-implementation
    sequence that should be detected.
    """
    records = _real_records()
    pairs = find_advocacy_then_usage(records)
    assert len(pairs) > 0
    gcp_pair = next(
        (p for p in pairs if p["advocated"]["object_name"] == "GCP"
         and p["advocated"]["subject_name"] == "Rohit Nair"),
        None,
    )
    assert gcp_pair is not None
    assert gcp_pair["used"]["timestamp"] > gcp_pair["advocated"]["timestamp"]


def test_find_advocacy_then_usage_never_returns_usage_before_advocacy():
    records = _real_records()
    pairs = find_advocacy_then_usage(records)
    for p in pairs:
        assert p["used"]["timestamp"] > p["advocated"]["timestamp"]


def test_find_argument_then_usage_finds_real_sequences():
    records = _real_records()
    pairs = find_argument_then_usage(records)
    assert len(pairs) > 0
    for p in pairs:
        assert p["used_anyway"]["timestamp"] > p["argued_against"]["timestamp"]
        assert p["used_anyway"]["object"] == p["argued_against"]["object"]


def test_summarize_entity_timeline_shape():
    records = _real_records()
    summary = summarize_entity_timeline(records, "Rohit")
    assert summary["entity"] == "Rohit"
    assert summary["event_count"] == len(summary["history"])
    assert summary["first"]["timestamp"] <= summary["latest"]["timestamp"]


def test_summarize_entity_timeline_empty_for_unknown_entity():
    records = _real_records()
    summary = summarize_entity_timeline(records, "NoSuchPersonXYZ")
    assert summary["event_count"] == 0
    assert summary["first"] is None
    assert summary["history"] == []
