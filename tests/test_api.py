from fastapi.testclient import TestClient

import api


client = TestClient(api.app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_rejects_empty_question():
    response = client.post(
        "/query",
        json={"question": ""},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "question must not be empty"


def test_query_rejects_whitespace_question():
    response = client.post(
        "/query",
        json={"question": "   "},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "question must not be empty"


def test_query_calls_pipeline(monkeypatch):
    def fake_answer_question(question):
        return {
            "question": question,
            "mode": "mock",
            "cypher": "",
            "cypher_reasoning": "",
            "cypher_valid": True,
            "cypher_validation_reasons": [],
            "evidence": [],
            "timeline": [],
            "answer": "Insufficient evidence in the available graph.",
            "citations": [],
            "error": None,
            "latency_seconds": 0.01,
        }

    monkeypatch.setattr(api, "answer_question", fake_answer_question)

    response = client.post(
        "/query",
        json={"question": "What happened after 2023-03-10?"},
    )

    assert response.status_code == 200
    assert response.json()["question"] == "What happened after 2023-03-10?"


def test_query_converts_pipeline_exception_to_500(monkeypatch):
    def failing_answer_question(question):
        raise RuntimeError("test pipeline failure")

    monkeypatch.setattr(api, "answer_question", failing_answer_question)

    response = client.post(
        "/query",
        json={"question": "What happened after 2023-03-10?"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "test pipeline failure"
