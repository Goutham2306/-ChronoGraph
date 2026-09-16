import pipeline.rag_pipeline as rag_pipeline


def test_end_to_end_temporal_query_in_mock_mode(monkeypatch):
    monkeypatch.setenv("CHRONOGRAPH_MOCK_MODE", "true")

    def fake_generate_answer(question, evidence):
        assert question == "What happened after 2023-03-10?"
        assert evidence

        return type(
            "GeneratedAnswer",
            (),
            {
                "answer": "Events occurred after the requested date.",
                "cited_source_ids": [evidence[0]["source_id"]],
            },
        )()

    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        fake_generate_answer,
    )

    result = rag_pipeline.answer_question(
        "What happened after 2023-03-10?"
    )

    assert result.mode == "mock"
    assert result.cypher
    assert result.cypher_valid is True
    assert result.evidence
    assert result.timeline
    assert result.answer == "Events occurred after the requested date."
    assert result.citations
    assert result.citations[0].source_id == result.evidence[0]["source_id"]


def test_end_to_end_unknown_question_returns_insufficient_evidence(monkeypatch):
    monkeypatch.setenv("CHRONOGRAPH_MOCK_MODE", "true")

    result = rag_pipeline.answer_question(
        "Tell me about CompletelyUnknownPerson999."
    )

    assert result.mode == "mock"
    assert result.evidence == []
    assert result.timeline == []
    assert result.answer == "Insufficient evidence in the available graph."
    assert result.citations == []


def test_end_to_end_entity_history_in_mock_mode(monkeypatch):
    monkeypatch.setenv("CHRONOGRAPH_MOCK_MODE", "true")

    def fake_generate_answer(question, evidence):
        assert question == "Show the history of Rohit Nair"
        assert evidence

        return type(
            "GeneratedAnswer",
            (),
            {
                "answer": "Rohit Nair has documented historical events.",
                "cited_source_ids": [evidence[0]["source_id"]],
            },
        )()

    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        fake_generate_answer,
    )

    result = rag_pipeline.answer_question(
        "Show the history of Rohit Nair"
    )

    assert result.mode == "mock"
    assert result.evidence
    assert result.timeline
    assert result.answer == "Rohit Nair has documented historical events."
    assert result.citations