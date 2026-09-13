from pipeline.temporal_router import classify_temporal_question


def test_after_route():
    route = classify_temporal_question(
        "What happened after 2023-03-10?"
    )

    assert route.intent == "AFTER"
    assert route.date == "2023-03-10"


def test_before_route():
    route = classify_temporal_question(
        "What happened before 2023-03-10?"
    )

    assert route.intent == "BEFORE"
    assert route.date == "2023-03-10"


def test_between_route():
    route = classify_temporal_question(
        "What happened between 2023-03-01 and 2023-03-20?"
    )

    assert route.intent == "BETWEEN"
    assert route.start_date == "2023-03-01"
    assert route.end_date == "2023-03-20"


def test_entity_history_route():
    route = classify_temporal_question(
        "Show the history of Rohit Nair"
    )

    assert route.intent == "ENTITY_HISTORY"
    assert route.entity == "Rohit Nair"


def test_project_history_route():
    route = classify_temporal_question(
        "Show the project history of Q2 2023 analytics pilot"
    )

    assert route.intent == "PROJECT_HISTORY"
    assert route.entity == "Q2 2023 analytics pilot"


def test_advocacy_then_usage_route():
    route = classify_temporal_question(
        "Was something advocated for and later used?"
    )

    assert route.intent == "ADVOCACY_THEN_USAGE"


def test_argument_then_usage_route():
    route = classify_temporal_question(
        "Was something argued against but later used anyway?"
    )

    assert route.intent == "ARGUMENT_THEN_USAGE"


def test_general_route():
    route = classify_temporal_question(
        "Who advocated for the Q2 2023 analytics pilot?"
    )

    assert route.intent == "GENERAL"