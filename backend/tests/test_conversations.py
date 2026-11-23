import json

from backend.schemas.context import UserContext
from backend.evaluation.evaluate_conversations import DEFAULT_CASES, run_cases, check_turn
from backend.conversation.session_state import SessionState


def test_fixture_conversations():
    report = run_cases(json.loads(DEFAULT_CASES.read_text()))
    assert report["turns"] >= 10
    assert report["failed_turns"] == 0, report["records"]


def test_acceptance_is_scoped_and_not_reused():
    session = SessionState()
    session.record("I am worried", "Try a thought record?", offer="thought_record")
    context = UserContext(readiness="medium")
    session.resolve_reply("yes", context)
    assert context.consent_to_exercise
    assert session.accepted_exercise == "thought_record"
    session.resolve_reply("Now about something else", context.model_copy(update={"consent_to_exercise": False}))
    assert session.consent_status == "unknown"
    assert session.accepted_exercise is None


def test_history_bound_and_pending_question():
    session = SessionState()
    for _ in range(20):
        session.record("hello", "What happened?")
    assert len(session.history) == 12
    assert session.turn_count == 20
    assert session.pending_question == "What happened?"


def test_harness_detects_unconsented_plan():
    result = {"session": {"consent_status": "unknown", "proposed_exercise": None,
                          "topic": None, "turn_count": 1, "accepted_exercise": None},
              "context": {"consent_to_exercise": False},
              "plan": {"status": "planned", "selected_path": ["thought_record"]},
              "warnings": [], "response": "test"}
    assert "Exercise planned without consent" in check_turn(result, {})
