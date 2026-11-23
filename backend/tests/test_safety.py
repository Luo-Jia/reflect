from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.workflow.safety import SAFETY_RESPONSE, assess_safety
from backend.workflow.prompt_workflow import answer_prompt
from backend.conversation.session_state import SessionState
from backend.workflow.workflow import graph


@pytest.mark.parametrize('message', [
    "I'm about to kill myself.",
    "I am in immediate danger.",
    "I have just taken an overdose.",
    "I can't breathe.",
    "I am not safe, but I am going to hurt myself now.",
])
def test_immediate_danger(message):
    assert assess_safety(message) == 'emergency'


@pytest.mark.parametrize('message', [
    "I am not going to kill myself.",
    "Last year I overdosed.",
    'Someone wrote "I am about to kill myself" in a book.',
    "What if I overdosed?",
    "I have thoughts of self-harm.",
])
def test_uncertain_or_historical_is_not_confirmed_emergency(message):
    assert assess_safety(message) == 'safety_check'


def test_regular_message():
    assert assess_safety('I feel anxious about a work meeting.') == 'standard_cbt'


def test_safety_followup_uses_history():
    assert assess_safety('yes', [{'role': 'assistant', 'content': SAFETY_RESPONSE}]) == 'emergency'
    assert assess_safety('yes', [{'role': 'assistant', 'content': 'Try an exercise?'}]) == 'standard_cbt'


def test_chat_bypasses_model_search_and_clears_exercise_consent():
    backend = SimpleNamespace(extract=Mock(), generate=Mock(), search=Mock())
    session = SessionState()
    session.record('hello', 'Try an exercise?', offer='thought_record')
    session.consent_status = 'granted'
    result = answer_prompt("I'm about to kill myself.", session=session, backend=backend)
    assert '911' in result['response']
    assert result['plan']['selected_path'] == []
    assert result['articles'] == []
    assert session.proposed_exercise is None
    assert session.accepted_exercise is None
    assert session.consent_status != 'granted'
    backend.extract.assert_not_called()
    backend.generate.assert_not_called()
    backend.search.assert_not_called()


def test_graph_bypasses_context_extraction():
    example = SimpleNamespace(user_message="I am in immediate danger.")
    result = graph.invoke({'example': example})
    assert result['decision_source'] == 'emergency'
    assert result['selected_path'] == []
    assert '911' in result['response']


def test_concern_followed_by_yes_skips_model():
    backend = SimpleNamespace(extract=Mock(), generate=Mock(), search=Mock())
    session = SessionState()
    first = answer_prompt('I have thoughts of self-harm.', session=session, backend=backend)
    assert first['plan']['status'] == 'safety_check'
    second = answer_prompt('yes', session=session, backend=backend)
    assert second['plan']['status'] == 'emergency'
    backend.extract.assert_not_called()
