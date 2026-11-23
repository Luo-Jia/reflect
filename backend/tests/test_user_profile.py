import json
import pytest
from types import SimpleNamespace
from unittest.mock import Mock

from backend.schemas.context import UserContext
from backend.workflow.prompt_workflow import answer_prompt
from backend.conversation.session_state import SessionState
from backend.conversation.user_profile import UserProfile, load_profile, load_profiles


def test_profile_loading_and_no_automatic_writes(tmp_path):
    path = tmp_path / 'profile.json'
    assert load_profile(path).preferred_name == ''
    assert not path.exists()
    path.write_text(json.dumps({'preferred_name': 'Test', 'goals': ['sleep better']}))
    assert load_profile(path).goals == ['sleep better']


def test_structured_profile_format():
    profile = UserProfile(
        response_preferences={'length': 'short'},
        confirmed_facts=[{'fact': 'Example fact', 'source': 'user_confirmed'}],
    )
    assert profile.model_dump()['confirmed_facts'][0]['fact'] == 'Example fact'


def test_profile_array_selection(tmp_path):
    path = tmp_path / 'profiles.json'
    path.write_text(json.dumps([
        {'profile_id': '1', 'preferred_name': 'First'},
        {'profile_id': '2', 'preferred_name': 'Second'},
    ]))
    assert len(load_profiles(path)) == 2
    assert load_profile(path).profile_id == '1'
    assert load_profile(path, profile_id='2').preferred_name == 'Second'
    with pytest.raises(ValueError, match='not found'):
        load_profile(path, profile_id='missing')


@pytest.mark.parametrize('data', [[], [{'profile_id': '1'}, {'profile_id': '1'}], 'invalid'])
def test_invalid_profile_array(tmp_path, data):
    path = tmp_path / 'profiles.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_profiles(path)


def test_profile_and_recap_reach_generation_without_consent():
    session = SessionState()
    for index in range(20):
        session.record(f'Past message {index}', 'Acknowledged.')
    assert len(session.history) == 12
    assert len(session.earlier_user_messages) == 8
    backend = SimpleNamespace(
        extract=Mock(return_value=UserContext(needs_clarification=True)),
        generate=Mock(return_value='Hello.'), search=Mock(),
    )
    profile = UserProfile(preferred_name='Test', goals=['try an exercise'])
    answer_prompt('Hello', session=session, backend=backend, profile=profile)
    messages = backend.generate.call_args.args[0]
    payload = json.loads(messages[-1]['content'])
    assert payload['user_profile']['preferred_name'] == 'Test'
    assert payload['session']['earlier_user_messages']
    assert payload['context']['consent_to_exercise'] is False
    assert session.consent_status == 'unknown'
    backend.search.assert_not_called()
