import json
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest

from backend.agents.supervisor import get_chat_graph
from backend.conversation.session_state import SessionState
from backend.conversation.user_profile import UserProfile
from backend.schemas.context import UserContext
from backend.workflow.prompt_workflow import answer_prompt, get_default_backend


def journal(profile_id='alice', entry_id='entry-a'):
    return dict(entry_id=entry_id, profile_id=profile_id, date='2026-11-20',
                title='Work reflection', content='Work felt stressful.', synthetic=True)


def critic(role, instructions, payload, schema):
    assert role == 'critic'
    return {'evaluations': [
        {'path': item['path'], 'emotional_fit': 0.9, 'readiness_fit': 0.8,
         'preference_fit': 0.7, 'cbt_fit': item['prior'], 'evidence_support': 0.5,
         'evidence_ids': [1], 'rationale': 'Fixture evaluation of supplied abstract.'}
        for item in payload['candidates']
    ]}


def make_backend():
    return SimpleNamespace(
        extract=Mock(return_value=UserContext(readiness='medium', intensity=0.5, emotions=['anxiety'])),
        search=Mock(return_value=[{'title': 'A CBT study', 'abstract': 'A supplied study abstract.'}]),
        delegate=Mock(side_effect=critic), generate=Mock(return_value='Here is a reflection.'),
    )


def test_langgraph_coordinates_four_agents_and_bounded_feedback():
    backend = make_backend()
    session = SessionState(profile_id='alice', journal_entries=[journal(), journal('bob', 'private-b')])
    result = answer_prompt('What does my journal say about work?', session=session, backend=backend,
                           profile=UserProfile(profile_id='alice'))
    nodes = get_chat_graph().get_graph().nodes
    assert {'context', 'planning', 'critic', 'response', 'dual_rag', 'output_validator'} <= nodes.keys()
    assert backend.delegate.call_count == 2
    assert all(call.args[0] == 'critic' for call in backend.delegate.call_args_list)
    for call in backend.delegate.call_args_list:
        payload = call.args[2]
        assert set(payload) == {'context', 'candidates', 'research', 'output_example'}
        assert 'private-b' not in json.dumps(payload)
        assert 'evidence' not in payload['context']
    payload = json.loads(backend.generate.call_args.args[0][-1]['content'])
    assert {entry['profile_id'] for entry in payload['journal_context']} == {'alice'}
    assert 'private-b' not in json.dumps(payload)
    assert result['decision_source'] == 'langgraph_four_agent'
    rounds = [item for item in result['agent_trace'] if item['agent'] == 'beam_search']
    assert [item['depth'] for item in rounds] == [1, 2]
    assert all(item['retained'] <= 3 for item in rounds)
    assert len(result['plan']['selected_path']) == 2
    assert result['plan']['selected_path'][0] == 'emotional_validation'
    assert result['plan']['candidate_evaluations']
    assert result['validation']['valid']


@pytest.mark.parametrize('bad', ['not json', {'evaluations': []}])
def test_invalid_critic_output_falls_back_to_contextual_scores(bad):
    backend = make_backend()
    backend.delegate.side_effect = None
    backend.delegate.return_value = bad
    result = answer_prompt('I am worried about work.', backend=backend)
    assert result['response']
    assert len(result['warnings']) == 2
    assert all(item['evidence_support'] == 0 for item in result['plan']['candidate_evaluations'])


def test_critic_repairs_once_then_continues_without_warning():
    backend = make_backend()
    count = 0
    def repair(role, instructions, payload, schema):
        nonlocal count
        count += 1
        if count == 1:
            return 'not JSON'
        if count == 2:
            assert payload['repair']['failure_category'] == 'missing_json_object'
        return critic(role, instructions, payload, schema)
    backend.delegate.side_effect = repair
    result = answer_prompt('I am worried about work.', backend=backend)
    assert not result['warnings']
    assert backend.delegate.call_count == 3  # One repair plus the second depth.


def test_critic_diagnostics_do_not_log_model_values(caplog):
    backend = make_backend()
    backend.delegate.side_effect = None
    backend.delegate.return_value = {'evaluations': [], 'PRIVATE_JOURNAL_VALUE': 'PRIVATE_CONTENT'}
    result = answer_prompt('Hello', backend=backend)
    assert 'schema_validation' in caplog.text
    assert 'PRIVATE_JOURNAL_VALUE' not in caplog.text
    assert 'PRIVATE_CONTENT' not in caplog.text
    assert len(result['warnings']) == 2
    assert backend.delegate.call_count == 4  # Two bounded attempts per depth.


def test_critic_runtime_failure_is_not_retried():
    backend = make_backend()
    backend.delegate.side_effect = RuntimeError('private runtime detail')
    result = answer_prompt('Hello', backend=backend)
    assert backend.delegate.call_count == 2  # One attempt per depth.
    assert all('generation_error' in warning for warning in result['warnings'])


def test_fabricated_critic_evidence_references_are_rejected():
    backend = make_backend()
    def invalid(role, instructions, payload, schema):
        result = critic(role, instructions, payload, schema)
        for item in result['evaluations']:
            item['evidence_ids'] = [999]
        return result
    backend.delegate.side_effect = invalid
    result = answer_prompt('I am worried about work.', backend=backend)
    assert all(item['evidence_ids'] == [] for item in result['plan']['candidate_evaluations'])
    assert len(result['warnings']) == 2


def test_critic_rejects_changed_or_duplicate_candidate_paths():
    backend = make_backend()
    def invalid(role, instructions, payload, schema):
        result = critic(role, instructions, payload, schema)
        result['evaluations'][0]['path'] = ['invented_intervention']
        return result
    backend.delegate.side_effect = invalid
    result = answer_prompt('I am worried about work.', backend=backend)
    assert result['plan']['selected_path'][0] == 'emotional_validation'
    assert len(result['warnings']) == 2


def test_safety_short_circuits_all_agents_but_still_validates_output():
    backend = make_backend()
    result = answer_prompt('I am in immediate danger.', backend=backend)
    assert [item['agent'] for item in result['agent_trace']] == ['safety_gate', 'output_validator']
    backend.extract.assert_not_called()
    backend.search.assert_not_called()
    backend.delegate.assert_not_called()
    backend.generate.assert_not_called()


def test_journal_retrieval_survives_context_failure_and_skips_planner():
    backend = make_backend()
    backend.extract.side_effect = ValueError('bad JSON')
    session = SessionState(profile_id='alice', journal_entries=[journal()])
    result = answer_prompt('Read my journal', session=session, backend=backend)
    backend.delegate.assert_not_called()
    backend.search.assert_not_called()
    payload = json.loads(backend.generate.call_args.args[0][-1]['content'])
    assert payload['journal_context']
    assert result['plan']['status'] == 'clarification'
    assert not any(item['agent'] == 'planning' for item in result['agent_trace'])


def test_no_consent_cannot_be_overridden_by_critic_scores():
    backend = make_backend()
    result = answer_prompt('Help me with work stress.', backend=backend)
    assert {tuple(item['path'])[0] for item in result['plan']['candidate_evaluations']} == {'emotional_validation'}
    assert result['context']['consent_to_exercise'] is False


def test_beam_expands_only_the_top_three_parents():
    backend = make_backend()
    backend.extract.return_value = UserContext(readiness='high', intensity=0.2, consent_to_exercise=True)
    result = answer_prompt('I consent to an exercise.', backend=backend)
    roots = backend.delegate.call_args_list[0].args[2]['candidates']
    leaves = backend.delegate.call_args_list[1].args[2]['candidates']
    assert len(roots) == 5
    assert len({item['path'][0] for item in leaves}) == 3
    assert len(leaves) <= 6
    assert result['plan']['max_depth'] == 2
    assert result['plan']['beam_width'] == 3


def test_agents_share_one_cached_model(monkeypatch):
    model = SimpleNamespace(device='cpu')
    tokenizer = object()
    generate = Mock(return_value='A response.')
    monkeypatch.setitem(sys.modules, 'backend.model.qwen_extractor', SimpleNamespace(
        extract_context=lambda *args, **kwargs: UserContext(needs_clarification=True),
        model=model, tokenizer=tokenizer,
    ))
    monkeypatch.setitem(sys.modules, 'backend.model.model_loader', SimpleNamespace(generate_response=generate))
    get_default_backend.cache_clear()
    try:
        backend = get_default_backend()
        assert get_default_backend() is backend
        backend.delegate('critic', 'Evaluate candidates.', {'candidates': []}, {})
        backend.generate([{'role': 'user', 'content': 'Hello'}])
        assert generate.call_count == 2
        assert all(call.args[0] is model and call.args[1] is tokenizer for call in generate.call_args_list)
    finally:
        get_default_backend.cache_clear()
