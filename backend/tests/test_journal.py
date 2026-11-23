from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.conversation.session_state import SessionState
from backend.ingestion.ingest_journal import ingest_journal, load_journal as load_sample_journal
from backend.retrieval.journal_retriever import retrieve_journal
from backend.schemas.context import UserContext
from backend.workflow.prompt_workflow import answer_prompt


def load_journal():
    # Keep test ownership independent of the locally edited sample profile.
    return [{**entry, 'profile_id': 'local-user', 'synthetic': True} for entry in load_sample_journal()]


def test_mock_ingestion_is_idempotent_and_updates_by_id():
    session = SessionState()
    entries = load_journal()
    assert len(entries) == 5
    assert all(entry['synthetic'] for entry in entries)
    assert ingest_journal(session, entries) == 5
    assert ingest_journal(session, entries) == 5
    entries[0]['content'] = 'Updated example text.'
    assert ingest_journal(session, entries[:1]) == 5
    assert session.journal_entries[0]['content'] == 'Updated example text.'
    assert 'journal_entries' not in session.summary()


@pytest.mark.parametrize('change', ['duplicate', 'blank', 'date', 'oversize'])
def test_bad_import_is_atomic(change):
    session = SessionState()
    ingest_journal(session, load_journal())
    before = deepcopy(session.journal_entries)
    entries = load_journal()
    if change == 'duplicate':
        entries.append(entries[0])
    elif change == 'blank':
        entries[0]['content'] = '   '
    elif change == 'date':
        entries[0]['date'] = 'not a date'
    else:
        entries[0]['content'] = 'x' * 4001
    with pytest.raises(ValueError):
        ingest_journal(session, entries)
    assert session.journal_entries == before


def test_retrieval_relevance_and_generic_journal_query():
    entries = load_journal()
    assert retrieve_journal('sleep and phone', entries)[0]['entry_id'] == 'demo-004'
    assert len(retrieve_journal('summarize the journal', entries)) == 3
    assert retrieve_journal('volcano geology', entries) == []


@pytest.mark.parametrize('query', [
    'Find my Nov 16 journal entry', 'What did I write on November 16?',
    'Read November 16th, 2026', 'My entry on 16 Nov 2026',
    'What happened on the 16th of November?', 'Read Nov. 16, 2026',
])
def test_month_name_date_retrieval(query):
    entries = retrieve_journal(query, load_journal())
    assert [entry['date'] for entry in entries] == ['2026-11-16']


def test_missing_named_date_does_not_fall_back_to_unrelated_entries():
    assert retrieve_journal('My November 21 entry', load_journal()) == []
    assert retrieve_journal('My November 16, 2025 entry', load_journal()) == []


def test_journal_reaches_reply_without_requiring_context_extraction_success():
    session = SessionState()
    ingest_journal(session, load_journal())
    backend = SimpleNamespace(extract=Mock(side_effect=ValueError('invalid JSON')),
                              generate=Mock(return_value='A demo reflection.'), search=Mock())
    result = answer_prompt('What does the journal say about sleep?', session=session, backend=backend)
    payload = json.loads(backend.generate.call_args.args[0][-1]['content'])
    assert payload['journal_context'][0]['entry_id'] == 'demo-004'
    assert payload['journal_context'][0]['synthetic'] is True
    assert payload['research'] == []
    assert result['context']['consent_to_exercise'] is False
    backend.search.assert_not_called()


def test_old_diary_does_not_override_current_safety():
    session = SessionState()
    ingest_journal(session, load_journal())
    backend = SimpleNamespace(extract=Mock(), generate=Mock(), search=Mock())
    result = answer_prompt('I am in immediate danger.', session=session, backend=backend)
    assert result['plan']['status'] == 'emergency'
    backend.extract.assert_not_called()
    backend.generate.assert_not_called()


def test_import_binds_unowned_entries_to_session_profile():
    session = SessionState(profile_id='another-user')
    entries = load_journal()
    for entry in entries:
        entry.pop('profile_id')
    ingest_journal(session, entries)
    assert all(entry['profile_id'] == 'another-user' for entry in session.journal_entries)


def test_foreign_profile_import_rejected_atomically():
    session = SessionState(profile_id='another-user')
    with pytest.raises(ValueError, match='different profile'):
        ingest_journal(session, load_journal())
    assert session.journal_entries == []


def test_chat_filters_foreign_entries_and_rejects_profile_mismatch():
    from backend.conversation.user_profile import UserProfile
    session = SessionState()
    ingest_journal(session, load_journal())
    session.journal_entries.append({**session.journal_entries[0], 'profile_id': 'someone-else', 'entry_id': 'foreign'})
    backend = SimpleNamespace(extract=Mock(return_value=UserContext(needs_clarification=True)),
                              generate=Mock(return_value='A reflection.'), search=Mock())
    answer_prompt('Summarize the journal', session=session, backend=backend, profile=UserProfile())
    payload = json.loads(backend.generate.call_args.args[0][-1]['content'])
    assert all(entry['profile_id'] == 'local-user' for entry in payload['journal_context'])
    with pytest.raises(ValueError, match='do not match'):
        answer_prompt('Hello', session=session, backend=backend, profile=UserProfile(profile_id='someone-else'))


def test_profile_journal_loading_filters_other_profiles(tmp_path):
    from backend.ingestion.ingest_journal import load_profile_journal
    entries = load_journal()
    entries[0]['profile_id'] = 'someone-else'
    path = tmp_path / 'journal.json'
    path.write_text(json.dumps(entries))
    assert len(load_profile_journal('local-user', path)) == 4
    assert load_profile_journal('missing', path) == []
    assert load_profile_journal('local-user', tmp_path / 'missing.json') == []


def test_journal_questions_include_catalog_and_access_counts():
    session = SessionState()
    ingest_journal(session, load_journal())
    backend = SimpleNamespace(extract=Mock(return_value=UserContext(needs_clarification=True)),
                              generate=Mock(return_value='Here are the entries.'), search=Mock())
    answer_prompt('List my journal entries', session=session, backend=backend)
    payload = json.loads(backend.generate.call_args.args[0][-1]['content'])
    assert payload['journal_access']['available_count'] == 5
    assert payload['journal_access']['retrieved_count'] == 3
    assert len(payload['journal_access']['catalog']) == 5
    answer_prompt('My journal entry on 2026-11-19', session=session, backend=backend)
    payload = json.loads(backend.generate.call_args.args[0][-1]['content'])
    assert [entry['entry_id'] for entry in payload['journal_context']] == ['demo-004']
    answer_prompt('My diary on 2020-01-01', session=session, backend=backend)
    payload = json.loads(backend.generate.call_args.args[0][-1]['content'])
    assert payload['journal_context'] == []
    assert payload['journal_access']['available_count'] == 5
