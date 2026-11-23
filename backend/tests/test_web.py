from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from backend.conversation.user_profile import UserProfile
from backend.schemas.context import UserContext
from backend.web.api import COOKIE, ChatRuntime, create_app
from backend.web.accounts import AccountStore

PASSWORD = 'test-password-long-enough'


@pytest.fixture
def web(monkeypatch, tmp_path):
    from backend.ingestion.ingest_journal import load_journal
    entries = [{**entry, 'profile_id': 'local-user'} for entry in load_journal()]
    monkeypatch.setattr('backend.web.api.load_journal', lambda: entries)
    backend = SimpleNamespace(
        extract=Mock(return_value=UserContext(needs_clarification=True)),
        generate=Mock(return_value="Let's talk about that."),
        search=Mock(return_value=[]),
    )
    runtime = ChatRuntime(backend, UserProfile(preferred_name="Test"))
    factory = Mock(return_value=runtime)
    accounts = AccountStore(tmp_path / 'accounts.sqlite3')
    accounts.create('tester', PASSWORD, 'local-user')
    app = create_app(factory, account_store=accounts)
    with TestClient(app) as client:
        assert client.post('/api/login', json={'username': 'tester', 'password': PASSWORD}).status_code == 200
        yield client, runtime, factory


def test_frontend_assets_and_initialization(web):
    client, _, factory = web
    assert client.get('/').status_code == 200
    assert 'What’s on your' in client.get('/chat').text
    assert client.get('/assets/chat.js').status_code == 200
    assert client.get('/assets/style.css').status_code == 200
    assert client.get('/assets/../../backend/.env').status_code == 404
    assert client.get('/api/session').json() == {'messages': [], 'ready': True, 'user': {'profile_id': 'local-user', 'preferred_name': 'Test'}}
    assert 'HttpOnly' in client.get('/api/session').headers['set-cookie']
    factory.assert_called_once()


def test_chat_history_and_new_conversation(web):
    client, runtime, _ = web
    first = client.post('/api/chat', json={'message': 'Hello'})
    assert first.status_code == 200
    assert first.json()['answer'] == "Let's talk about that."
    token = client.cookies.get(COOKIE)
    assert len(client.get('/api/session').json()['messages']) == 2
    assert client.post('/api/chat', json={'message': 'Another question'}).status_code == 200
    assert len(runtime.backend.extract.call_args.kwargs['history']) == 2
    assert client.post('/api/session').status_code == 200
    assert client.cookies.get(COOKIE) != token
    assert token not in runtime.sessions
    assert client.get('/api/session').json()['messages'] == []


def test_separate_browser_sessions(web):
    client, _, _ = web
    client.post('/api/chat', json={'message': 'Hello'})
    token = client.cookies.get(COOKIE)
    client.cookies.clear()
    assert client.get('/api/session').status_code == 401
    client.post('/api/login', json={'username': 'tester', 'password': PASSWORD})
    assert client.get('/api/session').json()['messages'] == []
    assert client.cookies.get(COOKIE) != token


@pytest.mark.parametrize('message', ['', '   ', 'a' * 4001])
def test_reject_invalid_input(web, message):
    client, runtime, _ = web
    assert client.post('/api/chat', json={'message': message}).status_code == 422
    runtime.backend.extract.assert_not_called()


def test_emergency_bypasses_llm(web):
    client, runtime, _ = web
    result = client.post('/api/chat', json={'message': 'I am in immediate danger.'})
    assert result.status_code == 200
    assert '911' in result.json()['answer']
    runtime.backend.extract.assert_not_called()
    runtime.backend.generate.assert_not_called()
    runtime.backend.search.assert_not_called()


def test_failure_preserves_session_and_hides_exception(web):
    client, runtime, _ = web
    client.get('/api/session')
    runtime.backend.generate.side_effect = RuntimeError('private model detail')
    result = client.post('/api/chat', json={'message': 'Hello'})
    assert result.status_code == 503
    assert 'private model detail' not in result.text
    assert client.get('/api/session').json()['messages'] == []


def test_reject_cross_origin_and_unknown_host(web):
    client, _, _ = web
    assert client.post('/api/session', headers={'Origin': 'https://example.com'}).status_code == 403
    assert client.get('/', headers={'Host': 'example.com'}).status_code == 400


def test_mock_journal_session_scope_reset_and_chat_context(web):
    import json
    client, runtime, _ = web
    assert client.get('/api/journal').json()['count'] == 0
    assert client.post('/api/journal/mock').json()['count'] == 5
    assert client.post('/api/journal/mock').json()['count'] == 5
    token = client.cookies.get(COOKIE)
    result = client.post('/api/chat', json={'message': 'What does the journal say about sleep?'})
    assert result.status_code == 200
    payload = json.loads(runtime.backend.generate.call_args.args[0][-1]['content'])
    assert payload['journal_context'][0]['entry_id'] == 'demo-004'
    client.cookies.clear()
    assert client.get('/api/journal').status_code == 401
    client.post('/api/login', json={'username': 'tester', 'password': PASSWORD})
    assert client.get('/api/journal').json()['count'] == 0
    client.cookies.set(COOKIE, token, domain='testserver.local', path='/')
    assert client.get('/api/journal').json()['count'] == 5
    assert client.post('/api/journal/clear').json()['count'] == 0
    client.post('/api/journal/mock')
    client.post('/api/session')
    assert client.get('/api/journal').json()['count'] == 0


def test_preloaded_journal_is_available_in_new_conversations(tmp_path):
    from backend.ingestion.ingest_journal import load_journal
    import json
    entries = [{**entry, 'profile_id': 'local-user'} for entry in load_journal()]
    backend = SimpleNamespace(extract=Mock(return_value=UserContext(needs_clarification=True)),
                              generate=Mock(return_value='Here is your demo journal.'), search=Mock())
    runtime = ChatRuntime(backend, UserProfile(), entries)
    accounts = AccountStore(tmp_path / 'accounts.sqlite3')
    accounts.create('tester', PASSWORD, 'local-user')
    with TestClient(create_app(lambda: runtime, account_store=accounts)) as client:
        client.post('/api/login', json={'username': 'tester', 'password': PASSWORD})
        assert client.get('/api/journal').json()['count'] == 5
        assert client.post('/api/chat', json={'message': 'List my journals'}).status_code == 200
        payload = json.loads(backend.generate.call_args.args[0][-1]['content'])
        assert payload['journal_access']['available_count'] == 5
        client.post('/api/journal/clear')
        assert client.get('/api/journal').json()['count'] == 0
        client.post('/api/session')
        assert client.get('/api/journal').json()['count'] == 5
