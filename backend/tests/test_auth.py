import json
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from backend.conversation.user_profile import UserProfile
from backend.schemas.context import UserContext
from backend.web.accounts import AccountStore
from backend.web.api import COOKIE, ChatRuntime, create_app

PASSWORD = 'test-only-long-password'


@pytest.fixture
def app(tmp_path, monkeypatch):
    profiles = [UserProfile(profile_id='alice', preferred_name='Alice'), UserProfile(profile_id='bob', preferred_name='Bob')]
    entries = [dict(entry_id='same-id', date='2026-11-20', title=f'{name} private entry',
                    content=f'{name} private journal content', profile_id=name, synthetic=False)
               for name in ['alice', 'bob']]
    # Unassigned entries must never be attributed to whoever signs in.
    monkeypatch.setattr('backend.web.api.load_journal', lambda: entries + [{**entries[0], 'profile_id': None}])
    backend = SimpleNamespace(extract=Mock(return_value=UserContext(needs_clarification=True)),
                              generate=Mock(return_value='A response.'), search=Mock())
    runtime = ChatRuntime(backend, profiles[0], entries, profiles=profiles)
    accounts = AccountStore(tmp_path / 'accounts.sqlite3')
    accounts.create('alice', PASSWORD, 'alice')
    accounts.create('bob', PASSWORD, 'bob')
    return create_app(lambda: runtime, account_store=accounts), runtime, accounts


def login(client, username):
    response = client.post('/api/login', json={'username': username, 'password': PASSWORD})
    assert response.status_code == 200


@pytest.mark.parametrize('method,path', [
    ('get', '/api/session'), ('post', '/api/session'),
    ('get', '/api/journal'), ('post', '/api/journal/load'),
    ('post', '/api/journal/mock'), ('post', '/api/journal/clear'),
    ('post', '/api/chat'),
])
def test_protected_endpoints_require_login(app, method, path):
    application, runtime, _ = app
    with TestClient(application) as client:
        response = getattr(client, method)(path, **({'json': {'message': 'Hello'}} if path == '/api/chat' else {}))
        assert response.status_code == 401
        runtime.backend.extract.assert_not_called()


def test_accounts_are_hashed_and_wrong_password_fails(app):
    application, _, accounts = app
    assert PASSWORD.encode() not in accounts.path.read_bytes()
    with TestClient(application) as client:
        for username in ['alice', 'not-a-user']:
            response = client.post('/api/login', json={'username': username, 'password': 'wrong'})
            assert response.status_code == 401
            assert response.json()['detail'] == 'Invalid username or password'
        assert client.get('/api/journal').status_code == 401


def test_account_switch_logout_and_profile_isolation(app):
    application, runtime, _ = app
    with TestClient(application) as client:
        login(client, 'alice')
        alice_cookie = client.cookies.get(COOKIE)
        assert 'bob private' not in client.get('/api/journal').text
        client.post('/api/journal/load')
        assert client.get('/api/journal').json()['count'] == 1
        assert client.post('/api/chat', json={'message': 'Read my journal'}).status_code == 200
        payload = json.loads(runtime.backend.generate.call_args.args[0][-1]['content'])
        assert payload['user_profile']['profile_id'] == 'alice'
        assert {entry['profile_id'] for entry in payload['journal_context']} == {'alice'}
        login(client, 'bob')
        assert alice_cookie not in runtime.sessions
        assert client.get('/api/session').json()['messages'] == []
        assert 'alice private' not in client.get('/api/journal').text
        assert client.post('/api/chat', json={'message': 'Read my journal'}).status_code == 200
        bob_messages = runtime.backend.generate.call_args.args[0]
        bob_payload = json.loads(bob_messages[-1]['content'])
        assert bob_payload['user_profile']['profile_id'] == 'bob'
        assert {entry['profile_id'] for entry in bob_payload['journal_context']} == {'bob'}
        assert not any(message['role'] == 'assistant' for message in bob_messages)
        assert client.post('/api/chat', json={'message': 'Read journal', 'profile_id': 'alice'}).status_code == 422
        token = client.cookies.get(COOKIE)
        assert client.post('/api/logout').status_code == 200
        assert token not in runtime.sessions
        assert client.get('/api/journal').status_code == 401
        client.cookies.set(COOKIE, token, domain='testserver.local', path='/')
        assert client.get('/api/journal').status_code == 401


def test_session_expiry_and_login_rate_limit(app):
    application, runtime, _ = app
    with TestClient(application) as client:
        login(client, 'alice')
        token = client.cookies.get(COOKIE)
        state, touched = runtime.sessions[token]
        runtime.sessions[token] = (state, touched - 1801)
        assert client.get('/api/journal').status_code == 401
        for _ in range(5):
            assert client.post('/api/login', json={'username': 'alice', 'password': 'wrong'}).status_code == 401
        assert client.post('/api/login', json={'username': 'alice', 'password': PASSWORD}).status_code == 429
