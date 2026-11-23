from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

from backend.conversation.user_profile import UserProfile
from backend.schemas.context import UserContext
from backend.web.accounts import AccountStore
from backend.web.api import COOKIE, ChatRuntime, create_app

PASSWORD = 'test-password-only'


def setup(tmp_path):
    started, release = Event(), Event()

    def generate(messages):
        started.set()
        if not release.wait(10):
            raise RuntimeError('Test generation was not released')
        return 'A response.'

    backend = SimpleNamespace(extract=Mock(return_value=UserContext(needs_clarification=True)),
                              generate=generate, search=Mock())
    runtime = ChatRuntime(backend, UserProfile())
    accounts = AccountStore(tmp_path / 'accounts.sqlite3')
    accounts.create('tester', PASSWORD, 'local-user')
    return create_app(lambda: runtime, account_store=accounts), runtime, started, release


def login(client):
    assert client.post('/api/login', json={'username': 'tester', 'password': PASSWORD}).status_code == 200


def test_login_session_and_logout_do_not_wait_for_generation(tmp_path):
    app, runtime, started, release = setup(tmp_path)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        login(client)
        old_token = client.cookies.get(COOKIE)
        # Explicit headers freeze this request's session while other calls rotate it.
        pending = pool.submit(client.post, '/api/chat', json={'message': 'Hello'}, headers={'cookie': f'{COOKIE}={old_token}'})
        assert started.wait(2)
        try:
            assert pool.submit(client.get, '/api/session').result(timeout=2).status_code == 200
            assert pool.submit(client.post, '/api/login', json={'username': 'tester', 'password': PASSWORD}).result(timeout=2).status_code == 200
            assert old_token not in runtime.sessions
            assert pool.submit(client.post, '/api/logout').result(timeout=2).status_code == 200
        finally:
            release.set()
        assert pending.result(timeout=2).status_code == 401
        assert old_token not in runtime.sessions
        assert client.get('/api/session').status_code == 401


def test_busy_chat_returns_promptly_and_reset_discards_late_reply(tmp_path):
    app, runtime, started, release = setup(tmp_path)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        login(client)
        old_token = client.cookies.get(COOKIE)
        pending = pool.submit(client.post, '/api/chat', json={'message': 'Hello'}, headers={'cookie': f'{COOKIE}={old_token}'})
        assert started.wait(2)
        try:
            busy = pool.submit(client.post, '/api/chat', json={'message': 'Another message'}).result(timeout=2)
            assert busy.status_code == 429
            assert busy.headers['retry-after'] == '2'
            assert pool.submit(client.post, '/api/session').result(timeout=2).status_code == 200
            new_token = client.cookies.get(COOKIE)
            assert old_token != new_token
        finally:
            release.set()
        assert pending.result(timeout=2).status_code == 401
        assert client.cookies.get(COOKIE) == new_token
        assert client.get('/api/session').json()['messages'] == []


def test_fixed_safety_response_does_not_wait_for_busy_model(tmp_path):
    app, _, started, release = setup(tmp_path)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        login(client)
        pending = pool.submit(client.post, '/api/chat', json={'message': 'Hello'})
        assert started.wait(2)
        try:
            urgent = pool.submit(client.post, '/api/chat', json={'message': 'I am in immediate danger.'}).result(timeout=2)
            assert urgent.status_code == 200
            assert '911' in urgent.json()['answer']
        finally:
            release.set()
        assert pending.result(timeout=2).status_code == 409
        assert '911' in client.get('/api/session').json()['messages'][-1]['content']
