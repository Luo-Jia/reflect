"""Same-origin local webchat. Run one worker to share one loaded model."""
from contextlib import asynccontextmanager
from copy import deepcopy
import logging
from pathlib import Path
from secrets import token_urlsafe
from threading import Lock
from time import monotonic

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend.conversation.session_state import SessionState
from backend.workflow.prompt_workflow import answer_prompt
from backend.ingestion.ingest_journal import ingest_journal, load_journal, load_profile_journal
from backend.conversation.user_profile import load_profiles
from backend.web.accounts import AccountStore
from backend.workflow.safety import assess_safety

FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "chat"
COOKIE = "backend_chat_session"
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    model_config = {"extra": "forbid"}
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value.strip()


class LoginRequest(BaseModel):
    model_config = {"extra": "forbid"}
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class ChatRuntime:
    def __init__(self, backend, profile, journal_entries=(), profiles=None):
        self.backend = backend
        self.profile = profile
        self.lock = Lock()
        self.generation_lock = Lock()
        self.sessions = {}
        self.journal_entries = list(journal_entries)
        self.profiles = {item.profile_id: item for item in (profiles or [profile])}
        self.login_attempts = {}

    def session(self, token, profile_id=None):
        now = monotonic()
        for key, (_, touched) in list(self.sessions.items()):
            if now - touched > 30 * 60:
                del self.sessions[key]
        if token not in self.sessions:
            if profile_id is None or profile_id not in self.profiles:
                raise HTTPException(401, "Sign in to continue")
            if len(self.sessions) >= 64:
                oldest = min(self.sessions, key=lambda key: self.sessions[key][1])
                del self.sessions[oldest]
            token = token_urlsafe(32)
            session = SessionState(profile_id=profile_id)
            entries = [entry for entry in self.journal_entries if entry.get('profile_id') == profile_id]
            if entries:
                ingest_journal(session, entries)
            self.sessions[token] = (session, now)
        state, _ = self.sessions[token]
        self.sessions[token] = (state, now)
        return token, state


def default_runtime():
    from backend.main import initialize_chat
    backend, profile = initialize_chat()
    profiles = load_profiles()
    entries = [entry for item in profiles for entry in load_profile_journal(item.profile_id)]
    return ChatRuntime(backend, profile, entries, profiles=profiles)


def create_app(runtime_factory=default_runtime, account_store=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = runtime_factory()
        app.state.accounts = account_store if account_store is not None else AccountStore()
        yield
        app.state.runtime.sessions.clear()

    app = FastAPI(title="Simple local chat", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def local_requests(request, call_next):
        origin = request.headers.get("origin")
        if request.method == "POST" and origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "Cross-origin requests are not allowed"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response

    def cookie(response, token):
        response.set_cookie(COOKIE, token, httponly=True, samesite="strict", max_age=1800)

    @app.post("/api/login")
    def login(body: LoginRequest, request: Request, response: Response):
        runtime = app.state.runtime
        with runtime.lock:
            now = monotonic()
            runtime.login_attempts = {key: value for key, value in runtime.login_attempts.items() if now - value[0] < 60}
            address = request.client.host if request.client else "unknown"
            started, failures = runtime.login_attempts.get(address, (now, 0))
            if failures >= 5:
                raise HTTPException(429, "Too many sign-in attempts. Try again in a minute.")
            # Reserve an attempt under the short state lock, but hash outside it.
            runtime.login_attempts[address] = (started, failures + 1)
        profile_id = app.state.accounts.verify(body.username, body.password)
        with runtime.lock:
            if profile_id is None or profile_id not in runtime.profiles:
                raise HTTPException(401, "Invalid username or password")
            runtime.login_attempts.pop(address, None)
            runtime.sessions.pop(request.cookies.get(COOKIE), None)
            token, _ = runtime.session(None, profile_id=profile_id)
            cookie(response, token)
            return {"authenticated": True}

    @app.post("/api/logout")
    def logout(request: Request, response: Response):
        runtime = app.state.runtime
        with runtime.lock:
            runtime.sessions.pop(request.cookies.get(COOKIE), None)
            response.delete_cookie(COOKIE)
            return {"authenticated": False}

    @app.get("/api/session")
    def current_session(request: Request, response: Response):
        runtime = app.state.runtime
        with runtime.lock:
            token, session = runtime.session(request.cookies.get(COOKIE))
            cookie(response, token)
            profile = runtime.profiles[session.profile_id]
            return {"messages": session.history, "ready": True,
                    "user": {"profile_id": profile.profile_id, "preferred_name": profile.preferred_name}}

    @app.post("/api/session")
    def new_session(request: Request, response: Response):
        runtime = app.state.runtime
        with runtime.lock:
            old_token, session = runtime.session(request.cookies.get(COOKIE))
            profile_id = session.profile_id
            runtime.sessions.pop(old_token, None)
            token, _ = runtime.session(None, profile_id=profile_id)
            cookie(response, token)
            return {"messages": []}

    @app.post("/api/chat")
    def chat(body: ChatRequest, request: Request, response: Response):
        runtime = app.state.runtime
        # Never hold the session/authentication lock during model inference.
        with runtime.lock:
            token, session = runtime.session(request.cookies.get(COOKIE))
            original = deepcopy(session)
            working = deepcopy(session)
        # Do not queue requests that occupy the HTTP thread pool while waiting.
        needs_model = assess_safety(body.message, working.history) == 'standard_cbt'
        acquired = needs_model and runtime.generation_lock.acquire(blocking=False)
        if needs_model and not acquired:
            raise HTTPException(429, "The assistant is responding to another message. Please try again shortly.",
                                headers={"Retry-After": "2"})
        try:
            try:
                result = answer_prompt(body.message, session=working, backend=runtime.backend, profile=runtime.profiles[session.profile_id])
            except Exception as exc:
                # Do not log prompts or exception values containing personal data.
                logger.error("Chat request failed (%s)", type(exc).__name__)
                raise HTTPException(503, "The assistant could not respond. Please try again.") from None
            with runtime.lock:
                current = runtime.sessions.get(token)
                if current is None or monotonic() - current[1] > 1800:
                    # Logout/reset/expiry must not be undone by a late reply.
                    runtime.sessions.pop(token, None)
                    raise HTTPException(401, "Your session ended. Please sign in again.")
                if current[0] is not session or current[0] != original:
                    raise HTTPException(409, "The conversation changed while responding. Please try again.")
                runtime.sessions[token] = (working, monotonic())
                cookie(response, token)
                return {
                    "answer": result["response"],
                    "sources": [
                        {"title": item.get("title"), "url": item.get("url")}
                        for item in result["articles"]
                    ],
                    "warnings": result["warnings"],
                    "agent_trace": result.get("agent_trace", []),
                }
        finally:
            if acquired:
                runtime.generation_lock.release()

    @app.get("/api/journal")
    def journal_status(request: Request, response: Response):
        runtime = app.state.runtime
        with runtime.lock:
            token, session = runtime.session(request.cookies.get(COOKIE))
            cookie(response, token)
            return {"count": len(session.journal_entries), "entries": session.journal_entries}

    @app.post("/api/journal/load")
    @app.post("/api/journal/mock", include_in_schema=False)
    def import_mock_journal(request: Request, response: Response):
        runtime = app.state.runtime
        with runtime.lock:
            token, session = runtime.session(request.cookies.get(COOKIE))
            try:
                entries = [entry for entry in load_journal() if entry['profile_id'] == session.profile_id]
                count = ingest_journal(session, entries) if entries else len(session.journal_entries)
            except ValueError:
                raise HTTPException(422, "Journal entries must match the active profile_id.") from None
            cookie(response, token)
            return {"count": count, "entries": session.journal_entries}

    @app.post("/api/journal/clear")
    def clear_journal(request: Request, response: Response):
        runtime = app.state.runtime
        with runtime.lock:
            token, session = runtime.session(request.cookies.get(COOKIE))
            session.journal_entries.clear()
            cookie(response, token)
            return {"count": 0, "entries": []}

    @app.get("/")
    @app.get("/chat")
    def index():
        return FileResponse(FRONTEND / "index.html")

    app.mount("/assets", StaticFiles(directory=FRONTEND), name="assets")
    return app
