# Reflect

Reflect is a local AI wellness assistant for journal-aware conversations and
CBT-informed reflection. It combines a four-agent LangGraph workflow, a shared
Qwen model, and retrieved PubMed research. The browser interface includes login
and profile-scoped journal access. This is a non-clinical educational prototype.

## Local development setup

Prerequisites: Python 3.11+, Docker with Docker Compose, and internet access for
package installation, model downloads, and PubMed ingestion. Local model inference
requires sufficient memory and disk space; the first startup can take time.

Run all commands from the repository root.

### 1. Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell.

### 2. Start the research database

```bash
docker compose up -d db
```

Wait for PostgreSQL to be ready, then enable pgvector before running ingestion:

```bash
docker compose exec db pg_isready -U research_user -d research_db
docker compose exec db psql -U research_user -d research_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
python -m backend.ingestion.ingest_pubmed
```

The backend defaults to the local database credentials in `docker-compose.yml`.
For a different database, export `DATABASE_URL` in the shell before starting the
app. Research retrieval searches previously ingested articles, not live PubMed.

### 3. Configure a profile and login

Edit `backend/data/user_profile.json` and `backend/data/user_journal.json`.
Both files contain JSON arrays; each journal entry must have the owner's
`profile_id`. Create an account linked to an existing profile, for example:

```bash
python -m backend.web.accounts --username jia --profile-id 1
```

Choose a password at the hidden prompt. There are no default credentials.
Accounts are stored locally in `backend/data/accounts.sqlite3`.

## Run locally

With the virtual environment active and database running:

```bash
python -m backend.main --web
```

After **Application startup complete**, open **http://127.0.0.1:8001** and sign in.
The Python server also serves `frontend/chat/`; no separate frontend server or
Node installation is needed. Matching journal entries load automatically.
Restart the app after editing profile or journal files.

For a different port or terminal-only chat:

```bash
python -m backend.main --web --port 8002
python -m backend.main
```

Logs are written to `backend/logs/model.log`. Conversations remain in memory and
are lost when the session expires or the server restarts. Stop the app with
`Ctrl+C`; stop PostgreSQL with `docker compose stop db`.

## Tests

```bash
python -m pytest backend/tests -q
```

The automated tests use model/database fixtures rather than downloading Qwen or
requiring a live PostgreSQL connection.
