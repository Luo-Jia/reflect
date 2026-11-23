"""Validate and import diary JSON into an in-memory chat session."""
import argparse
import json
from pathlib import Path

from pydantic import TypeAdapter

from backend.schemas.journal import JournalEntry

USER_JOURNAL = Path(__file__).resolve().parents[1] / "data" / "user_journal.json"


def load_journal(path=USER_JOURNAL):
    path = Path(path)
    if path.stat().st_size > 500_000:
        raise ValueError("Journal file must be smaller than 500 KB")
    return validate_entries(json.loads(path.read_text(encoding="utf-8")))


def load_profile_journal(profile_id, path=USER_JOURNAL):
    """Load only explicitly owned entries; a missing optional file is empty."""
    if not Path(path).exists():
        return []
    return [entry for entry in load_journal(path) if entry['profile_id'] == profile_id]


def validate_entries(data):
    if not isinstance(data, list) or not 1 <= len(data) <= 100:
        raise ValueError("Provide between 1 and 100 journal entries")
    entries = TypeAdapter(list[JournalEntry]).validate_python(data)
    if len({entry.entry_id for entry in entries}) != len(entries):
        raise ValueError("Journal entry IDs must be unique within an import")
    return [entry.model_dump(mode="json") for entry in entries]


def ingest_journal(session, entries):
    """Validate the whole batch before atomically upserting entries by ID."""
    entries = validate_entries(entries)
    if any(entry['profile_id'] not in (None, session.profile_id) for entry in entries):
        raise ValueError("Journal entry belongs to a different profile")
    entries = [{**entry, 'profile_id': session.profile_id} for entry in entries]
    if any(entry.get('profile_id') != session.profile_id for entry in session.journal_entries):
        raise ValueError("Existing journal entries belong to a different profile")
    merged = {entry['entry_id']: entry for entry in session.journal_entries}
    merged.update({entry['entry_id']: entry for entry in entries})
    if len(merged) > 100:
        raise ValueError("A session can hold at most 100 journal entries")
    session.journal_entries = sorted(merged.values(), key=lambda entry: (entry['date'], entry['entry_id']))
    return len(session.journal_entries)


def main():
    parser = argparse.ArgumentParser(description="Validate a journal import without loading models or saving chat data.")
    parser.add_argument('--input', type=Path, default=USER_JOURNAL)
    parser.add_argument('--profile', type=Path, help='Profile JSON to associate with this import (defaults to the local profile)')
    args = parser.parse_args()
    from backend.conversation.session_state import SessionState
    from backend.conversation.user_profile import load_profile
    profile = load_profile(args.profile) if args.profile else load_profile()
    session = SessionState(profile_id=profile.profile_id)
    count = ingest_journal(session, load_journal(args.input))
    print(f"Validated and ingested {count} journal entries for profile {profile.profile_id} into a temporary session; nothing was saved.")


if __name__ == '__main__':
    main()
