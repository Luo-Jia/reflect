"""Local password accounts. Provision from the terminal, never from profile JSON."""
import argparse
from getpass import getpass
import hashlib
import hmac
from pathlib import Path
import re
import secrets
import sqlite3

DEFAULT_ACCOUNTS = Path(__file__).resolve().parents[1] / "data" / "accounts.sqlite3"
ITERATIONS = 600_000


class AccountStore:
    def __init__(self, path=DEFAULT_ACCOUNTS):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(mode=0o600, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute('''CREATE TABLE IF NOT EXISTS accounts (
                username TEXT PRIMARY KEY, profile_id TEXT UNIQUE NOT NULL,
                salt BLOB NOT NULL, password_hash BLOB NOT NULL,
                iterations INTEGER NOT NULL
            )''')

    def create(self, username, password, profile_id):
        username = username.strip().lower()
        if not re.fullmatch(r'[a-z0-9_.-]{3,64}', username):
            raise ValueError('Username must be 3–64 letters, numbers, dots, dashes, or underscores')
        if not 1 <= len(password) <= 256:
            raise ValueError('Password must be 1–256 characters')
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, ITERATIONS)
        try:
            with sqlite3.connect(self.path) as db:
                db.execute('INSERT INTO accounts VALUES (?, ?, ?, ?, ?)',
                           (username, profile_id, salt, digest, ITERATIONS))
        except sqlite3.IntegrityError:
            raise ValueError('An account already exists for that username or profile') from None

    def verify(self, username, password):
        with sqlite3.connect(self.path) as db:
            row = db.execute('SELECT profile_id, salt, password_hash, iterations FROM accounts WHERE username = ?',
                             (username.strip().lower(),)).fetchone()
        # Perform the same expensive work for unknown accounts.
        profile_id, salt, expected, rounds = row or (None, b'\x00' * 16, b'\x00' * 32, ITERATIONS)
        actual = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, rounds)
        valid = hmac.compare_digest(actual, expected)
        return profile_id if valid and row else None


def main():
    parser = argparse.ArgumentParser(description='Create a local login for an existing profile.')
    parser.add_argument('--username', required=True)
    parser.add_argument('--profile-id', required=True)
    args = parser.parse_args()
    from backend.conversation.user_profile import load_profile
    load_profile(profile_id=args.profile_id)  # Reject nonexistent profiles.
    password = getpass('Password (1–256 characters): ')
    if password != getpass('Confirm password: '):
        parser.error('Passwords do not match')
    try:
        AccountStore().create(args.username, password, args.profile_id)
    except ValueError as exc:
        parser.error(str(exc))
    print('Account created. Sign in through the browser chatbot.')


if __name__ == '__main__':
    main()
