"""Validated diary entries, kept separate from research evidence."""
from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class JournalEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=100)
    profile_id: str | None = Field(default=None, min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    date: date
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=4000)
    synthetic: bool = False
