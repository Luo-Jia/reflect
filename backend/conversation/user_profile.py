"""Load manually maintained profile facts; chat never writes this file."""
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_PROFILE = Path(__file__).resolve().parents[1] / "data" / "user_profile.json"


class ConfirmedFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fact: str
    source: str = "user_confirmed"


class UserProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(default="local-user", min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    preferred_name: str = Field(default="", max_length=100)
    response_preferences: list[str] | dict[str, str] = Field(default_factory=list, max_length=10)
    goals: list[str] = Field(default_factory=list, max_length=10)
    confirmed_facts: list[str | ConfirmedFact] = Field(default_factory=list, max_length=20)


def load_profiles(path=DEFAULT_PROFILE):
    """Read a profile array; accept legacy single-object files as well."""
    path = Path(path)
    if not path.exists():
        return [UserProfile()]
    if path.stat().st_size > 20_000:
        raise ValueError("Profile must be smaller than 20 KB")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not data:
        raise ValueError("Profile file must contain a nonempty array of profiles")
    profiles = [UserProfile.model_validate(item) for item in data]
    if len({profile.profile_id for profile in profiles}) != len(profiles):
        raise ValueError("Profile IDs must be unique")
    return profiles


def load_profile(path=DEFAULT_PROFILE, profile_id=None):
    """Return the first profile by default, or select an explicit profile ID."""
    profiles = load_profiles(path)
    if profile_id is None:
        return profiles[0]
    for profile in profiles:
        if profile.profile_id == profile_id:
            return profile
    raise ValueError("Requested profile ID was not found")
