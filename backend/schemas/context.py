from typing import Literal
from pydantic import BaseModel, Field

Emotion = Literal[
    "anxiety", "sadness", "shame",
    "frustration", "anger", "neutral"
]

Readiness = Literal["low", "medium", "high", "unknown"]

Preference = Literal[
    "validation",
    "reflection",
    "action",
    "unspecified"
]

class UserContext(BaseModel):
    emotions: list[Emotion] = Field(default_factory=list)
    intensity: float | None = Field(default=None, ge=0, le=1)
    readiness: Readiness = "unknown"
    preference: Preference = "unspecified"
    automatic_thought: str | None = None
    thought_patterns: list[str] = Field(default_factory=list, max_length=8)
    lifestyle_context: list[str] = Field(default_factory=list, max_length=8)
    consent_to_exercise: bool = False
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0, ge=0, le=1)
    needs_clarification: bool = False
