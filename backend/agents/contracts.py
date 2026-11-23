"""Validated state messages for the four-agent CBT workflow."""
from pydantic import BaseModel, ConfigDict, Field


class CandidatePath(BaseModel):
    model_config = ConfigDict(extra='forbid')
    path: tuple[str, ...] = Field(min_length=1, max_length=2)
    prior: float = Field(default=0.5, ge=0, le=1)


class PathEvaluation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    path: tuple[str, ...] = Field(min_length=1, max_length=2)
    emotional_fit: float = Field(ge=0, le=1)
    readiness_fit: float = Field(ge=0, le=1)
    preference_fit: float = Field(ge=0, le=1)
    cbt_fit: float = Field(ge=0, le=1)
    evidence_support: float = Field(ge=0, le=1)
    evidence_ids: list[int] = Field(default_factory=list, max_length=10)
    rationale: str = Field(default='', max_length=300)

    @property
    def score(self):
        # The application controls aggregation, not the model.
        return round(0.30 * self.emotional_fit + 0.25 * self.readiness_fit
                     + 0.20 * self.preference_fit + 0.15 * self.cbt_fit
                     + 0.10 * self.evidence_support, 4)

    def record(self):
        return {**self.model_dump(mode='json'), 'score': self.score}


class CriticReport(BaseModel):
    model_config = ConfigDict(extra='forbid')
    evaluations: list[PathEvaluation] = Field(min_length=1, max_length=10)


class EvidenceRecord(BaseModel):
    source: int = Field(ge=1)
    title: str | None = None
    abstract: str = Field(default='', max_length=2500)
    url: str | None = None
    doi: str | None = None
    pmid: str | None = None


class OutputCheck(BaseModel):
    valid: bool
    issues: list[str] = Field(default_factory=list)
    fallback_used: bool = False
