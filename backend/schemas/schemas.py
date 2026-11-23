from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class Context(StrictModel):
    session_context: str
    exercise_readiness: str
    user_consent_to_store: bool

class Safety(StrictModel):
    route: Literal["standard_cbt", "bypass_cbt"]
    category: str
    human_support_indicated: bool

class Evaluation(StrictModel):
    cbt_fit: float = Field(ge=0, le=1)
    emotional_attunement: float = Field(ge=0, le=1)
    readiness_fit: float = Field(ge=0, le=1)
    clarity: float = Field(ge=0, le=1)

class Candidate(StrictModel):
    intervention_id: str
    evaluation: Evaluation
    overall_score: float = Field(ge=0, le=1)
    rationale: str

class ToT(StrictModel):
    planning_performed: bool
    candidate_interventions: list[Candidate]
    selected_intervention: str | None
    selection_rationale: str

class Metadata(StrictModel):
    split: Literal["train", "validation", "test"]
    scores_are: str
    review_status: str
    locale: str

class CBTExample(StrictModel):
    case_id: str
    synthetic: Literal[True]
    scenario_category: str
    user_message: str = Field(min_length=1)
    context: Context
    hypothesized_emotions: list[str]
    hypothesized_automatic_thought: str | None
    possible_cognitive_patterns: list[str]
    safety: Safety
    tot: ToT
    preferred_response: str = Field(min_length=1)
    next_step: str
    metadata: Metadata

    @model_validator(mode="after")
    def check_decision(self):
        candidates = self.tot.candidate_interventions
        ids = [c.intervention_id for c in candidates]

        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate intervention candidates")

        if self.safety.route == "bypass_cbt":
            if (
                self.tot.planning_performed
                or candidates
                or self.tot.selected_intervention is not None
            ):
                raise ValueError("Safety bypass must skip CBT planning")
        else:
            if not self.tot.planning_performed or not candidates:
                raise ValueError("Standard CBT requires candidates")

            if self.tot.selected_intervention not in ids:
                raise ValueError("Selected intervention not in candidates")

        return self
