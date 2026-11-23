
from typing import Literal
from pydantic import BaseModel, Field

Intervention = Literal[
    "socratic_questioning",
    "thought_record",
    "behavioral_activation",
    "problem_solving",
    "emotional_validation"
]

class Candidate(BaseModel):
    intervention: Intervention
    appropriateness: float = Field(ge=0, le=1)
    empathy: float = Field(ge=0, le=1)
    readiness: float = Field(ge=0, le=1)
    rationale: str

class InterventionPlan(BaseModel):
    candidates: list[Candidate]
    selected_intervention: Intervention
    selection_rationale: str
    suggested_response: str

def candidate_score(candidate: Candidate) -> float:
    return (
        0.4 * candidate.appropriateness
        + 0.3 * candidate.empathy
        + 0.3 * candidate.readiness
    )
