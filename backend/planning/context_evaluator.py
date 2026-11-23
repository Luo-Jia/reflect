from backend.schemas.context import UserContext

STEP_TYPES = {
    "reflect_emotion": "validation",
    "ask_permission": "validation",
    "explore_automatic_thought": "reflection",
    "ask_for_evidence": "reflection",
    "identify_trigger": "reflection",
    "record_emotions": "reflection",
    "identify_small_activity": "action",
    "plan_first_step": "action",
    "define_problem": "action",
    "brainstorm_options": "action",
}

READINESS_FIT = {
    "low": {
        "validation": 1.0,
        "reflection": 0.3,
        "action": 0.2,
    },
    "medium": {
        "validation": 0.9,
        "reflection": 0.7,
        "action": 0.65,
    },
    "high": {
        "validation": 0.7,
        "reflection": 1.0,
        "action": 1.0,
    },
}

def evaluate_step(
    step: str,
    context: UserContext,
    formulation_fit: float = 0.7,
) -> dict:
    kind = STEP_TYPES[step]

    # Higher distress favors supportive validation.
    if kind == "validation":
        emotional_fit = 1.0
    elif kind == "reflection":
        emotional_fit = max(
            0.25, 1.0 - context.intensity * 0.65
        )
    else:
        emotional_fit = max(
            0.2, 1.0 - context.intensity * 0.5
        )

    readiness_fit = READINESS_FIT[
        context.readiness
    ][kind]

    if context.preference == "unspecified":
        preference_fit = 0.65
    elif context.preference == kind:
        preference_fit = 1.0
    else:
        preference_fit = 0.3

    score = (
        0.35 * emotional_fit
        + 0.30 * readiness_fit
        + 0.25 * preference_fit
        + 0.10 * formulation_fit
    )

    return {
        "score": round(score, 4),
        "emotional_fit": emotional_fit,
        "readiness_fit": readiness_fit,
        "preference_fit": preference_fit,
        "formulation_fit": formulation_fit,
    }
