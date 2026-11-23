from backend.schemas.context import UserContext
from backend.planning.context_evaluator import STEP_TYPES

def allowed_step(step: str, context: UserContext) -> bool:
    kind = STEP_TYPES.get(step)

    if kind is None:
        return False

    # Without exercise consent, offer support
    # or ask permission rather than initiating CBT.
    if not context.consent_to_exercise:
        return step in {
            "reflect_emotion",
            "ask_permission",
        }

    return True
