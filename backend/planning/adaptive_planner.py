from backend.planning.beam_search import beam_search
from backend.planning.cbt_actions import expand_cbt_path
from backend.schemas.context import UserContext
from backend.planning.context_evaluator import evaluate_step
from backend.planning.constraints import allowed_step

def plan_intervention(
    context: UserContext,
    candidate_scores: dict[str, float],
):
    def expand(path: tuple[str, ...]) -> list[str]:
        return [
            step
            for step in expand_cbt_path(path)
            if allowed_step(step, context)
        ]

    def evaluate(path: tuple[str, ...]) -> float:
        intervention = path[0]
        base_score = candidate_scores[intervention]

        if len(path) == 1:
            return base_score

        step = path[1]
        step_result = evaluate_step(step, context)

        return round(
            0.4 * base_score
            + 0.6 * step_result["score"],
            4
        )

    # Keep only interventions with permitted next steps.
    eligible = [
        name for name in candidate_scores
        if expand_cbt_path((name,))
        and any(
            allowed_step(step, context)
            for step in expand_cbt_path((name,))
        )
    ]

    if not eligible:
        return {
            "selected_path": [],
            "score": None,
            "status": "offer_support_or_ask_permission"
        }

    result = beam_search(
        initial_candidates=eligible,
        expand=expand,
        evaluate=evaluate,
        beam_width=3,
        max_depth=2,
    )

    return {
        "selected_path": list(result.path),
        "score": result.score,
        "status": "planned"
    }
