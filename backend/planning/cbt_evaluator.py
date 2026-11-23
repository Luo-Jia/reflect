STEP_SCORES = {
    "explore_automatic_thought": 0.90,
    "ask_for_evidence": 0.85,
    "identify_trigger": 0.88,
    "record_emotions": 0.86,
    "identify_small_activity": 0.90,
    "plan_first_step": 0.84,
    "define_problem": 0.86,
    "brainstorm_options": 0.82,
    "reflect_emotion": 0.95,
    "ask_permission": 0.96
}

def build_evaluator(candidate_scores: dict[str, float]):
    def evaluate(path: tuple[str, ...]) -> float:
        intervention = path[0]
        base = candidate_scores[intervention]

        if len(path) == 1:
            return base

        step_score = STEP_SCORES[path[1]]

        return 0.7 * base + 0.3 * step_score

    return evaluate
