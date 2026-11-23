APPROVED_STEPS = {
    "socratic_questioning": [
        "explore_automatic_thought",
        "ask_for_evidence"
    ],
    "thought_record": [
        "identify_trigger",
        "record_emotions"
    ],
    "behavioral_activation": [
        "identify_small_activity",
        "plan_first_step"
    ],
    "problem_solving": [
        "define_problem",
        "brainstorm_options"
    ],
    "emotional_validation": [
        "reflect_emotion",
        "ask_permission"
    ]
}

def expand_cbt_path(path: tuple[str, ...]) -> list[str]:
    if len(path) != 1:
        return []
    return APPROVED_STEPS.get(path[0], [])
