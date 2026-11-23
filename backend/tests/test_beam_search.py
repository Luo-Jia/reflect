
# tests/test_beam_search.py

from backend.planning.beam_search import beam_search
from backend.planning.cbt_actions import expand_cbt_path
from backend.planning.cbt_evaluator import build_evaluator

def test_beam_search_returns_two_step_path():
    scores = {
        "socratic_questioning": 0.94,
        "thought_record": 0.82,
        "behavioral_activation": 0.58
    }

    result = beam_search(
        initial_candidates=list(scores),
        expand=expand_cbt_path,
        evaluate=build_evaluator(scores),
        beam_width=3,
        max_depth=2
    )

    assert len(result.path) == 2
    assert result.path[0] in scores
    assert 0 <= result.score <= 1

def test_highest_scoring_path():
    scores = {
        "socratic_questioning": 0.94,
        "thought_record": 0.82,
        "behavioral_activation": 0.58
    }

    result = beam_search(
        initial_candidates=list(scores),
        expand=expand_cbt_path,
        evaluate=build_evaluator(scores)
    )

    assert result.path == (
        "socratic_questioning",
        "explore_automatic_thought"
    )
