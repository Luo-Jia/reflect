from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class ThoughtNode:
    path: tuple[str, ...]
    score: float

def beam_search(
    initial_candidates: list[str],
    expand: Callable[[tuple[str, ...]], list[str]],
    evaluate: Callable[[tuple[str, ...]], float],
    beam_width: int = 3,
    max_depth: int = 2,
) -> ThoughtNode:
    if beam_width < 1 or max_depth < 1:
        raise ValueError("Invalid search configuration")

    beam = [ThoughtNode(path=(), score=0.0)]

    for depth in range(max_depth):
        expanded = []

        for node in beam:
            actions = (
                initial_candidates
                if depth == 0
                else expand(node.path)
            )

            for action in actions:
                path = node.path + (action,)
                score = evaluate(path)

                if not 0 <= score <= 1:
                    raise ValueError("Score must be between 0 and 1")

                expanded.append(
                    ThoughtNode(path=path, score=score)
                )

        if not expanded:
            break

        expanded.sort(
            key=lambda n: (-n.score, n.path)
        )
        beam = expanded[:beam_width]

    if not beam:
        raise ValueError("No valid intervention path")

    return beam[0]
