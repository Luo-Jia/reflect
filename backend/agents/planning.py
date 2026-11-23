"""CBT Planning Agent: bounded ToT expansion over approved intervention paths."""
from backend.agents.contracts import CandidatePath
from backend.planning.cbt_actions import APPROVED_STEPS, expand_cbt_path
from backend.planning.constraints import allowed_step


def valid_path(path, context, accepted_exercise=None):
    if not path or len(path) > 2 or path[0] not in APPROVED_STEPS:
        return False
    if accepted_exercise and path[0] != accepted_exercise:
        return False
    steps = expand_cbt_path((path[0],))
    if len(path) == 1:
        return any(allowed_step(step, context) for step in steps)
    return path[1] in steps and allowed_step(path[1], context)


class PlanningAgent:
    def expand(self, context, beam, depth, examples, accepted_exercise=None):
        scores = {name: 0.5 for name in APPROVED_STEPS}
        for name in scores:
            values = [candidate.overall_score for example in examples
                      for candidate in example.tot.candidate_interventions
                      if candidate.intervention_id == name]
            if values:
                scores[name] = sum(values) / len(values)
        if depth == 0:
            paths = [(name,) for name in scores]
        else:
            # The critic's scores determine which parents are expanded next.
            paths = [parent.path + (step,) for parent in beam
                     for step in expand_cbt_path(parent.path)]
        return [CandidatePath(path=path, prior=scores[path[0]]) for path in paths
                if valid_path(path, context, accepted_exercise)]
