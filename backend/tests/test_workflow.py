
# tests/test_workflow.py
from backend.database.loader import load_dataset
from backend.workflow.workflow import graph
from pathlib import Path
from types import SimpleNamespace
import sys
from backend.schemas.context import UserContext

def test_dataset_and_workflow(monkeypatch):
    # Test graph routing without downloading or invoking an LLM.
    monkeypatch.setitem(sys.modules, "backend.model.qwen_extractor", SimpleNamespace(
        extract_context=lambda message: UserContext(
            readiness="medium", intensity=0.5, needs_clarification=False
        )
    ))
    examples = load_dataset(
        Path(__file__).resolve().parents[1] / "data/cbt_tot_synthetic_200.json"
    )

    assert len(examples) == 200

    for example in examples:
        result = graph.invoke({"example": example, "user_message": example.user_message})

        assert result["response"]

        if example.safety.route == "bypass_cbt":
            assert result["selected_intervention"] is None
        else:
            valid_ids = {
                c.intervention_id
                for c in example.tot.candidate_interventions
            }
            assert result["selected_intervention"] in valid_ids
