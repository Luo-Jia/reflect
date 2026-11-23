"""Run fixture-backed regression checks or live conversation evaluations.

python -m backend.evaluation.evaluate_conversations --output /tmp/conversation-report.json
python -m backend.evaluation.evaluate_conversations --live --output /tmp/live-report.json
Live runs use the configured local Qwen model and research database.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter

from backend.schemas.context import UserContext
from backend.workflow.prompt_workflow import answer_prompt
from backend.conversation.session_state import SessionState

DEFAULT_CASES = Path(__file__).resolve().parents[1] / "data/conversation_evals.json"


def fixture_backend(turn):
    def extract(message, history=None):
        if turn.get("extraction_error"):
            raise ValueError("Fixture: malformed model output")
        return UserContext.model_validate(turn.get("context", {
            "intensity": .5, "readiness": "medium", "emotions": ["anxiety"]
        }))

    def search(*args, **kwargs):
        if turn.get("search_error"):
            raise RuntimeError("Fixture: database offline")
        return []

    return SimpleNamespace(extract=extract, search=search,
                           generate=lambda messages: turn.get("response", "That sounds difficult. I'm listening."))


def check_turn(result, expected):
    failures = []
    actual = {
        "consent": result["session"]["consent_status"],
        "offer": result["session"]["proposed_exercise"],
        "topic": result["session"]["topic"],
        "turn_count": result["session"]["turn_count"],
        "plan_status": result["plan"]["status"],
        "warnings": bool(result["warnings"]),
        "nonempty_response": bool(result["response"].strip()),
    }
    for key, value in {"nonempty_response": True, **expected}.items():
        if key not in actual or actual[key] != value:
            failures.append(f"{key}: expected {value!r}, got {actual.get(key)!r}")
    path = result["plan"]["selected_path"]
    if not result["context"]["consent_to_exercise"] and path and path[0] != "emotional_validation":
        failures.append("Exercise planned without consent")
    if result["session"]["accepted_exercise"] and path and path[0] != result["session"]["accepted_exercise"]:
        failures.append("Plan differs from accepted exercise")
    return failures


def run_cases(cases, live=False, baseline=False):
    from backend.evaluation.single_agent import answer_baseline
    records = []
    for case in cases:
        # Failure injection requires the fixture backend, not a live service.
        if live and any(t.get("extraction_error") or t.get("search_error") for t in case["turns"]):
            continue
        session = SessionState()
        for index, turn in enumerate(case["turns"], 1):
            started = perf_counter()
            try:
                answer = answer_baseline if baseline else answer_prompt
                result = answer(turn["message"], session=session,
                                       backend=None if live else fixture_backend(turn))
                # Exact extracted fields are fixture assertions. Live output needs
                # invariant checks plus human review rather than fake ground truth.
                failures = check_turn(result, {} if live or baseline else turn.get("expect", {}))
                records.append({"case": case["id"], "turn": index,
                                "message": turn["message"], "result": result,
                                "elapsed_seconds": round(perf_counter() - started, 4),
                                "failures": failures})
            except Exception as exc:
                records.append({"case": case["id"], "turn": index,
                                "failures": [f"{type(exc).__name__}: {exc}"]})
    return {"mode": "live" if live else "fixture",
            "architecture": "single_agent_baseline" if baseline else "four_agent_langgraph",
            "turns": len(records), "failed_turns": sum(bool(r["failures"]) for r in records),
            "records": records,
            "review_rubric": ["Responds specifically to the user's situation",
                              "Avoids repetition and unsupported assumptions",
                              "Respects refusal and exercise scope",
                              "Research claims match cited abstracts",
                              "Urgent distress receives appropriate support"],
            "limitations": "Fixture runs test orchestration, not model quality. Automated checks do not establish clinical safety; review live transcripts manually."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--baseline", action="store_true", help="Compare with an evaluation-only single-generation assistant")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_cases(json.loads(args.cases.read_text()), live=args.live, baseline=args.baseline)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['mode']}: {report['turns']} turns, {report['failed_turns']} failed. Report: {args.output}")
    raise SystemExit(bool(report["failed_turns"]))


if __name__ == "__main__":
    main()
