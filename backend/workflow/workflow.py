from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from backend.schemas.schemas import CBTExample
from backend.planning.beam_search import beam_search
from backend.planning.cbt_actions import expand_cbt_path
from backend.planning.cbt_evaluator import build_evaluator
from backend.workflow.qwen_node import qwen_context_node
from backend.workflow.safety import assess_safety, safety_response_for

class AgentState(TypedDict, total=False):
    example: CBTExample
    user_message: str
    history: list[dict[str, str]]
    safety_route: str
    selected_intervention: str | None
    selected_path: list[str]
    context: dict | None
    context_status: str
    response: str
    decision_source: str

def safety_gate(state: AgentState):
    example = state["example"]
    route = assess_safety(state.get("user_message", example.user_message), state.get("history", []))
    if route != "standard_cbt":
        return {"safety_route": route}
    return {"safety_route": example.safety.route}

def route_after_safety(state: AgentState):
    if state["safety_route"] in {"emergency", "safety_check"}:
        return "emergency_response"
    if state["safety_route"] == "bypass_cbt":
        return "safety_response"
    return "extract_context"


def emergency_response(state: AgentState):
    return {
        "selected_intervention": None,
        "selected_path": [],
        "response": safety_response_for(state["safety_route"]),
        "decision_source": state["safety_route"],
    }

def safety_response(state: AgentState):
    return {
        "selected_intervention": None,
        "response": state["example"].preferred_response,
        "decision_source": "synthetic_safety_reference"
    }

# def select_intervention(state: AgentState):
#     example = state["example"]
#     candidates = example.tot.candidate_interventions

#     if not candidates:
#         return {
#             "selected_intervention": None,
#             "response": "Would you like to tell me more?",
#             "decision_source": "fallback"
#         }

#     selected = max(
#         candidates,
#         key=lambda c: c.overall_score
#     )

#     return {
#         "selected_intervention": selected.intervention_id,
#         "response": example.preferred_response,
#         "decision_source": "synthetic_candidate_scores"
#     }

def select_intervention(state: AgentState):
    example = state["example"]

    if example.safety.route != "standard_cbt":
        raise ValueError("CBT planner called for safety bypass")

    candidates = example.tot.candidate_interventions

    candidate_scores = {
        candidate.intervention_id: candidate.overall_score
        for candidate in candidates
    }

    result = beam_search(
        initial_candidates=list(candidate_scores),
        expand=expand_cbt_path,
        evaluate=build_evaluator(candidate_scores),
        beam_width=3,
        max_depth=2
    )

    return {
        "selected_intervention": result.path[0],
        "selected_path": list(result.path),
        "response": example.preferred_response,
        "decision_source": "beam_search"
    }

def clarification_response(state: AgentState) -> dict:
    """
    Fallback when Qwen context extraction fails
    or clarification is required.
    """

    context_status = state.get("context_status", "unknown")
    context = state.get("context") or {}

    if context_status != "success":
        response = (
            "I'm here to listen. Could you tell me a little "
            "more about what's on your mind?"
        )

    elif context.get("needs_clarification", False):
        response = (
            "Would you prefer to talk about how you're feeling, "
            "explore your thoughts, or work through a small exercise?"
        )

    else:
        response = (
            "What would feel most helpful to you right now?"
        )

    return {
        "response": response,
        "selected_intervention": None,
        "selected_path": [],
        "decision_source": "clarification",
    }


builder = StateGraph(AgentState)
builder.add_node("safety_gate", safety_gate)
builder.add_node("safety_response", safety_response)
builder.add_node("emergency_response", emergency_response)
builder.add_node("extract_context", qwen_context_node)
builder.add_node("select_intervention", select_intervention)
builder.add_node("clarification_response", clarification_response)

builder.add_edge(START, "safety_gate")
builder.add_conditional_edges(
    "safety_gate",
    route_after_safety,
    {
        "safety_response": "safety_response",
        "emergency_response": "emergency_response",
        "extract_context": "extract_context"
    }
)
def route_after_extraction(state: AgentState) -> str:
    if state.get("context_status") != "success":
        return "clarification_response"

    context = state.get("context") or {}

    if context.get("needs_clarification", False):
        return "clarification_response"

    return "select_intervention"

builder.add_conditional_edges(
    "extract_context",
    route_after_extraction,
    {
        "select_intervention": "select_intervention",
        "clarification_response": "clarification_response"
    }
)
builder.add_edge("safety_response", END)
builder.add_edge("emergency_response", END)
builder.add_edge("select_intervention", END)
builder.add_edge("clarification_response", END)

graph = builder.compile()
