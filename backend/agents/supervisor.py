"""Four-agent LangGraph coordinator with independent safety and output controls."""
from functools import lru_cache
import logging
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END

from backend.agents.context import ContextAgent
from backend.agents.critic import CriticAgent
from backend.agents.planning import PlanningAgent, valid_path
from backend.agents.response import ResponseAgent
from backend.agents.contracts import CandidatePath, PathEvaluation
from backend.conversation.session_state import SessionState
from backend.conversation.user_profile import UserProfile
from backend.retrieval.dual_rag import retrieve_context, asks_about_research
from backend.schemas.context import UserContext
from backend.workflow.output_validator import validate_output, FALLBACK_RESPONSE
from backend.workflow.safety import assess_safety, safety_response_for

logger = logging.getLogger(__name__)
BEAM_WIDTH = 3
MAX_DEPTH = 2


class ChatState(TypedDict, total=False):
    message: str
    top_k: int
    backend: Any
    profile: UserProfile | None
    session: SessionState
    history: list[dict]
    safety_route: str
    context: UserContext
    eligible: bool
    examples: list
    articles: list[dict]
    evidence: list[dict]
    journal: list[dict]
    journal_access: dict
    research_access: dict
    candidates: list[CandidatePath]
    evaluations: list[PathEvaluation]
    evaluation_history: list[dict]
    beam: list[CandidatePath]
    depth: int
    plan: dict
    response: str
    offer: str | None
    validation: dict
    warnings: list[str]
    agent_trace: list[dict]


def traced(state, name, status, **metadata):
    return state['agent_trace'] + [{'agent': name, 'status': status, **metadata}]


def safety_gate(state):
    route = assess_safety(state['message'], state['history'])
    return {'safety_route': route, 'agent_trace': traced(state, 'safety_gate', route)}


def safety_response(state):
    context = UserContext(needs_clarification=True)
    state['session'].resolve_reply(state['message'], context)
    return {'context': context, 'response': safety_response_for(state['safety_route']),
            'plan': {'selected_path': [], 'status': state['safety_route']}, 'offer': None}


def context_agent(state):
    backend = state['backend']
    if backend is None:
        from backend.workflow.prompt_workflow import get_default_backend
        backend = get_default_backend()
    context, status, warnings = ContextAgent(backend).run(state['message'], state['history'])
    return {'backend': backend, 'context': context, 'warnings': state['warnings'] + warnings,
            'agent_trace': traced(state, 'context', status)}


def consent_gate(state):
    context = state['context'].model_copy(deep=True)
    state['session'].resolve_reply(state['message'], context)
    eligible = not context.needs_clarification and context.intensity is not None and context.readiness != 'unknown'
    return {'context': context, 'eligible': eligible,
            'agent_trace': traced(state, 'consent_gate', state['session'].consent_status)}


def dual_rag(state):
    # Research tool receives only a generic topic; it never sees the journal or profile.
    examples, articles, evidence, journal, access, warnings = retrieve_context(
        state['message'], state['session'], state['context'], state['backend'], state['top_k'], state['eligible']
    )
    return {'examples': examples, 'articles': articles, 'evidence': evidence,
            'journal': journal, 'journal_access': access, 'warnings': state['warnings'] + warnings,
            'research_access': {
                'requested': asks_about_research(state['message']),
                'available_count': len(evidence),
                'status': 'unavailable' if warnings else 'retrieved' if evidence else 'empty' if state['eligible'] or asks_about_research(state['message']) else 'skipped',
            },
            'agent_trace': traced(state, 'dual_rag', 'degraded' if warnings else 'completed',
                                  synthetic_count=len(examples), research_count=len(evidence), journal_count=len(journal))}


def planning_agent(state):
    candidates = PlanningAgent().expand(state['context'], state['beam'], state['depth'],
                                        state['examples'], state['session'].accepted_exercise)
    return {'candidates': candidates,
            'agent_trace': traced(state, 'planning', 'completed', depth=state['depth'] + 1, candidate_count=len(candidates))}


def critic_agent(state):
    evaluations, status, warnings = CriticAgent(state['backend']).evaluate(
        state['candidates'], state['context'], state['evidence']
    )
    return {'evaluations': evaluations, 'warnings': state['warnings'] + warnings,
            'evaluation_history': state['evaluation_history'] + [item.record() for item in evaluations],
            'agent_trace': traced(state, 'critic', status, depth=state['depth'] + 1)}


def beam_decision(state):
    # Independent authorization checks are reapplied after critic scoring.
    valid = [item for item in state['evaluations'] if valid_path(
        item.path, state['context'], state['session'].accepted_exercise
    )]
    valid.sort(key=lambda item: (-item.score, item.path))
    kept = valid[:BEAM_WIDTH]
    priors = {item.path: item.prior for item in state['candidates']}
    beam = [CandidatePath(path=item.path, prior=priors[item.path]) for item in kept]
    depth = state['depth'] + 1
    plan = {'selected_path': [], 'score': None, 'status': 'offer_support_or_ask_permission'}
    if kept and depth == MAX_DEPTH:
        plan = {'selected_path': list(kept[0].path), 'score': kept[0].score, 'status': 'planned'}
    plan.update(beam_width=BEAM_WIDTH, max_depth=MAX_DEPTH, candidate_evaluations=state['evaluation_history'])
    return {'beam': beam, 'depth': depth, 'plan': plan,
            'agent_trace': traced(state, 'beam_search', 'completed', depth=depth, retained=len(beam))}


def response_agent(state):
    # Planning agents cannot grant consent or expand their action/tool permissions.
    plan = state['plan']
    if plan['selected_path'] and not valid_path(plan['selected_path'], state['context'], state['session'].accepted_exercise):
        raise ValueError('Selected plan violates intervention constraints')
    payload = {
        'user_profile': state['profile'].model_dump() if state['profile'] else {},
        'user_message': state['message'], 'context': state['context'].model_dump(),
        'session': state['session'].summary(), 'plan': plan,
        'synthetic_examples': [{'message': ex.user_message, 'intervention': ex.tot.selected_intervention}
                               for ex in state['examples']],
        'research': state['evidence'], 'journal_context': state['journal'],
        'research_access': state['research_access'],
        'journal_access': state['journal_access'],
    }
    response = ResponseAgent(state['backend']).run(payload, state['history'])
    offer = None
    if state['eligible'] and state['session'].consent_status == 'unknown' and isinstance(response, str):
        offer = 'thought_record'
        response += '\n\nWould you like to try a brief thought-record exercise?'
    return {'response': response, 'offer': offer, 'agent_trace': traced(state, 'response', 'completed')}


def output_validator(state):
    check = validate_output(state['response'], state['context'], len(state['evidence']))
    updates = {'validation': check.model_dump(),
               'agent_trace': traced(state, 'output_validator', 'passed' if check.valid else 'fallback')}
    if not check.valid:
        updates.update(response=FALLBACK_RESPONSE, offer=None,
                       plan={'selected_path': [], 'status': 'output_validation_fallback'},
                       warnings=state['warnings'] + ['Response checks requested a safer fallback.'])
    return updates


def finalize(state):
    state['session'].record(state['message'], state['response'], offer=state['offer'])
    logger.info('Agent execution: %s', ', '.join(f"{item['agent']}={item['status']}" for item in state['agent_trace']))
    return {}


@lru_cache(maxsize=1)
def get_chat_graph():
    builder = StateGraph(ChatState)
    for name, node in [('safety_gate', safety_gate), ('safety_response', safety_response),
                       ('context', context_agent), ('consent_gate', consent_gate), ('dual_rag', dual_rag),
                       ('planning', planning_agent), ('critic', critic_agent), ('beam_search', beam_decision),
                       ('response', response_agent), ('output_validator', output_validator), ('finalize', finalize)]:
        builder.add_node(name, node)
    builder.add_edge(START, 'safety_gate')
    builder.add_conditional_edges('safety_gate', lambda state: 'context' if state['safety_route'] == 'standard_cbt' else 'safety_response')
    builder.add_edge('safety_response', 'output_validator')
    builder.add_edge('context', 'consent_gate')
    builder.add_edge('consent_gate', 'dual_rag')
    builder.add_conditional_edges('dual_rag', lambda state: 'planning' if state['eligible'] else 'response')
    builder.add_edge('planning', 'critic')
    builder.add_edge('critic', 'beam_search')
    builder.add_conditional_edges('beam_search', lambda state: 'planning' if state['beam'] and state['depth'] < MAX_DEPTH else 'response')
    builder.add_edge('response', 'output_validator')
    builder.add_edge('output_validator', 'finalize')
    builder.add_edge('finalize', END)
    # No checkpointer: sessions from different users never share persistent graph state.
    return builder.compile()


class ChatSupervisor:
    def __init__(self, backend=None):
        self.backend = backend

    def run(self, message, top_k=5, history=None, session=None, profile=None):
        if top_k < 1:
            raise ValueError('top_k must be at least 1')
        session = session if session is not None else SessionState(
            history=list(history or []), profile_id=profile.profile_id if profile else 'local-user'
        )
        if profile is not None and session.profile_id != profile.profile_id:
            raise ValueError('Session and user profile do not match')
        state = get_chat_graph().invoke({
            'message': message, 'top_k': top_k, 'backend': self.backend, 'profile': profile,
            'session': session, 'history': list(session.history), 'warnings': [], 'agent_trace': [],
            'beam': [], 'depth': 0, 'evaluation_history': [], 'evidence': [], 'articles': [],
            'examples': [], 'journal': [], 'journal_access': {}, 'eligible': False, 'offer': None,
            'plan': {'selected_path': [], 'status': 'clarification'},
        }, config={'recursion_limit': 24})
        return {'response': state['response'], 'articles': state['articles'], 'warnings': state['warnings'],
                'context': state['context'].model_dump(), 'plan': state['plan'], 'session': session.summary(),
                'decision_source': 'langgraph_four_agent' if state['safety_route'] == 'standard_cbt' else state['safety_route'],
                'agent_trace': state['agent_trace'], 'validation': state['validation']}
