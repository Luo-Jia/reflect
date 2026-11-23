"""Evaluation-only single-generation baseline with the same access/safety controls."""
from backend.agents.response import ResponseAgent
from backend.conversation.session_state import SessionState
from backend.retrieval.dual_rag import retrieve_context, asks_about_research
from backend.schemas.context import UserContext
from backend.workflow.output_validator import validate_output, FALLBACK_RESPONSE
from backend.workflow.safety import assess_safety, safety_response_for


def answer_baseline(message, session=None, backend=None, profile=None, top_k=5):
    session = session if session is not None else SessionState(profile_id=profile.profile_id if profile else 'local-user')
    if profile is not None and session.profile_id != profile.profile_id:
        raise ValueError('Session and user profile do not match')
    context = UserContext(needs_clarification=True)
    history = list(session.history)
    route = assess_safety(message, history)
    session.resolve_reply(message, context)
    articles, warnings, evidence = [], [], []
    if route != 'standard_cbt':
        response = safety_response_for(route)
    else:
        if backend is None:
            from backend.workflow.prompt_workflow import get_default_backend
            backend = get_default_backend()
        # Retrieval uses a generic topic, not private journal text. No context
        # extraction, critic, or planning generation is performed in the baseline.
        examples, articles, evidence, journal, access, warnings = retrieve_context(
            message, session, context, backend, top_k, eligible=True
        )
        response = ResponseAgent(backend).run({
            'user_profile': profile.model_dump() if profile else {}, 'user_message': message,
            'context': context.model_dump(), 'session': session.summary(),
            'plan': {'selected_path': [], 'status': 'single_agent_baseline'},
            'research': evidence, 'journal_context': journal, 'journal_access': access,
            'research_access': {'requested': asks_about_research(message), 'available_count': len(evidence),
                                'status': 'unavailable' if warnings else 'retrieved' if evidence else 'empty'},
            'synthetic_examples': [{'message': ex.user_message, 'intervention': ex.tot.selected_intervention} for ex in examples],
        }, history)
    check = validate_output(response, context, len(evidence))
    if not check.valid:
        response = FALLBACK_RESPONSE
        warnings.append('Response checks requested a safer fallback.')
    session.record(message, response)
    return {'response': response, 'articles': articles, 'warnings': warnings,
            'context': context.model_dump(), 'plan': {'selected_path': [], 'status': route if route != 'standard_cbt' else 'single_agent_baseline'},
            'session': session.summary(), 'decision_source': 'single_agent_baseline',
            'validation': check.model_dump()}
