
from backend.retrieval.synthetic_rag import retrieve_similar_cases
from backend.retrieval.search_knowledge import search_research as retrieve_evidence

def dual_rag_node(state):
    context = state.get("context") or {}

    message = state["user_message"]

    similar_cases = retrieve_similar_cases(
        message,
        top_k=3
    )

    # Generic intervention-focused research query.
    # Avoid including sensitive personal text in queries.
    preference = context.get("preference", "unspecified")

    evidence = retrieve_evidence(
        f"cognitive behavioral therapy {preference}",
        top_k=5
    )

    return {
        "similar_cases": similar_cases,
        "retrieved_evidence": evidence
    }
