def qwen_context_node(state):
    from backend.model.qwen_extractor import extract_context
    try:
        context = extract_context(
            state["user_message"]
        )
        return {
            "context": context.model_dump(),
            "context_status": "success"
        }
    except Exception:
        return {
            "context": None,
            "context_status": "needs_clarification"
        }
