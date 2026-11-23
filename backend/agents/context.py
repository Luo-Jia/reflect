"""Context extraction agent with a validated, conservative fallback."""
import logging
from backend.schemas.context import UserContext

logger = logging.getLogger(__name__)

class ContextAgent:
    def __init__(self, backend):
        self.backend = backend

    def run(self, message, history):
        try:
            context = self.backend.extract(message, history=history)
            if not isinstance(context, UserContext):
                context = UserContext.model_validate(context)
            return context, 'completed', []
        except (ValueError, AttributeError) as exc:
            # Record diagnostic metadata, not user text or the model's output.
            logger.warning('Context extraction failed: exception=%s attribute=%s',
                           type(exc).__name__, getattr(exc, 'name', None))
            return UserContext(readiness='unknown', needs_clarification=True), 'degraded', [
                f'Structured context unavailable; continuing with conversation only: {exc}'
            ]
