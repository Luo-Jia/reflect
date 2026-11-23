"""Parse context JSON without loading the language model."""
import json
import logging

from pydantic import ValidationError

from backend.schemas.context import UserContext

logger = logging.getLogger(__name__)

def parse_context(text: str) -> UserContext:
    # Some models wrap otherwise valid JSON in Markdown or explanatory text.
    decoder = json.JSONDecoder()
    errors = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[index:])
            context = UserContext.model_validate(data)
            if context.readiness == "unknown" or context.intensity is None:
                context.needs_clarification = True
            return context
        except ValidationError as exc:
            # Record field paths and error types, never user/model field values.
            detail = "; ".join(
                f"{'.'.join(map(str, error['loc']))}: {error['type']}"
                for error in exc.errors(include_input=False, include_context=False)
            )
            errors.append(detail)
        except ValueError as exc:
            errors.append(str(exc))
    detail = errors[0] if errors else "Model output did not contain a JSON object"
    logger.warning("Context extraction failed: %s", detail)
    raise ValueError(detail)
