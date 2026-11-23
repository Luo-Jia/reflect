import json
from pathlib import Path
from pydantic import ValidationError
from backend.schemas.schemas import CBTExample

def load_dataset(path: str) -> list[CBTExample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = data.get("examples", data.get("records"))
    else:
        raise ValueError("Unsupported dataset format")

    if not isinstance(records, list):
        raise ValueError("Expected an examples or records array")

    examples = []
    seen = set()

    for index, raw in enumerate(records):
        try:
            example = CBTExample.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid example at index {index}: {exc}"
            ) from exc

        if example.case_id in seen:
            raise ValueError(f"Duplicate ID: {example.case_id}")

        seen.add(example.case_id)
        examples.append(example)

    return examples
