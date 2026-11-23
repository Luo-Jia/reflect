"""Retrieve training-only synthetic planning references."""
import re
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=1)
def get_examples():
    from backend.database.loader import load_dataset

    return load_dataset(str(Path(__file__).resolve().parents[1] / "data/cbt_tot_synthetic_200.json"))


def similar_examples(message, top_k=3):
    # Local lexical retrieval keeps personal messages on this machine.
    examples = get_examples()
    words = set(re.findall(r"\w+", message.lower()))
    matches = []
    for example in examples:
        if example.metadata.split != "train" or example.safety.route != "standard_cbt":
            continue
        other = set(re.findall(r"\w+", example.user_message.lower()))
        score = len(words & other) / max(1, len(words | other))
        if score:
            matches.append((score, example))
    matches.sort(key=lambda item: item[0], reverse=True)
    return [example for _, example in matches[:top_k]]


