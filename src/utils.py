"""Common helper functions."""

import string


def normalize_answer(answer: str) -> str:
    """Lowercase an answer and remove punctuation for simple comparison."""
    cleaned = answer.strip().lower()
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))

    yes_values = {"yeah", "yep", "true"}
    no_values = {"nope", "false"}

    if cleaned in yes_values:
        return "yes"
    if cleaned in no_values:
        return "no"

    return " ".join(cleaned.split())
