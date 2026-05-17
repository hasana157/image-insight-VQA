"""Rule-based question type detection for VQA examples."""

from __future__ import annotations

QUESTION_TYPES = (
    "counting",
    "yes_no",
    "color",
    "object",
    "action",
    "spatial_scene",
)

QUESTION_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "counting": ("how many", "number of", "count"),
    "yes_no": ("is there", "are there", "does", "do ", "can you see", "has ", "have "),
    "color": ("what color", "colour", "wearing", "type", "shape", "material"),
    "object": ("what is this", "what object", "identify", "what item", "what animal"),
    "action": ("doing", "playing", "riding", "holding", "eating", "drinking", "walking", "running"),
    "spatial_scene": ("where", "behind", "in front", " on ", "under", "scene", "next to", "near"),
}


def classify_question_type(question: str) -> str:
    """Classify a question into the project taxonomy."""
    normalized = f" {question.lower().strip()} "

    for question_type, keywords in QUESTION_TYPE_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return question_type

    return "object"
