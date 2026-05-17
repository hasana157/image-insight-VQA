"""Rule-based question type detection for VQA examples."""

from __future__ import annotations

YES_NO_STARTERS = ("is", "are", "does", "do", "can", "could", "has", "have", "was", "were", "will", "would")

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
    "yes_no": ("is there", "are there", "can you see"),
    "color": ("what color", "color of", "color is", "color are", "colour", "wearing", "shape", "material"),
    "object": ("what is this", "what object", "identify", "what item", "what animal", "what vehicle", "what fruit"),
    "action": (
        "doing",
        "playing",
        "riding",
        "holding",
        "eating",
        "drinking",
        "walking",
        "running",
        "sport",
        "activity",
        "game",
    ),
    "spatial_scene": (
        "where",
        "behind",
        "left",
        "in front",
        " on ",
        "under",
        "right",
        "scene",
        "next to",
        "near",
        "room",
        "place",
        "location",
        "outside",
        "inside",
    ),
}


def classify_question_type(question: str) -> str:
    """Classify a question into the project taxonomy."""
    compact_question = question.lower().strip()
    normalized = f" {compact_question} "

    first_word = compact_question.split(maxsplit=1)[0] if compact_question else ""
    if first_word in YES_NO_STARTERS:
        return "yes_no"

    for question_type, keywords in QUESTION_TYPE_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return question_type

    return "object"
