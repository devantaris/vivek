"""Prompt templates for instruction-tuned LLM SFT (Phase 2).

The model must output BOTH a binary label and a short rationale. A fixed
machine-parseable output format is essential: the Phase-3 reward functions
parse exactly this format out of the generated text.
"""

from __future__ import annotations

INSTRUCTION = (
    "You are a content moderation assistant. Classify the following social "
    "media post as hateful (1) or non-hateful (0), and briefly state which "
    "words or reasoning justify your decision.\n"
    "Answer in exactly this format:\n"
    "Label: <0 or 1>\n"
    "Rationale: <one or two short sentences naming the relevant words>"
)

USER_TEMPLATE = "{instruction}\n\nPost: {text}"


def format_prompt(text: str) -> str:
    """Render the user prompt for one post.

    Args:
        text: The (cleaned) social media post.

    Returns:
        The complete user-turn prompt string.
    """
    return USER_TEMPLATE.format(instruction=INSTRUCTION, text=text)


def parse_output(generation: str) -> tuple[int | None, str | None]:
    """Parse ``Label: <int>`` and ``Rationale: <str>`` out of a generation.

    Args:
        generation: Raw model output.

    Returns:
        ``(label, rationale)`` where either element is None if unparseable.
        Unparseable outputs must be surfaced (and punished by the label
        reward in Phase 3), never silently defaulted.
    """
    label: int | None = None
    rationale: str | None = None
    for line in generation.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("label:"):
            token = stripped.split(":", 1)[1].strip()
            if token and token[0] in "01":
                label = int(token[0])
        elif stripped.lower().startswith("rationale:"):
            rationale = stripped.split(":", 1)[1].strip() or None
    return label, rationale
