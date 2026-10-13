"""Build Phase-2 SFT prompt/completion pairs from the unified schema.

Completions use the fixed ``Label:/Rationale:`` format that
:func:`pj03.sft.prompt_templates.parse_output` parses, so every SFT target
round-trips through the Phase-3 reward parser.

Gold rationale priority:
1. ``rationale_mask`` tokens (annotator-flagged words, ≥2/3 agreement).
2. Target-group template for hateful posts without a mask.
3. Generic template (kept deliberately short and non-committal).
"""

from __future__ import annotations

from pj03.preprocess.schema import Example
from pj03.sft.prompt_templates import format_prompt, parse_output

MAX_RATIONALE_WORDS = 20


def build_completion(example: Example) -> str:
    """Render the SFT target string for one example.

    Args:
        example: Unified-schema example (with optional rationale_mask).

    Returns:
        A ``Label: <0|1>\\nRationale: <text>`` string. The rationale quotes
        at most :data:`MAX_RATIONALE_WORDS` annotator-flagged tokens.
    """
    tokens = example.text.split()
    if example.rationale_mask:
        words = [tokens[i] for i in example.rationale_mask if i < len(tokens)]
        rationale = "the key words are: " + " ".join(words[:MAX_RATIONALE_WORDS])
    elif example.label == 1 and example.target_group:
        groups = example.target_group.replace("|", " and ")
        rationale = f"the post contains hateful language targeting {groups}"
    elif example.label == 1:
        rationale = "the post contains hateful language"
    else:
        rationale = "the post does not contain hateful language"
    return f"Label: {example.label}\nRationale: {rationale}"


def build_sft_records(examples: list[Example]) -> list[dict[str, str]]:
    """Convert schema examples to SFT records.

    Args:
        examples: Unified-schema examples from any processed dataset.

    Returns:
        List of ``{"id", "prompt", "completion"}`` dicts. Every completion is
        asserted to round-trip through the output parser.
    """
    records: list[dict[str, str]] = []
    for ex in examples:
        completion = build_completion(ex)
        parsed_label, _ = parse_output(completion)
        if parsed_label != ex.label:
            raise ValueError(f"completion for {ex.id} does not parse back to label {ex.label}")
        records.append(
            {"id": ex.id, "prompt": format_prompt(ex.text), "completion": completion}
        )
    return records
