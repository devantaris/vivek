"""Bias reward (penalised): does the rationale lean on identity terms?

TODO(Phase 3): plan - the reward should penalise rationales that simply quote
an identity term ("muslim", "women", ...) as the justification instead of
citing actual toxic content:

1. Build a lexicon of protected/identity terms (carefully sourced; see
   docs/ethics.md).
2. Score = fraction of rationale tokens that are identity terms.
3. Return ``1.0 - score`` (so higher is less identity-reliant), clipped to
   [0, 1]; the composite subtracts this with weight gamma.

Keep the contract: return a float in [0, 1].
"""

from __future__ import annotations

# Placeholder seed lexicon; TODO(Phase 3): source a proper, documented lexicon
# and have it reviewed for the ethics writeup.
_IDENTITY_TERMS: frozenset[str] = frozenset(
    {"muslim", "hindu", "women", "man", "gay", "jew", "christian", "dalit", "immigrant"}
)


def r_bias(generation: str) -> float:
    """Reward in [0, 1]; higher = rationale does NOT merely cite identity.

    The composite reward subtracts ``gamma * r_bias``.
    """
    from pj03.sft.prompt_templates import parse_output

    _, rationale = parse_output(generation)
    if not rationale:
        return 0.5
    tokens = [t.strip(".,!?;:\"'").lower() for t in rationale.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return 0.5
    identity_hits = sum(1 for t in tokens if t in _IDENTITY_TERMS)
    return max(0.0, min(1.0, 1.0 - identity_hits / len(tokens)))
