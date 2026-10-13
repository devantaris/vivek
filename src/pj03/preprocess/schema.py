"""Unified PJ-03 data schema.

Every dataset (HateXplain, HASOC, Hinglish offensive tweets, HateCheck) is
normalised into :class:`Example` / :class:`DatasetSplit` before it reaches any
model code, so downstream phases never depend on dataset-specific formats.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Language(str, Enum):
    """Language / code-mix tag for an example."""

    EN = "en"
    HI = "hi"
    HINGLISH = "hinglish"
    UNKNOWN = "unknown"


class Split(str, Enum):
    """Standard split names."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Example(BaseModel):
    """One normalised hate-speech example.

    Attributes:
        id: Unique identifier, conventionally ``{source_dataset}:{orig_id}``.
        text: Raw (post-cleaning) social media text.
        label: 0 = non-hateful, 1 = hateful. Binary for now; multi-class
            target-group labelling lives in ``target_group``.
        language: Language or code-mix tag.
        source_dataset: Which dataset the example came from
            (e.g. ``hatexplain``, ``hasoc_2021_hi``).
        split: Which split the example belongs to.
        rationale_mask: Optional token indices (into the tokenised ``text``)
            marking the words that justify the label; used for faithfulness
            metrics in Phase 4. Index convention is dataset-specific until
            tokenisation is fixed - document the choice in the loader.
        target_group: Optional identity group targeted (e.g. ``women``,
            ``muslims``); used for the bias audit in Phase 4.
    """

    id: str = Field(min_length=1)
    text: str
    label: int
    language: Language = Language.UNKNOWN
    source_dataset: str = Field(min_length=1)
    split: Split
    rationale_mask: list[int] | None = None
    target_group: str | None = None

    @field_validator("label")
    @classmethod
    def _label_is_binary(cls, value: int) -> int:
        if value not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {value!r}")
        return value

    @field_validator("rationale_mask")
    @classmethod
    def _mask_indices_nonnegative(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and any(i < 0 for i in value):
            raise ValueError("rationale_mask indices must be non-negative")
        return value


class DatasetStats(BaseModel):
    """Lightweight summary of a loaded dataset, for EDA and logging."""

    source_dataset: str
    n_train: int = 0
    n_validation: int = 0
    n_test: int = 0
    n_positive: int = 0
    n_negative: int = 0

    @property
    def total(self) -> int:
        """Total number of examples across all splits."""
        return self.n_train + self.n_validation + self.n_test
