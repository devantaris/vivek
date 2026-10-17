"""Human-in-the-loop feedback store for the moderation workflow.

Moderators review the decision log (queue) and record confirmations or
overrides. Overrides are the training signal for future model updates:
``final_label != model_label`` rows are exactly the hard examples to add to
the next training round.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel, Field


class FeedbackRecord(BaseModel):
    """One moderator decision on one model prediction."""

    decision_id: str = Field(min_length=1)
    model_label: int
    final_label: int
    agrees: bool
    moderator: str = Field(min_length=1)
    note: str = ""
    ts: str = ""


class FeedbackStore:
    """Appends feedback records and indexes reviewed decision ids."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: FeedbackRecord) -> FeedbackRecord:
        """Store one feedback record (ts filled if empty)."""
        if not record.ts:
            record.ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")
        return record

    def reviewed_ids(self) -> set[str]:
        """Decision ids that already have feedback."""
        if not self.path.is_file():
            return set()
        ids: set[str] = set()
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    ids.add(json.loads(line)["decision_id"])
        return ids

    def pending_queue(self, decisions_path: str | Path, limit: int = 50) -> list[dict]:
        """Unreviewed decisions, newest first.

        Args:
            decisions_path: The JSONL decision log written by the API.
            limit: Max entries returned.

        Returns:
            List of decision dicts (scrubbed input included, never raw text).
        """
        reviewed = self.reviewed_ids()
        decisions_path = Path(decisions_path)
        if not decisions_path.is_file():
            return []
        pending: list[dict] = []
        with decisions_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("decision_id") not in reviewed:
                    pending.append(entry)
        pending.reverse()  # newest first
        return pending[:limit]

    def override_rate(self) -> float | None:
        """Fraction of feedback that overrode the model (None if no feedback)."""
        if not self.path.is_file():
            return None
        total, overrides = 0, 0
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                total += 1
                overrides += 1 if not json.loads(line)["agrees"] else 0
        return overrides / total if total else None
