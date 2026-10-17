"""REST API service for the PJ-03 classifier (productisation layer).

Endpoints:
    GET  /health   - service + model status (no PII, safe for monitors)
    POST /predict  - label + probability + rationale + live faithfulness

Every prediction is appended to a JSONL decision log with the input PII
scrubbed (never the raw text), a unique ``decision_id``, the threshold used
and the full explanation payload - the auditable record a moderation
workflow needs. Run with ``make serve`` or ``python scripts/serve_api.py``.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from pj03.preprocess.pii import has_pii, scrub
from pj03.utils.logging import get_logger

logger = get_logger("pj03.service")

DISCLAIMER = (
    "Research prototype. Assists human moderators; not a decision-maker. "
    "Hate-speech annotation is subjective."
)


class PredictRequest(BaseModel):
    """Body of POST /predict."""

    text: str = Field(min_length=1, max_length=20_000)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    scrub_pii: bool = False  # scrub before *prediction* (logs always scrub)


class RationaleItem(BaseModel):
    """One n-gram contribution to the decision."""

    ngram: str
    contribution: float


class PredictResponse(BaseModel):
    """Response of POST /predict."""

    decision_id: str
    label: int
    prob_hateful: float
    threshold: float
    rationale: list[RationaleItem]
    comprehensiveness: float | None
    sufficiency: float | None
    pii_in_input: bool
    input_scrubbed_for_prediction: bool
    latency_ms: float
    disclaimer: str


def create_app(predictor, decision_log: str | Path = "logs/decisions.jsonl"):
    """Build the FastAPI app around a :class:`~pj03.demo.predictor.DemoPredictor`."""
    from fastapi import FastAPI

    log_path = Path(decision_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def append_log(entry: dict) -> None:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    app = FastAPI(
        title="PJ-03 Hate Speech Detection",
        description=DISCLAIMER,
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict:
        metrics = predictor.metadata.get("test_metrics", {})
        return {
            "status": "ok",
            "model": "tfidf_word+char_logreg",
            "test_macro_f1": metrics.get("macro_f1"),
            "test_auc": metrics.get("auc"),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        started = time.perf_counter()
        decision_id = uuid.uuid4().hex[:12]
        flagged = has_pii(req.text)
        model_input = scrub(req.text) if (flagged and req.scrub_pii) else req.text

        result = predictor.predict(model_input)
        label = int(result.prob_hateful >= req.threshold)
        response = PredictResponse(
            decision_id=decision_id,
            label=label,
            prob_hateful=round(result.prob_hateful, 4),
            threshold=req.threshold,
            rationale=[
                RationaleItem(ngram=ngram, contribution=round(weight, 4))
                for ngram, weight in result.rationale
            ],
            comprehensiveness=(
                round(result.comprehensiveness, 4)
                if result.comprehensiveness is not None
                else None
            ),
            sufficiency=(
                round(result.sufficiency, 4) if result.sufficiency is not None else None
            ),
            pii_in_input=flagged,
            input_scrubbed_for_prediction=bool(flagged and req.scrub_pii),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            disclaimer=DISCLAIMER,
        )
        # Stored log never contains raw text, by construction.
        append_log(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "decision_id": decision_id,
                "input_scrubbed": scrub(req.text),
                "label": label,
                "prob_hateful": response.prob_hateful,
                "threshold": req.threshold,
                "rationale": [item.model_dump() for item in response.rationale],
                "comprehensiveness": response.comprehensiveness,
                "sufficiency": response.sufficiency,
                "pii_in_input": flagged,
                "latency_ms": response.latency_ms,
            }
        )
        return response

    return app
