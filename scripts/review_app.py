"""Moderation review UI: human-in-the-loop queue over the decision log.

Run with ``make review`` (or ``python scripts/review_app.py``) while the API
(``make serve``) is producing decisions. Each confirmation/override is stored
in ``logs/feedback.jsonl`` and becomes training signal for the next model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pj03.service.feedback import FeedbackRecord, FeedbackStore  # noqa: E402

BANNER = (
    "⚠️ Review queue for the PJ-03 research prototype. Items shown are the "
    "model's *scrubbed* decision records. Your confirmations and overrides "
    "are logged and become the hard-example set for the next training round."
)


def _entry_summary(entry: dict) -> str:
    rationale = ", ".join(
        f"{r['ngram']} ({r['contribution']:+.2f})" for r in entry.get("rationale", [])[:4]
    )
    comp = entry.get("comprehensiveness")
    suff = entry.get("sufficiency")
    return (
        f"**{entry['decision_id']}** — model label **{entry['label']}** "
        f"(P={entry['prob_hateful']:.3f}, threshold={entry['threshold']})\n\n"
        f"> {entry.get('input_scrubbed', '')}\n\n"
        f"Rationale: {rationale or '—'}\n\n"
        f"Faithfulness: comprehensiveness={comp if comp is not None else '—'}, "
        f"sufficiency={suff if suff is not None else '—'}"
    )


def create_review_app(decisions: Path, feedback: Path):
    """Build the Gradio Blocks review interface."""
    import gradio as gr

    store = FeedbackStore(feedback)

    def refresh(limit: int = 20):
        pending = store.pending_queue(decisions, limit=limit)
        if not pending:
            return [[]], "Queue empty — no unreviewed decisions.", ""
        rows = [
            [
                e["decision_id"],
                e["label"],
                f"{e['prob_hateful']:.3f}",
                (e.get("input_scrubbed", "") or "")[:120],
            ]
            for e in pending
        ]
        return rows, f"{len(pending)} unreviewed decision(s).", _entry_summary(pending[0])

    def submit(decision_id: str, final_label: str, moderator: str, note: str):
        if not decision_id:
            return "No decision selected.", refresh()[0]
        final = 1 if final_label.lower().startswith("hate") else 0
        entries = [
            json.loads(line)
            for line in decisions.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {e["decision_id"]: e for e in entries}
        if decision_id not in by_id:
            return f"Unknown decision_id {decision_id!r}.", refresh()[0]
        model_label = by_id[decision_id]["label"]
        record = store.append(
            FeedbackRecord(
                decision_id=decision_id,
                model_label=model_label,
                final_label=final,
                agrees=model_label == final,
                moderator=moderator or "anonymous",
                note=note or "",
            )
        )
        status = "confirmed" if record.agrees else "OVERRIDDEN"
        _, table, _ = refresh()
        return (
            f"Stored: {record.decision_id} {status} (final={record.final_label}). "
            f"Override rate so far: {store.override_rate():.1%}",
            table,
        )

    with gr.Blocks(title="PJ-03 Moderation Review") as demo:
        gr.Markdown("# PJ-03 — Moderation Review Queue")
        gr.Markdown(BANNER)
        with gr.Row():
            refresh_btn = gr.Button("Refresh queue")
        queue_table = gr.Dataframe(
            headers=["decision_id", "model label", "P(hateful)", "text (scrubbed)"],
            label="Unreviewed decisions (newest first)",
            interactive=False,
        )
        queue_status = gr.Markdown()
        with gr.Row():
            selected_id = gr.Textbox(label="decision_id (paste from table)")
            final_choice = gr.Radio(
                ["hateful (1)", "non-hateful (0)"], label="Final label", value="hateful (1)"
            )
        with gr.Row():
            moderator = gr.Textbox(label="Moderator", placeholder="your name/id")
            note = gr.Textbox(label="Note (optional)")
        submit_btn = gr.Button("Record decision", variant="primary")
        submit_status = gr.Markdown()
        selected_detail = gr.Markdown()

        refresh_btn.click(refresh, outputs=[queue_table, queue_status, selected_detail])
        submit_btn.click(
            submit,
            inputs=[selected_id, final_choice, moderator, note],
            outputs=[submit_status, queue_table],
        )
        demo.load(refresh, outputs=[queue_table, queue_status, selected_detail])
    return demo


def main(decisions: Path, feedback: Path, port: int = 7861) -> None:
    demo = create_review_app(decisions, feedback)
    demo.queue(default_concurrency_limit=2)
    demo.launch(server_port=port, inbrowser=True, show_error=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    main(
        REPO_ROOT / "logs" / "decisions.jsonl",
        REPO_ROOT / "logs" / "feedback.jsonl",
        args.port,
    )
