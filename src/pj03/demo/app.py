"""Gradio demo app for the PJ-03 hate-speech classifier.

Run with ``python scripts/demo_app.py`` (or ``make demo``) after training the
artifact with ``python scripts/train_demo_model.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pj03.demo.predictor import DEFAULT_ARTIFACT, DemoPredictor
from pj03.utils.logging import get_logger

logger = get_logger("pj03.demo_app")

ETHICS_BANNER = (
    "⚠️ **Research prototype (PJ-03, Bennett University).** This tool is "
    "trained on social-media data containing offensive content and exists "
    "for research and demonstration only. It is designed to *support* human "
    "moderators, not replace them — every automated decision should be "
    "reviewed by a person. Annotation of hate speech is subjective; the "
    "label mapping follows HateXplain's majority-vote annotations "
    "('hatespeech' vs 'offensive/normal'). See `docs/ethics.md`."
)

EXAMPLES = [
    ["kill all jews"],
    ["you are all subhuman trash and should leave"],
    ["good morning friends, have a lovely day"],
    ["i hate women. they are all worthless"],  # 'offensive', not 'hatespeech', in HateXplain
]


def _highlighted(result) -> list[tuple[str, str | None]]:
    """Convert a prediction to Gradio HighlightedText segments."""
    if not result.rationale:
        return [(result.text, None)]
    weights = {ngram: w for ngram, w in result.rationale}
    tokens = result.text.split()
    spans: list[tuple[str, str | None]] = []
    for tok in tokens:
        low = tok.lower()
        score = 0.0
        for ngram, w in weights.items():
            if low and low in ngram and ngram.strip().isalnum():
                score = w if abs(w) > abs(score) else score
        if score > 0.05:
            spans.append((tok + " ", "pushes hateful"))
        elif score < -0.05:
            spans.append((tok + " ", "pushes non-hateful"))
        else:
            spans.append((tok + " ", None))
    return spans


def _summary_md(result) -> str:
    label_word = "HATEFUL" if result.label == 1 else "NON-HATEFUL"
    md = f"## Prediction: **{label_word}** (P(hateful) = {result.prob_hateful:.3f})\n"
    if result.rationale:
        rows = "\n".join(
            f"| `{ngram}` | {weight:+.3f} |"
            for ngram, weight in result.rationale[:5]
        )
        md += (
            "\n**Model-grounded rationale** — each n-gram's contribution to "
            "the decision (tfidf × weight):\n\n"
            "| n-gram | contribution |\n|---|---|\n" + rows + "\n"
        )
    if result.comprehensiveness is not None:
        md += (
            f"\n**Faithfulness (live, this input)**: comprehensiveness = "
            f"{result.comprehensiveness:.3f}, sufficiency = "
            f"{result.sufficiency:.3f}\n\n"
            "- *Comprehensiveness*: removing the top-3 attributed tokens "
            "changes P(hateful) by this much. Higher = the highlighted words "
            "genuinely drive the decision.\n"
            "- *Sufficiency*: keeping only those tokens retains this much of "
            "the prediction. Closer to 0 = the rationale alone suffices.\n"
        )
    md += (
        "\n*Note: borderline or surprising labels are informative — "
        "hate-speech annotation is subjective (see the banner above).*"
    )
    return md


def _model_card_md() -> str:
    # src/pj03/demo/app.py -> repo root (works regardless of the launch CWD)
    repo_root = Path(__file__).resolve().parents[3]
    results_dir = repo_root / "data" / "processed" / "results"
    md = "## Model card\n\n- **Model**: TF-IDF (word 1–2 + char_wb 2–5) → "
    md += "Logistic Regression (`balanced` class weights)\n"
    md += "- **Training data**: HateXplain train split (15,383 posts; "
    md += "hatespeech=1, offensive/normal=0 by majority vote)\n"
    path = results_dir / "demo_tfidf.json"
    if path.is_file():
        metrics = json.loads(path.read_text(encoding="utf-8"))
        md += "\n**HateXplain test metrics**:\n\n| metric | value |\n|---|---|\n"
        md += "".join(f"| {k} | {v:.4f} |\n" for k, v in metrics.items())
    audit_path = results_dir / "tfidf_lr_hatexplain.json"
    if audit_path.is_file():
        data = json.loads(audit_path.read_text(encoding="utf-8"))
        audit = data.get("bias_audit", {})
        if audit:
            md += "\n**Bias audit (test, Borkan et al. metrics)**:\n\n"
            md += "| group | subgroup AUC | BPSN | BNSP |\n|---|---|---|---|\n"
            md += "".join(
                f"| {g} | {v['subgroup_auc']:.3f} | {v['bpsn_auc']:.3f} | "
                f"{v['bnsp_auc']:.3f} |\n"
                for g, v in audit.items()
            )
            md += (
                "\nLow BPSN = benign posts merely *mentioning* the group are "
                "over-flagged; low BNSP = hateful posts targeting the group "
                "are missed. See `docs/ethics.md`.\n"
            )
    return md


def create_demo(predictor: DemoPredictor):
    """Build the Gradio Blocks interface."""
    import gradio as gr

    with gr.Blocks(title="PJ-03 Hate Speech Detection — Explainable Demo") as demo:
        gr.Markdown("# PJ-03 — Explainable Hate Speech Detection (demo)")
        gr.Markdown(ETHICS_BANNER)

        with gr.Tab("Classify & Explain"):
            text_in = gr.Textbox(
                label="Post text", placeholder="Paste a social media post…", lines=3
            )
            btn = gr.Button("Classify & explain", variant="primary")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Highlighted rationale")
                    highlight = gr.HighlightedText(
                        label="Token attributions",
                        color_map={
                            "pushes hateful": "red",
                            "pushes non-hateful": "green",
                        },
                        combine_adjacent=True,
                        show_legend=True,
                    )
                with gr.Column():
                    summary = gr.Markdown()
            gr.Examples(examples=EXAMPLES, inputs=text_in)
            def _run(text: str):
                result = predictor.predict(text)
                return _highlighted(result), _summary_md(result)

            btn.click(_run, inputs=text_in, outputs=[highlight, summary])

        with gr.Tab("Model card & fairness"):
            gr.Markdown(_model_card_md())

        gr.Markdown(
            "*Next phases (in development): LoRA fine-tuned LLM rationales, "
            "reward-optimised post-training (GRPO), SHAP/Integrated-Gradients "
            "attributions.*"
        )
    return demo


def main(artifact: str | Path = DEFAULT_ARTIFACT, port: int = 7860) -> None:
    """Launch the demo app."""

    predictor = DemoPredictor(artifact)
    demo = create_demo(predictor)
    demo.queue(default_concurrency_limit=4)
    demo.launch(server_port=port, inbrowser=True, show_error=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PJ-03 demo app")
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(args.artifact, args.port)
