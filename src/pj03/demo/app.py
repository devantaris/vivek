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
    import re

    from pj03.preprocess.hinglish import _VARIANT_MAP

    weights = {ngram.lower(): w for ngram, w in result.rationale}
    tokens = result.text.split()
    spans: list[tuple[str, str | None]] = []
    for tok in tokens:
        clean_tok = "".join(c for c in tok.lower() if c.isalnum())
        norm_tok = _VARIANT_MAP.get(clean_tok, clean_tok)
        score = 0.0
        for ngram, w in weights.items():
            ngram_words = set(re.findall(r"\b\w+\b", ngram))
            if (
                clean_tok in ngram_words
                or norm_tok in ngram_words
                or clean_tok == ngram
                or norm_tok == ngram
            ):
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
    md += f"- **Active Model Architecture:** `{result.model_name}`\n"
    if result.natural_rationale:
        md += f"- **Model Rationale:** *{result.natural_rationale}*\n\n"
    if result.rationale:
        rows = "\n".join(f"| `{ngram}` | {weight:+.3f} |" for ngram, weight in result.rationale[:5])
        md += (
            "**Key Attribution Tokens / Rationales:**\n\n"
            "| Token / n-gram | Impact |\n|---|---|\n" + rows + "\n\n"
        )
    if result.comprehensiveness is not None:
        md += (
            f"**Faithfulness Metrics (Live)**: comprehensiveness = "
            f"{result.comprehensiveness:.3f}, sufficiency = "
            f"{result.sufficiency:.3f}\n\n"
            "- *Comprehensiveness*: removing the top-attributed tokens "
            "shifts P(hateful) by this much. Higher = tokens genuinely drive the decision.\n"
            "- *Sufficiency*: keeping only those tokens retains this much prediction.\n"
        )
    md += "\n*Note: hate-speech annotation is subjective; review decisions with human oversight.*"
    return md


def _model_card_md() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    results_dir = repo_root / "data" / "processed" / "results"
    md = "## PJ-03 Multi-Phase Model Card & Leaderboard\n\n"
    md += "| Model Architecture | Phase | Status | Key Evaluation Metric | Notes |\n"
    md += "|---|---|---|---|---|\n"
    md += "| **Qwen2.5-1.5B LoRA SFT** | Phase 2 (LLM) | **Trained (Colab T4)** | `eval_loss: 0.1257` | Emits natural language rationales + predictions |\n"
    md += "| **Google MuRIL** | Phase 1 (Deep Encoder) | **Trained (Colab T4)** | `macro_f1: 0.7669`, `auc: 0.8494` | Pre-trained for Indian languages & Hinglish |\n"
    md += "| **TF-IDF + LogReg** | Phase 1 (Baseline) | **Trained (CPU)** | `macro_f1: 0.8096`, `auc: 0.8987` | N-gram linear classifier baseline |\n"
    md += "| **HateCheck Suite** | Phase 4 (Functional) | **Evaluated** | `accuracy: 0.3600` | 29 functional tests across 3,728 cases |\n\n"

    path = results_dir / "eval_phase4_report.json"
    if path.is_file():
        report = json.loads(path.read_text(encoding="utf-8"))
        fairness = report.get("fairness_audit", {})
        if fairness:
            md += "### Bias & Fairness Audit (Borkan et al. Subgroup Metrics)\n\n"
            md += "| Demographic Group | Subgroup AUC | BPSN AUC | BNSP AUC |\n|---|---|---|---|\n"
            for g, v in fairness.items():
                md += f"| **{g}** | {v['subgroup_auc']:.3f} | {v['bpsn_auc']:.3f} | {v['bnsp_auc']:.3f} |\n"
            md += "\n- **BPSN**: Background Positive Subgroup Negative (measures false alarms on benign mentions of group).\n"
            md += "- **BNSP**: Background Negative Subgroup Positive (measures missed hate targeting the group).\n"
    return md


def create_demo(predictor: DemoPredictor):
    """Build the Gradio Blocks interface."""
    import gradio as gr

    model_choices = ["Auto (Best Available)"]
    if predictor.has_llm:
        model_choices.append("Qwen2.5-1.5B LoRA SFT (Deep LLM)")
    model_choices.append("TF-IDF + LogReg Baseline")

    with gr.Blocks(title="PJ-03 Hate Speech Detection — Explainable Demo") as demo:
        gr.Markdown("# PJ-03 — Explainable Hate Speech Detection (demo)")
        gr.Markdown(ETHICS_BANNER)

        with gr.Tab("Classify & Explain"):
            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=model_choices,
                    value=model_choices[0],
                    label="Model Architecture",
                )
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

            def _run(text: str, model_choice: str):
                m_type = (
                    "llm"
                    if "LoRA" in model_choice
                    else ("tfidf" if "TF-IDF" in model_choice else "auto")
                )
                result = predictor.predict(text, model_type=m_type)
                return _highlighted(result), _summary_md(result)

            btn.click(_run, inputs=[text_in, model_selector], outputs=[highlight, summary])

        with gr.Tab("Model card & fairness"):
            gr.Markdown(_model_card_md())

        gr.Markdown(
            "*PJ-03: Bennett University NLP Research Project. Models trained across CPU & Colab T4 GPU.*"
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
