# PJ-03: Hate Speech Detection through Explainable AI and Reward-Optimized LLMs

Bennett University NLP research project (PJ-03).

We classify **English** and **Hinglish** (Hindi–English code-mixed) social
media text as hateful / non-hateful, and output a **rationale** — the words or
reason behind the decision. The pipeline has four phases:

1. **Data prep + baselines** — TF-IDF+LogReg, mBERT, XLM-R, MuRIL.
2. **LoRA SFT** — a small (1–3B) instruction-tuned LLM fine-tuned to emit
   `Label:` + `Rationale:`.
3. **Reward-optimised post-training** — GRPO/DPO (Hugging Face TRL) with a
   composite reward `R = alpha·R_label + beta·R_faith − gamma·R_bias`.
4. **XAI & audit** — SHAP / Integrated Gradients (Captum), faithfulness
   metrics (comprehensiveness, sufficiency), bias audit (subgroup AUC,
   BPSN/BNSP), HateCheck functional tests, OOD evaluation.

**Planned datasets:** HateXplain, HASOC (Hindi/English/code-mixed), a Hinglish
offensive-tweet dataset, HateCheck. See `docs/dataset_notes.md` for licences
and access requirements.

> ⚠️ **Content warning:** this research involves offensive/hateful text.
> Research use only — see `docs/ethics.md`.

## Demo (demonstrable product)

A Gradio web app wraps the trained TF-IDF+LogReg classifier (word + char
n-grams, **macro-F1 0.812** on HateXplain test) with:

- **Classify & explain** — probability + highlighted token attributions
  (each n-gram's exact tfidf×weight contribution to the decision) and
  **live faithfulness metrics** (comprehensiveness/sufficiency computed on
  your input).
- **Model card & fairness** — test metrics plus the per-group bias audit
  (subgroup AUC, BPSN, BNSP) with plain-language interpretation.
- Ethics banner on every page: research prototype, supports (not replaces)
  human moderators, annotation is subjective.

```bash
make demo        # trains the artifact if missing (~30s), then opens the UI
```

Screenshots for slides: `demo screenshots coming — capture from the UI`.
Roadmap: LoRA-LLM natural-language rationales and GRPO reward-optimised
versions will slot into the same interface.

## Repository layout

```
pj03-hate-speech/
├── data/{raw,interim,processed}/   # datasets (gitignored; .gitkeep kept)
├── src/pj03/
│   ├── preprocess/   # cleaning, Hinglish normalisation, loaders, unified schema
│   ├── baselines/    # tfidf_lr.py, encoder_finetune.py        (Phase 1)
│   ├── sft/          # lora_sft.py, prompt_templates.py         (Phase 2)
│   ├── reward/       # r_label/r_faith/r_bias, composite, train_grpo (Phase 3)
│   ├── xai/          # attribution.py, faithfulness.py          (Phase 4)
│   ├── eval/         # metrics.py, fairness.py, hatecheck.py, ood.py
│   └── utils/        # config.py, seed.py, logging.py
├── configs/          # base.yaml, baselines.yaml, sft.yaml, reward.yaml, eval.yaml
├── scripts/          # download_data.py, run_baseline.py, run_sft.py,
│                     # run_reward.py, run_eval.py
├── notebooks/        # 00_eda.ipynb placeholder
├── tests/            # smoke tests (config, schema, metrics, faithfulness, reward)
├── paper/            # ACL LaTeX skeleton, references.bib, figures/
├── docs/             # ethics.md, dataset_notes.md, experiment_log.md
└── README.md, pyproject.toml, requirements.txt, Makefile, .gitignore,
    .env.example, LICENSE
```

## Setup

Requires Python **3.10+**. Everything imports and tests pass on CPU; actual
training (Phases 2–3) needs a GPU.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate        # Git Bash / Windows
# .\.venv\Scripts\Activate.ps1       # PowerShell
# source .venv/bin/activate          # Linux/macOS

# 2. Install
pip install -r requirements.txt
pip install -e .[dev]

# 3. Environment (Hugging Face token etc.)
cp .env.example .env                  # then edit .env
```

Or simply: `make setup`.

## Running the phases

```bash
make data        # download datasets (prints manual steps for gated ones)
make baseline    # Phase 1: TF-IDF+LogReg / encoder fine-tuning  (stub for now)
make sft         # Phase 2: LoRA SFT                             (stub for now)
make reward      # Phase 3: GRPO/DPO with composite reward       (stub for now)
make eval        # Phase 4: XAI, faithfulness, bias audit, HateCheck, OOD (stub)
```

Or directly: `python scripts/run_baseline.py baselines.tfidf_lr.C=3.0`
(any `key=value` pairs override the YAML config; nested keys use dots).

Config files live in `configs/`. The composite-reward weights are in
`configs/reward.yaml` under `reward.weights`.

## Tests & lint

```bash
make test     # pytest -q
make lint     # ruff check .
```

## Status

Scaffold + tested core utilities. Implemented for real: config loader with
overrides, seeding, logging, macro-F1/classification metrics, comprehensiveness
& sufficiency, subgroup AUC / BPSN / BNSP, composite reward with weights from
config, prompt templates with parsing, text cleaning, unified pydantic schema.
Everything else is an honest `NotImplementedError` stub with a TODO — see the
module docstrings for the intended behaviour.

## Data & ethics notes

- Datasets: see `docs/dataset_notes.md` (access requirements, licences).
- Ethics: see `docs/ethics.md` — offensive content, research use only, the
  system supports human moderators and does not replace them, annotation is
  subjective, bias risks are real and audited in Phase 4.

## Team

| Member | ID |
|---|---|
| Devansh Kumar | S24CSEU0828 |
| Ayush | S24CSEU0130 |
| Saksham Tyagi | S24CSEU1099 |
| Mudit Agrawal | S24CSEU0349 |

**Supervisor:** Dr. Pradeepika Verma

## Licence

MIT (placeholder) — see `LICENSE`. Dataset licences are separate and recorded
in `docs/dataset_notes.md`.
