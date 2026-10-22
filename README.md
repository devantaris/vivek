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

## Serving as a REST API (productisation)

```bash
make serve        # FastAPI on http://127.0.0.1:8000 (docs at /docs)
```

- `GET /health` — model status + headline metrics (monitor-safe, no PII).
- `POST /predict` — `{text, threshold, scrub_pii}` → label, probability,
  n-gram rationale with contributions, live comprehensiveness/sufficiency,
  latency. Threshold is per-request (0–1) so sensitivity is tunable per
  customer/workflow.
- **Auditable decision log** (`logs/decisions.jsonl`): every prediction is
  stored with a `decision_id`, the threshold used and the full explanation —
  with the input text **PII-scrubbed at write time** (emails/URLs/mentions/
  phones → `[EMAIL]`/`[URL]`/`[USER]`/`[PHONE]`); raw user text is never
  persisted.
- `Dockerfile` included: `docker build -t pj03-api . && docker run -p 8000:8000 pj03-api`.

## Release gate (quality + fairness regression check)

```bash
make gate         # exit 0 = shippable, exit 1 = regressions, report JSON written
```

Retrains the current model fresh (~25s CPU), then enforces the floors in
`configs/release.yaml`: macro-F1 ≥ 0.78, AUC ≥ 0.85, and worst-group
fairness floors (subgroup AUC / BPSN / BNSP ≥ 0.60–0.65). Verified both
ways: passes on the current model, fails with clear per-gate reasons when
floors are impossible. Every future model (SFT, GRPO) must clear this gate
before shipping.

## Human-in-the-loop review (`make review`)

The moderation workflow closes the loop: while the API serves predictions,
the review app shows the unreviewed decision queue (scrubbed text, model
label, probability, rationale, faithfulness). Moderators confirm or override;
every override is logged and becomes the hard-example set for the next
training round. Endpoints: `GET /queue`, `POST /feedback`.

## Drift monitoring

`python scripts/drift_report.py` computes label/PII/latency/override-rate
statistics over the decision log (by-day breakdown) and raises alerts when
thresholds are crossed — exit 1 in CI. Reports land in
`data/processed/results/drift_report.json`.

## Data licences

See `docs/licence_audit.md` before any deployment or commercial
conversation. Short version: HateCheck is CC BY 4.0 (eval-only); HateXplain
models are research artefacts; HASOC/Hinglish terms pending.

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
python scripts/build_data.py           # Normalize datasets to unified schema (toxic_abusive mode)
python scripts/build_sft.py            # Generate SFT instruction prompt/completion dataset
python scripts/train_demo_model.py     # Train TF-IDF word+char baseline (CPU, ~25s)
python scripts/run_eval.py             # Full Phase 4 evaluation: HateXplain, Fairness audit, HateCheck, OOD
python scripts/summarize_results.py    # Executive experimental summary table across all models
python scripts/release_gate.py         # CI/CD release gate verifying quality & fairness thresholds
```

### Deep Model Training on Google Colab (GPU)
For fine-tuning transformer encoders and LLMs on Tesla T4/A100 GPUs, use our turnkey notebook:
- Open [`notebooks/colab_training.ipynb`](notebooks/colab_training.ipynb) in Google Colab.
- Run Phase 1 to fine-tune `google/muril-base-cased` for Indian multilingual & Hinglish context.
- Run Phase 2 to fine-tune `Qwen/Qwen2.5-1.5B-Instruct` using PEFT LoRA with prompt-masked loss.
- Run Phase 3 to execute GRPO post-training with the composite reward function.
- Checkpoints save automatically to `checkpoints/` and results to `data/processed/results/`.

## Tests & Linting

```bash
pytest -q                              # 92 unit tests passing (100% test suite health)
ruff check .                           # Strict linting clean
```

## System Status & Benchmarks

| Component | Status | Benchmark / Metric |
|---|---|---|
| **TF-IDF (word+char) + LogReg** | Production Ready | Macro-F1: 0.810, AUC: 0.899 |
| **Google MuRIL Transformer** | Trained on GPU | Macro-F1: 0.767, AUC: 0.849, Eval Loss: 0.520 |
| **Qwen2.5-1.5B LoRA SFT LLM** | Trained & Integrated | Eval Loss: 0.1257 (Train Loss: 0.145) |
| **HateCheck Functional Suite** | Fully Audited | 3,728 functional cases evaluated (Acc: 0.360) |
| **Fairness & Bias Audit** | Fully Audited | Audited across 7 demographic groups (BPSN/BNSP) |
| **Interactive Gradio UI** | Live Serving | Dual-model selection, live attribution & faithfulness |
| **FastAPI REST Service** | Production Ready | Async endpoints, PII scrubbing, auditable decision logs |
| **Academic Paper** | Compiled PDF | `paper/main.pdf` (ACL 2024 camera-ready layout) |

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
