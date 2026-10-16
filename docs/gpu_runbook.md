# GPU Runbook — running Phases 1–3 on a Linux GPU box

The laptop is for code, data prep and CPU smoke tests. Full encoder training
(Phase 1), real LoRA SFT (Phase 2) and GRPO (Phase 3) need a CUDA GPU on
Linux (Triton kernels are Linux-only — see the note in
`src/pj03/sft/lora_sft.py`). Kaggle (free T4/P100, 30h/week) or Google Colab
both work; a lab machine is even better for SFT/GRPO.

All code lives in this repo — clone it, don't rewrite.

## 0. One-time setup (any GPU box)

```bash
git clone https://github.com/devantaris/vivek.git pj03 && cd pj03
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .[dev]
make test && make lint          # must pass before anything else
```

On Kaggle/Colab (no venv): `!git clone ... && !pip install -r requirements.txt`
from the notebook's first cell instead.

## 1. Data (CPU, seconds)

```bash
python scripts/download_data.py            # HateXplain + HateCheck (public)
python scripts/build_data.py               # -> data/processed/*
python scripts/build_sft.py                # -> Phase-2 SFT records
```

HASOC/Hinglish need manual downloads on a machine with a browser — do that
on the laptop, then `git`-less-copy `data/raw/hasoc/` onto the box (or zip +
upload; data/ is gitignored on purpose).

## 2. Phase 1 — encoder baselines (minutes to ~1h each)

```bash
python scripts/run_baseline.py --model encoder \
    baselines.encoders.smoke.enabled=false \
    baselines.encoders.models='[google/muril-base-cased]'
# then repeat with bert-base-multilingual-cased / xlm-roberta-base
python scripts/summarize_results.py
```

Smoke mode (`baselines.encoders.smoke.enabled=true`) is the CPU sanity check
— run it first on the box too; if it passes, the GPU run is config-only.

## 3. Phase 2 — LoRA SFT (hours)

```bash
python scripts/run_sft.py                    # config defaults: Qwen2.5-1.5B-Instruct
```

Notes:
- If the base model is gated (Llama/Gemma), set `HF_TOKEN` in the environment
  first (`.env.example` documents it) — requires a human-approved HF account.
- Outputs: adapter at `checkpoints/sft/`, metrics JSON in
  `data/processed/results/`. Commit the metrics JSON, never the checkpoint.

## 4. Phase 3 — GRPO with the composite reward (hours)

```bash
python scripts/run_reward.py                 # weights from configs/reward.yaml
python scripts/run_reward.py reward.weights.gamma=0.5   # ablate via overrides
```

The reward function is already validated against TRL's calling convention
(`tests/test_grpo_reward_contract.py` runs on CPU). `run_grpo` loads the
Phase-2 adapter, merges it, and optimises
`R = alpha*R_label + beta*R_faith - gamma*R_bias` with group-normalised
advantages (`num_generations=8`).

Budget note: `num_generations=8` means 8 generations per prompt per step —
start with `train.epochs=1` and a subset
(`run_reward.py train.epochs=1` on ~2k prompts) before full runs.

## 5. Bring results home

Only small files travel back:

```bash
git add data/processed/results/*.json docs/experiment_log.md
git commit -m "GPU run: <what>" && git push
```

Checkpoints stay on the box (or upload to the HF hub under the team account
if needed later). Append a row to `docs/experiment_log.md` for every run —
including failures.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Couldn't instantiate the backend tokenizer` | Model lacks `tokenizer.json` (e.g. prajjwal1/bert-tiny); use `google/bert_uncased_L-2_H-128_A-2` for smoke. |
| `Your setup doesn't support bf16` on CPU | Add/use the `use_cpu` flag path (already handled in both trainers). |
| TRL SFTTrainer crash `'NoneType' object has no attribute 'apply'` | Windows + trl>=1.15; we use the plain-Trainer SFT — don't switch back. |
| HF 429/500 during downloads | Unauthenticated rate limits; `export HF_TOKEN=<token>` or retry. |
