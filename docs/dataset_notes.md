# Dataset Notes — PJ-03

> Rule of the house: **verify every licence before use** and record it here.
> No invented download URLs — get links from the official sources at download
> time (see `scripts/download_data.py`).

## 1. HateXplain

- **Content:** English Twitter/Gab posts, 3-way annotation (hatespeech,
  offensive, normal) with majority vote over 3 annotators + rationale spans
  + target communities.
- **Status:** DOWNLOADED (2026-10-14) via the hub's auto-converted parquet
  (`refs/convert/parquet` — the dataset repo is script-based, unsupported by
  `datasets>=3`). Splits: train 15,383 / val 1,922 / test 1,924.
- **Licence:** TODO — check the dataset card (`Hate-speech-CNERG/hatexplain`)
  before the paper's data release; record the exact terms here.
- **Mapping decisions** (implemented in `loaders_hatexplain.py`):
  - hatespeech → 1; normal/offensive → 0 (`_to_binary`).
  - 3-way annotator ties resolve toward the more harmful class
    (hatespeech > offensive > normal) — documented in `_majority`.
  - `rationale_mask` = tokens flagged by ≥2 of 3 annotators.
  - `target_group` = communities named by ≥ half the annotators, '|'-joined.
- **Processed stats (train):** 4,748 hateful / 10,635 non-hateful (~31%);
  rationale masks present for 9,132 posts; top targets: african, islam,
  jewish, homosexual, women.

## 2. HASOC (Hindi / English / code-mixed tracks)

- **Content:** shared-task data on hate/offensive speech, several years
  (2019–2021+), tracks for Hindi, English, and code-mixed (Hi-En).
- **Access:** **manual download after registration** with the organisers —
  the shared task distributes data under its own terms. Do not redistribute.
- **Licence:** TODO — record the exact terms accepted at registration.
- **Planned mapping:** HOF → 1, NOT → 0; sublabel handling (HATE/OFFN/PRFN)
  to be decided (DECISION OPEN).
- **Used for:** Phase 1 training, cross-lingual OOD evaluation (Phase 4).

## 3. Hinglish offensive-tweet dataset

- **Status:** NOT CHOSEN YET (team decision). Candidates all require a
  licence check; some are gated on the HF hub (access request) or come from
  papers with request forms.
- **Used for:** Hinglish training/evaluation.
- **Decision log:** record the chosen dataset, version, access date and
  licence here once decided.

## 4. HateCheck

- **Content:** ~3,700 functional tests for hate-speech detection, covering
  29 functionalities and multiple target identities.
- **Status:** DOWNLOADED (2026-10-14, 3,728 test cases) from the HF hub
  (`Paul/hatecheck`).
- **Licence:** TODO — check the dataset card.
- **Used for:** Phase 4 functional testing only — **never train on it**.

## General notes

- All raw data goes under `data/raw/<dataset>/` (gitignored).
- All normalised data goes under `data/processed/<dataset>/{train,validation,test}.jsonl`
  in the unified schema (`src/pj03/preprocess/schema.py`).
- Never commit raw data, model checkpoints or `.env` to git.
