# Dataset Notes — PJ-03

> Rule of the house: **verify every licence before use** and record it here.
> No invented download URLs — get links from the official sources at download
> time (see `scripts/download_data.py`).

## 1. HateXplain

- **Content:** English Twitter/Gab posts, 3-way annotation (hatespeech,
  offensive, normal) with majority vote over 3 annotators + rationale spans
  + target communities.
- **Access:** public on the Hugging Face hub (see `scripts/download_data.py`
  for the loader id used).
- **Licence:** TODO — check the dataset card at download time; record here.
- **Planned mapping:** hatespeech → 1, offensive/normal → 0 (DECISION OPEN —
  document the final choice in `docs/experiment_log.md`).
- **Used for:** Phase 1 training, Phase 2 SFT, in-domain evaluation.

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
- **Access:** public (HF hub id used in `scripts/download_data.py`).
- **Licence:** TODO — check at download time.
- **Used for:** Phase 4 functional testing only — **never train on it**.

## General notes

- All raw data goes under `data/raw/<dataset>/` (gitignored).
- All normalised data goes under `data/processed/<dataset>/{train,validation,test}.jsonl`
  in the unified schema (`src/pj03/preprocess/schema.py`).
- Never commit raw data, model checkpoints or `.env` to git.
