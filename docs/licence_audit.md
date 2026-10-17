# Licence Audit — PJ-03

> **Purpose:** determine, before any commercial or deployment conversation,
> what we are legally allowed to do with each dataset and model in this repo.
> Status legend: ✅ verified · ⚠️ partially verified (check the source before
> commercial use) · ❌ blocking · ❓ unknown/needs human action.
>
> Last reviewed: 2026-10-17 (desktop research; the authoritative sources are
> the LICENSE files / dataset cards linked below — re-verify at release time).

## Summary table

| Asset | Role | Licence status | Commercial training allowed? | Redistribution allowed? |
|---|---|---|---|---|
| HateXplain | training data | ⚠️ code MIT; dataset terms = citation + platform ToS | **unclear — treat as no** | likely no (platform ToS) |
| HateCheck | eval-only data | ✅ CC BY 4.0 | yes, with attribution | yes, with attribution |
| HASOC | training data | ❌ not yet obtained | depends on registration terms | depends |
| Hinglish tweets | training data | ❓ dataset not chosen | — | — |
| Base LLMs (Qwen/Llama/Gemma) | Phase-2 model | ⚠️ per-model | Qwen Apache-2.0 ✅; Llama/Gemma community licences ❓ | per licence |
| Our code | repo | ✅ MIT (ours) | yes | yes |
| Trained checkpoints | ours | ✅ derivative of training data's terms | inherits input constraints | inherits |

## 1. HateXplain — the load-bearing risk

- **Sources:** [GitHub hate-alert/HateXplain](https://github.com/hate-alert/HateXplain)
  (code under MIT), [HF dataset card](https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain),
  paper: Mathew et al., AAAI 2021.
- **Status:** ⚠️ The GitHub *code* is MIT, but the *data* is drawn from Twitter
  and Gab posts. Standard terms: cite the paper, and **compliance with the
  source platforms' terms of service is the user's responsibility**. Twitter/X
  ToS restricts redistribution of content at scale.
- **Consequences for a product:**
  - Academic use (our paper): fine with citation.
  - Shipping a model *trained* on it commercially: grey zone — model weights
    are not the dataset, but a buyer's legal team may still object. Treat as
    "research-only model" until counsel says otherwise.
  - Redistributing the dataset itself (e.g., in a product download): **no**.
- **Mitigation paths:** (a) customer-supplied data for production models;
  (b) synthetic/generated data; (c) a licensed commercial dataset.
- **Action before any commercial use:** read the exact dataset terms on the
  HF card + repo README; record the acceptance date and terms here. ❌ pending

## 2. HateCheck — safe

- **Sources:** [HF Paul/hatecheck](https://huggingface.co/datasets/Paul/hatecheck);
  original distribution [paul-rottger/hatecheck-data](https://github.com/paul-rottger/hatecheck-data);
  Röttger et al., ACL 2021.
- **Status:** ✅ **CC BY 4.0** (the English test cases; multilingual variants
  are tagged the same on HF). Share/adapt/commercially use allowed **with
  attribution**.
- **Consequences:** we may use it as a release-gate suite even in a product,
  provided we credit Röttger et al. in the docs. We already never train on it.

## 3. HASOC — blocked on access

- Requires **registration** with the shared-task organisers; their terms apply.
- **Action:** register, download, then transcribe the accepted terms into this
  file *before* training anything on it. ❌ pending (human task)

## 4. Hinglish dataset — undecided

- Not chosen yet. **Action:** when choosing, require a licence that permits
  research use at minimum and record it here. Prefer CC BY / ODbL-style terms;
  avoid "request-only, non-commercial" if a product is plausible. ❓ open

## 5. Base LLMs (Phase 2)

| Model | Licence | Notes |
|---|---|---|
| Qwen2.5-1.5B-Instruct | ⚠️ verify (Apache-2.0 family) | current default; permissive |
| Llama-3.2-1B-Instruct | ❓ Llama Community License | gated; commercial OK <700M MAU but brand terms apply |
| Gemma-2-2B-it | ❓ Gemma Terms of Use | gated; permissive with restrictions |
- **Action:** confirm on the model card at download time; the HF token holder
  accepts the terms on the team's behalf — record who accepted and when.

## 6. Our own artefacts

- Code: MIT (LICENSE) — fine to build a product on.
- Trained checkpoints: carry the constraints of their training data (see
  HateXplain above). The demo/serve models are HateXplain-trained → same
  research-only caveat until the data question is resolved.
- Decision logs / feedback logs: contain scrubbed user text — govern them
  under the deployment's privacy policy, delete on request.

## Bottom line

The only dataset we hold today whose terms clearly permit commercial use is
**HateCheck (CC BY 4.0) — and it is evaluation-only by design**. Every
training corpus is research-constrained or not yet obtained. **The sellable
product path is therefore: architecture + workflow + eval suite (ours), with
models trained on customer/licensed data in deployment** — not shipping
models trained on scraped academic data.
