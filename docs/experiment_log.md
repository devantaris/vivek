# Experiment Log — PJ-03

One entry per meaningful experiment/run. Keep it honest: negative results too.

| Date | Phase | What | Config/commit | Result | Notes |
|---|---|---|---|---|---|
| 2026-10-14 | setup | Repo scaffold created; core utilities implemented + smoke-tested | `20b0be3` | — | No training yet |
| 2026-10-14 | p1 | HateXplain downloaded (via parquet conversion) + build step + EDA stats | this session | train 15,383; 31% hateful | datasets>=3 can't load script datasets — use `refs/convert/parquet` |
| 2026-10-14 | p1 | TF-IDF+LogReg baseline on HateXplain test | default `baselines.yaml` | **macro-F1 0.799, AUC 0.891, acc 0.825** | results in `data/processed/results/tfidf_lr_hatexplain.json` |
| 2026-10-14 | p4 | HateCheck downloaded (3,728 cases) | — | — | eval-only, never train on it |
| 2026-10-14 | p1/p4 | **Bug fix**: BPSN/BNSP were pooling all four sets (they came out as exact complements). Reimplemented to the Borkan pair construction; tests now pin hand-computed values. | this session | — | caught before any numbers were reported |
| 2026-10-14 | p1 | Per-group bias audit of TF-IDF baseline (test, corrected metrics) | `baselines.yaml` terms | subgroup AUC 0.71–0.83 (arab lowest); **BPSN low: jewish 0.65, african 0.70** (identity-mention false positives); **BNSP low: refugee 0.72, women 0.80** (missed in-group hate) | saved in `results/tfidf_lr_hatexplain.json`; R_bias in Phase 3 targets exactly this |
| 2026-10-14 | p1 | Encoder fine-tuning implemented (HF Trainer); CPU smoke validated | `smoke.enabled=true` | tiny-BERT 200-ex path runs end-to-end in ~1s | smoke validates plumbing only; full mBERT/XLM-R/MuRIL need GPU |
| 2026-10-14 | p2 | SFT dataset built from HateXplain rationales | `scripts/build_sft.py` | 15,383 train / 1,922 val records; 8,862 train with annotator rationales | completions round-trip through the reward parser (tested) |
| 2026-10-14 | p1 | HateCheck loader implemented (unified schema + `functionality` field) | `2adf604` | 3,728 cases, 29 functionalities, normalised targets | eval-only, all TEST split |
| 2026-10-14 | p3 | **R_faith implemented for real**: remove the rationale's content words from the post, reward = \|ΔP(hateful)\| | `2d1dfba` | faithful rationale ≈ 0.85 vs boilerplate = 0 on toy model | `post_text` threaded through composite/batch rewards |
| 2026-10-14 | p2 | **LoRA SFT implemented** with plain HF Trainer + completion-only masking | `a02a72b` | CPU smoke (tiny-gpt2, 64 recs): trains + saves adapter in ~3s | TRL 1.15 SFTTrainer crashes on Windows (Linux-only Triton kernel) — documented; TRL still planned for GRPO on Linux GPU |
| 2026-10-16 | tooling | `scripts/summarize_results.py` aggregates all result JSONs into one table | this session | 3 experiments tracked so far | — |
| 2026-10-14 | p3 | GRPO implemented (trl.GRPOTrainer + composite reward via TRL contract); contract validated on CPU with simulated trainer calls | `fa5e87b`, `2553957` | reward ranking correct: faithful>boilerplate>wrong-label | GPU-only; runbook written |
| 2026-10-16 | demo | **Demo product**: char_wb n-grams added → macro-F1 **0.8115** (word-only 0.799); joblib artifact + Gradio app (rationale highlights, live faithfulness, bias audit tab) | `9932c4a`, `8d53f45` | "kill all jews" 0.93 P, comp 0.93 | demo trains in ~30s CPU; `make demo` |
| 2026-10-16 | demo | 'i hate women' scores 0.37 = labelled *offensive* (not hatespeech) by HateXplain majority → maps to 0 | — | — | kept as the annotation-subjectivity talking point in the demo |
| 2026-10-17 | product | PII scrubber (email/url/mention/phone → placeholders); decision logs always scrubbed | `61ab65b` | 8 tests | raw user text never persisted |
| 2026-10-17 | product | FastAPI service: /predict (per-request threshold), /health, JSONL decision log, Dockerfile | `7a0246d` | 6 tests incl. PII-never-stored | `make serve`; docs at /docs |
| 2026-10-17 | product | Release gate (quality+fairness floors, `make gate`): PASS on current model (F1 0.810, worst subgroup 0.717, BPSN 0.696, BNSP 0.743); FAIL verified with impossible floors | `68eb5a1` | exit codes 0/1 verified both ways | CI hook for all future models |

## Open decisions

- [ ] Binary label mapping for HateXplain (hatespeech=1; is offensive=1?)
- [ ] HASOC sublabel handling (HATE/OFFN/PRFN → binary)
- [ ] Hinglish dataset choice + licence
- [ ] Base LLM for Phase 2 (Qwen2.5-1.5B-Instruct vs Llama-3.2-1B vs Gemma-2-2b-it)
- [ ] GRPO vs DPO for Phase 3
- [ ] Identity-term lexicon for R_bias (needs supervisor review)
- [ ] GPU access plan for Phases 2–3
