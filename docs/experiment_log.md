# Experiment Log — PJ-03

One entry per meaningful experiment/run. Keep it honest: negative results too.

| Date | Phase | What | Config/commit | Result | Notes |
|---|---|---|---|---|---|
| 2026-10-14 | setup | Repo scaffold created; core utilities implemented + smoke-tested | `20b0be3` | — | No training yet |
| 2026-10-14 | p1 | HateXplain downloaded (via parquet conversion) + build step + EDA stats | this session | train 15,383; 31% hateful | datasets>=3 can't load script datasets — use `refs/convert/parquet` |
| 2026-10-14 | p1 | TF-IDF+LogReg baseline on HateXplain test | default `baselines.yaml` | **macro-F1 0.799, AUC 0.891, acc 0.825** | results in `data/processed/results/tfidf_lr_hatexplain.json` |
| 2026-10-14 | p4 | HateCheck downloaded (3,728 cases) | — | — | eval-only, never train on it |

## Open decisions

- [ ] Binary label mapping for HateXplain (hatespeech=1; is offensive=1?)
- [ ] HASOC sublabel handling (HATE/OFFN/PRFN → binary)
- [ ] Hinglish dataset choice + licence
- [ ] Base LLM for Phase 2 (Qwen2.5-1.5B-Instruct vs Llama-3.2-1B vs Gemma-2-2b-it)
- [ ] GRPO vs DPO for Phase 3
- [ ] Identity-term lexicon for R_bias (needs supervisor review)
- [ ] GPU access plan for Phases 2–3
