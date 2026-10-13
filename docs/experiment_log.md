# Experiment Log — PJ-03

One entry per meaningful experiment/run. Keep it honest: negative results too.

| Date | Phase | What | Config/commit | Result | Notes |
|---|---|---|---|---|---|
| 2026-10-14 | setup | Repo scaffold created; core utilities implemented + smoke-tested | initial commit | — | No training yet |

## Open decisions

- [ ] Binary label mapping for HateXplain (hatespeech=1; is offensive=1?)
- [ ] HASOC sublabel handling (HATE/OFFN/PRFN → binary)
- [ ] Hinglish dataset choice + licence
- [ ] Base LLM for Phase 2 (Qwen2.5-1.5B-Instruct vs Llama-3.2-1B vs Gemma-2-2b-it)
- [ ] GRPO vs DPO for Phase 3
- [ ] Identity-term lexicon for R_bias (needs supervisor review)
- [ ] GPU access plan for Phases 2–3
