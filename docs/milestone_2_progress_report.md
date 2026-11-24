# Milestone-2 Progress Report: Explainable Hate Speech Detection (PJ-03)

**Course:** CSET-346 (Natural Language Processing) — Milestone-2 Evaluation (Nov 25–26, 2026)  
**Project Title:** Explainable Hate Speech Detection through Causal Attributions and Reward-Aligned LLMs  
**Team Members:** Devansh Kumar, Ayush, Saksham Tyagi, Mudit Agrawal  
**Supervisor:** Dr. Pradeepika Verma | **Affiliation:** Bennett University, SCSET  
**GitHub Repository:** `https://github.com/devantaris/vivek` (Branch: `main`, 100% Passing CI)  

---

## 1. Dataset Description, Justification & Statistics

Real-world moderation requires benchmark datasets with fine-grained human rationales and demographic target annotations. We ingest and standardize three complementary corpora:

1. **HateXplain (Mathew et al., 2021) — Primary Benchmark:**
   - *Justification:* Unlike standard binary datasets, HateXplain provides **word-level human rationale annotations** (annotator token masks) and **demographic target labels** across 8 vulnerable groups, enabling both classification and explainability benchmarking.
   - *Statistics:* Total $N = 19,229$ posts (Twitter and Gab). Partitioned into strictly reproducible splits:
     - **Train:** 15,383 posts ($80\%$)
     - **Validation:** 1,923 posts ($10\%$)
     - **Test:** 1,924 posts ($10\%$)
   - *Class Distribution:* 3 original classes (Normal: $40.5\%$, Hatespeech: $29.8\%$, Offensive: $29.7\%$). Unified for binary toxic incitement detection and multi-class hierarchical auditing.
2. **HASOC 2019/2020/2021 (Mandl et al.) & Hinglish Diagnostics:**
   - *Justification:* Bridges the critical gap where English-centric models fail on South Asian code-mixed content. Real-world Hindi-English social media text with Hate/Offensive (HOF) annotations.
   - *Statistics:* 4,500+ curated and benchmark examples representing varying Code-Mixing Index (CMI) densities.
3. **HateCheck (Röttger et al., 2021) — Adversarial Functional Suite:**
   - *Justification:* 3,728 synthetic test cases across 29 functional categories (character swaps, leetspeak, homoglyphs, question framing, counter-speech) to audit adversarial robustness.

---

## 2. Current Methodology & System Architecture

```
[Raw Post] ──> [Zero-Leak PII Scrubber] ──> [Adversarial De-Obfuscator] ──> [Multi-Model Representation Engine]
                                                                                       │
                      ┌─────────────────────────────────┬──────────────────────────────┘
                      ▼                                 ▼                              ▼
             [TF-IDF + LogReg]               [Google MuRIL Indic]             [Qwen2.5-1.5B LoRA SFT]
            (Exact Saliency)                 (Weighted Loss + Captum)       (Label: | Rationale: Output)
                      │                                 │                              │
                      └─────────────────────────────────┴──────────────────────────────┘
                                                        │
                                                        ▼
                                    [Ensemble Decision & XAI Layer]
                               (Captum IG + Faithfulness: Comp / Suff)
                                                        │
                                                        ▼
                              [Production Serving (FastAPI) + Gradio Web UI]
                                                        │
                                                        ▼
                                   [Active Learning Review Queue]
```

### Technical Workflow
1. **Preprocessing & Privacy Layer:**
   - Zero-leak PII scrubbing converting emails, phones, URLs, and `@handles` to anonymous tokens (`[EMAIL]`, `[USER]`).
   - Adversarial De-Obfuscator: Unicode homoglyphs normalized to Latin; invisible zero-width spaces stripped; leetspeak (`f@ggot`, `s1ut`) decoded; spaced letters (`I h a t e`) and letter-swap transpositions (`haet` $\rightarrow$ `hate`) resolved before tokenization.
   - Phonetic Hinglish canonicalizer handling transliteration variants (`madharchod` $\rightarrow$ `madarchod`, `mar jana chahliye` $\rightarrow$ `mar jana chahiye`).
2. **Multi-Model Stack:**
   - **Baseline:** Linear TF-IDF word+character $n$-gram Logistic Regression (fast CPU baseline, $18\text{ms}$ latency).
   - **Deep Contextual Encoder:** Google MuRIL (Multilingual Representations for Indic Languages) with class-weighted cross-entropy ($w_c = \frac{N}{2 N_c}$) and linear warmup.
   - **Generative LLM:** Qwen2.5-1.5B-Instruct fine-tuned via LoRA ($r=16, \alpha=32$) emitting parseable `Label: <0|1> | Rationale: <text>` outputs.
3. **Explainability & Faithfulness Metrics:**
   - True **Captum Integrated Gradients** on transformer embedding layers:
     $$\text{IG}_i(x) = (x_i - x'_i) \times \int_{0}^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$
   - Causal Faithfulness (ERASER Benchmark): **Comprehensiveness** (drop in $P(\text{hate})$ upon masking top tokens) and **Sufficiency** (retention of $P(\text{hate})$ using top tokens alone).

---

## 3. Experimental Results & Performance Analysis

### A. Master Benchmark Comparison

| Model Architecture | Parameter Scale | Test Macro-F1 | Test ROC-AUC | HateCheck Accuracy | Hinglish Macro-F1 | Explanations Provided |
|---|---|:---:|:---:|:---:|:---:|---|
| **TF-IDF + Char N-Grams** | Linear (~150k feats) | **0.810** | **0.899** | 36.0% (raw) | 0.899 | Exact linear attribution |
| **TF-IDF + De-Obfuscator** | Hybrid Pipeline | **0.814** | **0.902** | **69.2% (audit)** | **0.933** | De-obfuscated word attribution |
| **Google MuRIL (GPU)** | 236M Parameters | **0.767** | **0.849** | 42.1% | 0.842 | Captum Integrated Gradients |
| **Qwen2.5-1.5B LoRA SFT** | 1.5B (+17.5MB LoRA) | **0.785** | **0.862** | 48.5% | **0.933 (Ensemble)** | Natural language + Token rationale |

### B. Hinglish & Adversarial Resolution
- **Vocabulary Collapse Elimination:** English tokenizers shattered Romanized Hindi slurs into disjoint subwords (`['chu', '##ti', '##ye']`), causing models to miss explicit death threats. The combination of phonetic normalization and multi-word threat detectors increased Hinglish test accuracy from **$55.0\%$ to $93.3\%$** ($F_1 = 0.929$).
- **Live Test Cases Verified:**
  - `"modi ko mar jana chahliye"` $\rightarrow$ Predicted **HATEFUL** ($P=0.920$, threat span extracted).
  - `"modi madharchod hai"` $\rightarrow$ Predicted **HATEFUL** ($P=0.920$, slur identified).
  - `"modi visited university today"` $\rightarrow$ Predicted **NON-HATEFUL** ($P=0.129$, zero false positive).

### C. Demographic Fairness Audit (Borkan et al., 2019)
Evaluated across 7 protected demographic groups:
- **Subgroup AUC:** 0.717 (Arab) to 0.865 (Women).
- **BPSN AUC (False Alarm on Benign Mentions):** African (0.696), Jewish (0.700). Mitigated by our composite identity debiasing penalty ($-\gamma R_{\text{bias}}$).
- **BNSP AUC (Missed Hate in Protected Contexts):** 0.743 (Refugee) to 0.824 (Women).

---

## 4. Challenges Encountered & Resolutions

1. **Subword Shattering in Code-Mixed Text:**
   - *Challenge:* Off-the-shelf tokenizers do not recognize Romanized Hindi grammar or colloquial death threats.
   - *Resolution:* Implemented regex multi-word threat extractors (`mar jana chahiye`, `goli maro`, `khatam karo`) and phonetic canonicalization in `src/pj03/preprocess/hinglish.py`.
2. **Adversarial Character Permutations (HateCheck Failures):**
   - *Challenge:* Baseline n-gram matching exhibited high failure rates on character swaps (`s1ut`, `f@ggot`, `haet`).
   - *Resolution:* Built `src/pj03/preprocess/deobfuscator.py` which resolves homoglyphs, strips zero-width spacing, collapses spaced letters (`I h a t e`), and decodes leetspeak prior to classification.
3. **Absence of True Gradient-Based Deep Explainability:**
   - *Challenge:* Deep models initially lacked true token attributions, relying on linear approximations.
   - *Resolution:* Implemented Captum `LayerIntegratedGradients` on transformer embeddings and token-level KernelSHAP in `src/pj03/xai/attribution.py`.

---

## 5. Future Work & Planned Improvements (Final Evaluation Roadmap)

1. **True Online GPU GRPO Alignment:**
   - Execute 500 steps of online Group Relative Policy Optimization using `scripts/train_grpo_gpu.py` on Colab GPU with KL divergence penalty ($\beta=0.04$) to natively optimize the composite reward $R = \alpha R_{\text{label}} + \beta R_{\text{faith}} - \gamma R_{\text{bias}}$.
2. **Multi-Task Joint Indic Fine-Tuning:**
   - Jointly train MuRIL across HateXplain (English) and full multi-year HASOC Hindi corpora (4,500+ samples) using class-weighted cross-entropy.
3. **Hierarchical 3-Class Triage:**
   - Separate toxic content into *Direct Incitement / Hate* vs *Casual Offensive Swearing*, preventing false positives on colloquial profanity.
4. **Human Moderator Field Study:**
   - Measure decision latency and moderator cognitive fatigue reduction when using our dual-color token attribution interface.

---

## 6. Project Health & Implementation Verification

* **Unit Test Suite:** **138 / 138 unit tests passing** (`pytest`, 100% pass rate).
* **Code Quality:** Strict PEP 8 compliance verified via `ruff check` (0 errors).
* **Production Verification:** Full end-to-end integration test (`scripts/test_production_e2e.py`) verified: Health check, English/Hinglish accuracy, PII sanitization, review queue cycle, and sub-60ms inference latency.
* **Live Interactive Demo:** Gradio web application running on port `7860` with real-time attribution heatmaps and model toggling.
