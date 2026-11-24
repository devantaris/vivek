# Milestone-2 Defense & Demonstration Guide — Project PJ-03

**Course:** CSET-346 (Natural Language Processing)  
**Evaluation Dates:** 25th – 26th November 2026 (01:00 PM – 03:00 PM) | **Venue:** P2-CUB-096  
**Supervisor:** Dr. Pradeepika Verma  
**Team Members:** Devansh Kumar, Ayush, Saksham Tyagi, Mudit Agrawal  
**Repository:** `https://github.com/devantaris/vivek`  

---

## 1. Individual Member Responsibilities & Presentation Flow

When the evaluator asks: *"Explain each member's responsibilities and individual contributions"*, follow this coordinated 4-minute presentation sequence:

```
[Devansh Kumar]           [Ayush]               [Saksham Tyagi]          [Mudit Agrawal]
   (Minute 1)            (Minute 2)                (Minute 3)               (Minute 4)
System Architecture,   Data Engineering,       Adversarial Defense,     XAI Attributions,
Production Serving &    Baselines (TF-IDF &     HateCheck Stress Tests,  Faithfulness & Bias
Gradio Demo Lead        MuRIL) & PII Scrubber   De-Obfuscator Engine     Fairness Auditing
```

### Member 1: Devansh Kumar (Lead — Architecture, Production & Demo)
* **What you say:**
  > *"I led the overall project architecture, production serving pipeline, and interactive UI. I built our FastAPI microservice and Gradio interactive interface, implementing sub-60ms response latencies. I also structured our parameter-efficient fine-tuning on Qwen2.5-1.5B using LoRA (rank 16, alpha 32) so the model generates machine-parseable outputs containing both the classification label and natural language rationale. In production, I integrated our calibrated ensemble where deep LLM logits are blended with the fast linear model, backed by an active learning human-in-the-loop review queue."*
* **Files you own:** `src/pj03/service/api.py`, `src/pj03/demo/app.py`, `src/pj03/demo/predictor.py`, `src/pj03/sft/lora_sft.py`, `scripts/test_production_e2e.py`.

### Member 2: Ayush (Data Engineering, Encoders & Privacy)
* **What you say:**
  > *"I was responsible for our dataset pipeline, zero-leak privacy guard, and baseline modeling. I standardized the HateXplain corpus of over 19,000 examples into reproducible train/validation/test splits and ingested the HASOC code-mixed corpora. I implemented our PII scrubber, which automatically sanitizes emails, phone numbers, and handles into anonymous tokens before writing to audit logs. On the modeling side, I developed our linear word and character n-gram baseline (Macro-F1 0.810) and fine-tuned Google MuRIL Indic BERT on GPU with class-weighted cross-entropy loss and linear warmup."*
* **Files you own:** `src/pj03/preprocess/schema.py`, `src/pj03/preprocess/loaders_hatexplain.py`, `src/pj03/preprocess/pii.py`, `src/pj03/baselines/tfidf_lr.py`, `src/pj03/baselines/encoder_finetune.py`.

### Member 3: Saksham Tyagi (Adversarial Robustness & Hinglish Engine)
* **What you say:**
  > *"I focused on adversarial robustness, HateCheck functional evaluations, and solving the vocabulary collapse on Hinglish text. English subword tokenizers break Romanized Hindi slurs into meaningless fragments. I developed our phonetic canonicalization layer and multi-word threat detectors (`mar jana chahiye`, `goli maro`), boosting code-mixed accuracy from 55% to over 93%. Furthermore, to combat HateCheck evasion attacks like character swaps, leetspeak, and homoglyphs, I built our de-obfuscator engine (`deobfuscator.py`) which normalizes distorted text before tokenization, boosting our adversarial pass rate to 69.2%."*
* **Files you own:** `src/pj03/preprocess/hinglish.py`, `src/pj03/preprocess/deobfuscator.py`, `src/pj03/preprocess/adversarial.py`, `src/pj03/eval/hatecheck.py`, `tests/test_hinglish_threats.py`, `tests/test_deobfuscator_adversarial.py`.

### Member 4: Mudit Agrawal (Explainability, Faithfulness & Fairness Audit)
* **What you say:**
  > *"I led our Explainable AI (XAI) and demographic fairness auditing modules. Unlike black-box classifiers, our system provides mathematically verified attributions. I implemented Captum Integrated Gradients directly on the transformer embedding layer, calculating exact path integrals for token saliency, along with token-level KernelSHAP. To verify that explanations are causally grounded and not decorative, I built our ERASER faithfulness evaluation measuring Comprehensiveness and Sufficiency. Finally, I audited the model across 7 demographic groups using Borkan et al.'s Subgroup, BPSN, and BNSP AUC metrics to quantify and penalize unintended identity bias."*
* **Files you own:** `src/pj03/xai/attribution.py`, `src/pj03/xai/faithfulness.py`, `src/pj03/eval/fairness.py`, `src/pj03/eval/rationale_eval.py`, `src/pj03/reward/r_faith.py`, `src/pj03/reward/r_bias.py`.

---

## 2. Step-by-Step Live Demonstration Script

When the evaluator asks to **see the code run live**, execute these exact terminal commands:

### Step 1: Prove Testing & Code Quality (Takes 10 seconds)
Open a terminal in the project directory and run:
```bash
# 1. Run all 138 unit tests (100% pass rate)
pytest tests/ -q

# 2. Show strict linting compliance (0 errors)
ruff check src/ scripts/ tests/
```
*Expected output:* `138 passed in ~6s` and `All checks passed!`.

### Step 2: Run End-to-End Production Verification (Takes 5 seconds)
```bash
python scripts/test_production_e2e.py
```
*Expected output:*
```json
{
  "status": "ALL_TESTS_PASSED",
  "results": {
    "health_check": true,
    "english_accuracy": true,
    "hinglish_accuracy": true,
    "pii_scrubbing": true,
    "threshold_tuning": true,
    "moderation_queue_cycle": true,
    "latency_sub_100ms": true
  },
  "avg_latency_ms": 56.19
}
```

### Step 3: Launch Interactive Gradio Web Demo
```bash
python scripts/demo_app.py --port 7860
```
Open your browser to `http://localhost:7860`.

---

## 3. Live Demo Inputs to Show the Evaluator

Test these 4 distinct categories in the UI to prove breadth and depth:

### Case 1: Complex Hinglish Death Threat (Vocabulary Collapse Fix)
* **Input Text:** `modi ko mar jana chahliye`
* **Click:** *Classify & explain*
* **What the UI Shows:**
  - **Prediction:** `HATEFUL` ($P(\text{hateful}) = 0.920$)
  - **Highlighted Rationale:** Red highlight over `mar jana chahliye`
  - **Attributions:** `mar jana chahiye (+0.95)`, `modi (+0.38)`
* **What to tell the evaluator:**
  > *"Notice how the system correctly identifies the threat span `mar jana chahliye` even with non-standard transliterations. Notice also that the political named entity is NOT falsely flagged as the sole cause of hate; the causal weight is placed on the death incitement."*

### Case 2: Common Hinglish Abuse
* **Input Text:** `modi madharchod hai`
* **What the UI Shows:**
  - **Prediction:** `HATEFUL` ($P(\text{hateful}) = 0.920$)
  - **Highlighted Rationale:** Red highlight over `madharchod`

### Case 3: Adversarial Evasion & Leetspeak (HateCheck Defense)
* **Input Text:** `f@ggot trash`
* **What the UI Shows:**
  - **Prediction:** `HATEFUL` ($P(\text{hateful}) = 0.789$)
* **What to tell the evaluator:**
  > *"Attackers deliberately replace characters with numbers and symbols like `@` or `1`. Our de-obfuscation pre-filter decodes `f@ggot` $\rightarrow$ `faggot` before tokenization, preventing adversarial bypass."*

### Case 4: Benign Identity / Counter-Speech (False Positive Defense)
* **Input Text:** `modi visited the university today`
* **What the UI Shows:**
  - **Prediction:** `NON-HATEFUL` ($P(\text{hateful}) = 0.129$)
* **What to tell the evaluator:**
  > *"Notice that simply mentioning a political figure or demographic term does NOT trigger a false positive. The probability remains at 0.129, proving low demographic bias."*

### Case 5: Show the "Model Card & Fairness" Tab
* Click the second tab: **Model card & fairness**
* Show the live table comparing TF-IDF, MuRIL, and Qwen SFT, along with the **Borkan et al. Subgroup AUC and BPSN/BNSP fairness breakdown**.

---

## 4. Tough Questions Evaluators Ask & How to Answer Them

### Q1: "Why does your TF-IDF linear model have a higher Macro-F1 (0.810) than MuRIL (0.767)?"
* **Answer:**
  > *"That is a well-documented empirical phenomenon on HateXplain: hate speech often hinges on distinct discriminative n-grams and abusive keywords, which linear models capture aggressively with subword character n-grams. MuRIL is a 236M-parameter deep transformer that suffered from class imbalance under standard cross-entropy and conservative learning rates. In our latest upgrade, we implemented a class-weighted cross-entropy loss ($w_c = \frac{N}{2N_c}$) with linear warmup in `WeightedTrainer` (`src/pj03/baselines/encoder_finetune.py`), which balances precision and recall on Indic text."*

### Q2: "How do you prove your explanations are real and not just made up by the model?"
* **Answer:**
  > *"We use the ERASER benchmark faithfulness metrics: Comprehensiveness and Sufficiency (`src/pj03/xai/faithfulness.py`). Comprehensiveness measures the probability drop when we ablate the top-attributed tokens: if the explanation is causal, removing those tokens collapses the model's confidence. Sufficiency measures how much confidence is retained when keeping only those tokens. For deep models, we implemented Captum Integrated Gradients on the word embedding layer, ensuring mathematical adherence to the Axiom of Completeness."*

### Q3: "What is the difference between hatespeech and offensive language in your dataset?"
* **Answer:**
  > *"HateXplain annotators distinguish three classes: Hatespeech (direct attacks or incitement against protected demographic characteristics), Offensive (casual profanity, swearing, or vulgarity not targeting protected identities), and Normal. While binary models combine toxic categories, our Phase 6 roadmap implements a hierarchical 2-stage classifier: Stage 1 filters Benign vs Toxic, and Stage 2 disentangles illegal Incitement from casual offensive swearing to eliminate false positives in gaming or colloquial conversation."*

### Q4: "Where are user privacy and ethical concerns addressed?"
* **Answer:**
  > *"In our production pipeline (`src/pj03/preprocess/pii.py`), user text is scrubbed before any persistence. Emails, phone numbers, URLs, and `@handles` are converted to anonymous tokens at write time, ensuring compliance with DPDP and GDPR. Furthermore, our Gradio UI explicitly displays an ethics disclaimer stating that the system is an assistive tool for human moderators, not an autonomous decider."*
