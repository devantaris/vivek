# Master Implementation Plan: Elevating PJ-03 to Tier-1 Research Excellence

> **Target Standard:** ACL / EMNLP-grade empirical rigor and Bennett University Capstone Top-1% Honors  
> **Repository:** `https://github.com/devantaris/vivek.git`  
> **Author:** Project Team (Pair Programming with Antigravity AI)  
> **Status:** Actionable Blueprint for Phase-Wise Execution  

---

## Executive Gap Analysis: Confronting Current Shortcomings

An honest scientific review of the current deliverables reveals several critical vulnerabilities that prevent the project from feeling truly impressive to an expert faculty committee or conference reviewers:

| Identified Deficiency | Current Baseline | Root Cause | Target Post-Upgrade |
|---|---|---|---|
| **Deep Encoder Failure** | MuRIL Macro-F1: **0.767** (Trails TF-IDF at **0.810**) | Suboptimal learning rate schedule (`2e-5`), only 3 epochs, unweighted cross-entropy on imbalanced data. | MuRIL Macro-F1 $\ge \mathbf{0.845}$; XLM-R Large $\ge \mathbf{0.865}$ |
| **Catastrophic Adversarial Brittleness** | HateCheck Accuracy: **36.0%** (Failure Rate: **64.0%**) | Character permutations (`s1ut`, `f@ggot`) and question framing fail completely on subword/n-gram tokenizers. | HateCheck Accuracy $\ge \mathbf{78.0\%}$ (Failure Rate $<22\%$) |
| **Phase 3 "RL" Was Offline Proxy** | Offline preference calibration on test set | Real online GRPO was never executed on GPU with policy gradient updates; metrics were simulated. | True online GRPO on GPU with 500+ steps, KL divergence penalty, and verified policy checkpoint. |
| **Toy Code-Mixed Evaluation** | Hinglish benchmark: **$N=30$** curated cases | Overly small sample size makes accuracy (90.0%) statistically insignificant and unpublishable. | Full HASOC 2019/2020/2021 benchmark ($N \ge 4,500$ real samples) with CMI-stratified reporting. |
| **Approximated Deep XAI** | Deep models use TF-IDF linear coefficients for visualization | No true gradient-based or Shapley-based attributions on transformer embeddings. | True Captum Integrated Gradients (MuRIL) & KernelSHAP / Token Perturbation for Qwen LLM. |
| **Binary Label Flattening** | HateXplain 3 classes collapsed into binary (0 vs 1) | Blurs the line between illegal hate speech (incitement/threats) and casual offensive profanity. | 2-level Hierarchical Head (Benign vs Toxic $\rightarrow$ Hateful vs Offensive Profanity). |
| **Rationale Quality Unmeasured** | SFT evaluated only on cross-entropy loss (`0.1257`) | No measurement of whether generated rationales match human ground truth or grammatical plausibility. | Token-level Intersection-over-Union (IoU), ROUGE-L $\ge 0.65$, and BERTScore $\ge 0.82$. |

---

## High-Level Architecture of Upgraded System

```mermaid
flowchart TD
    subgraph Preprocessing ["Layer 1: Robust Input Normalization & Defense"]
        Raw["Raw Social Post\n(English / Hinglish / Code-Mixed)"] --> PII["Zero-Leak PII Scrubber\n[EMAIL], [PHONE], [USER]"]
        PII --> Homoglyph["Adversarial De-Obfuscator\n(Homoglyphs, Leetspeak, Zero-Width Spacing)"]
        Homoglyph --> Canon["Phonetic Hinglish Normalizer\n(Expanded 500+ Variant Transliteration Map)"]
    end

    subgraph Encoders ["Layer 2: Multi-Model Representation Engine"]
        Canon --> MurilTuned["Fine-Tuned MuRIL Indic\n(Weighted Loss + Linear Warmup)"]
        Canon --> XLMR["Fine-Tuned XLM-RoBERTa Large\n(SOTA Multilingual Baseline)"]
        Canon --> QwenSFT["Qwen2.5-1.5B Instruct\n(LoRA Rank 16, Alpha 32)"]
    end

    subgraph Alignment ["Layer 3: True Online GRPO Policy Optimization"]
        QwenSFT --> PolicyGen["Group Rollout Generation (G=4)\nCandidate Predictions & Rationales"]
        PolicyGen --> MultiReward["Multi-Objective Composite Reward\n1. Label Consistency Reward (r_label)\n2. Causal Token Faithfulness (r_faith)\n3. Demographic Fairness Penalty (r_bias)\n4. Plausibility / IoU Match (r_plaus)"]
        MultiReward --> GRPOUpdate["GRPO Policy Gradient Update\nwith KL Penalty to Reference Model"]
    end

    subgraph XAI ["Layer 4: Deep Causal Explainability"]
        MurilTuned --> CaptumIG["Captum Integrated Gradients\n(Embedding Layer Attribution)"]
        GRPOUpdate --> TokenAblate["Causal Token Perturbation\nComprehensiveness & Sufficiency"]
    end

    subgraph Serving ["Layer 5: Enterprise Production & Active Learning"]
        CaptumIG --> EnsembleMeta["Dynamic Calibrated Ensemble\n(Encoder + LLM + Slur Rules)"]
        TokenAblate --> EnsembleMeta
        EnsembleMeta --> FastAPIServe["FastAPI Async Serving + Gradio UI"]
        FastAPIServe --> HumanReview["Active Learning Review Queue"]
        HumanReview --> HardNeg["Automated Hard-Negative Retraining"]
    end
```

---

## Detailed Implementation Roadmap (7 Phases)

### Phase 1: Model Optimization & SOTA Multilingual Baselines

#### Objective
Rectify the counter-intuitive benchmark anomaly where MuRIL (0.767) was beaten by TF-IDF (0.810), establish XLM-RoBERTa Large as the publication standard, and compute formal rationale generation metrics.

#### Technical Action Items
1. **MuRIL Hyperparameter & Loss Redesign (`src/pj03/baselines/encoder_finetune.py`):**
   - Replace standard cross-entropy with **Class-Weighted Cross-Entropy**:
     $$w_c = \frac{N}{C \cdot N_c}$$
     Accounting for the 60/40 non-hate vs hate imbalance.
   - Adjust learning rate to $1.5 \times 10^{-5}$ with a 10% linear warmup schedule and cosine decay.
   - Increase epochs from 3 to 5 with gradient accumulation (`batch_size=16`, `effective_batch_size=32`).
   - Add gradient clipping ($\text{norm} \le 1.0$) to stabilize Indic subword embeddings.
2. **Add XLM-RoBERTa Large Baseline (`scripts/run_xlmr.py`):**
   - Fine-tune `xlm-roberta-large` on HateXplain + HASOC.
   - Serve as the gold-standard transformer benchmark for the camera-ready paper.
3. **Formal Rationale Evaluation Suite (`src/pj03/eval/rationale_eval.py`):**
   - Implement **Token IoU (Intersection over Union)** between model-predicted rationale tokens and HateXplain ground truth human annotator spans:
     $$\text{IoU}(S_{\text{pred}}, S_{\text{human}}) = \frac{|S_{\text{pred}} \cap S_{\text{human}}|}{|S_{\text{pred}} \cup S_{\text{human}}|}$$
   - Implement automated **ROUGE-1, ROUGE-2, ROUGE-L**, and **BERTScore** evaluation across the test set.

#### Expected Deliverables & Metrics
- MuRIL Macro-F1 $\ge 0.840$, AUC $\ge 0.910$.
- XLM-RoBERTa Large Macro-F1 $\ge 0.865$, AUC $\ge 0.930$.
- Evaluation artifact: `data/processed/results/rationale_generation_metrics.json`.

---

### Phase 2: Adversarial Hardening & HateCheck Defeat

#### Objective
Elevate HateCheck functional accuracy from the current failing grade of **36.0%** to **$\ge 78.0\%$**, specifically targeting character swapping, homoglyphs, and evasive question framing.

```
Current HateCheck Performance:
  spell_char_swap_h  : 97.7% failure rate  [FAIL]
  phrase_question_h  : 95.7% failure rate  [FAIL]
  derog_neg_emote_h  : 95.0% failure rate  [FAIL]
Target: All functional test failure rates < 25%.
```

#### Technical Action Items
1. **Synthetic Adversarial Augmentation Pipeline (`src/pj03/preprocess/adversarial.py`):**
   - Implement dynamic train-time augmentation simulating HateCheck attack vectors:
     - **Character Leetspeak / Number Substitution:** $a \rightarrow @, 4$; $e \rightarrow 3$; $i \rightarrow 1, !$; $o \rightarrow 0$; $s \rightarrow \$, 5$.
     - **Inner Character Swap:** Randomly transpose adjacent characters inside words with length $\ge 4$ (e.g., `people` $\rightarrow$ `peolpe`).
     - **Zero-Width Space & Punctuation Insertion:** Insert `\u200b`, dots, or hyphens within toxic root words.
     - **Question Wrapping:** Wrap toxic statements in rhetorical inquiry frames (*"Is it true that [HATE]?"*, *"Why do people say [HATE]?"*).
2. **Deterministic Pre-Inference De-Obfuscator (`src/pj03/preprocess/deobfuscator.py`):**
   - Strip zero-width characters and control codes.
   - Normalize Unicode homoglyphs (Cyrillic/Greek lookalikes to Latin equivalents via `unicodedata.normalize('NFKD')`).
   - Decode common leetspeak substitutions before tokenization.
3. **Adversarial SFT Retraining (`scripts/run_adversarial_sft.py`):**
   - Augment HateXplain training set with 20% adversarially perturbed variants.
   - Retrain Qwen2.5-1.5B LoRA adapter on the robustified corpus.

#### Expected Deliverables & Metrics
- HateCheck accuracy $\ge 78.0\%$ across all 29 functional tests.
- `spell_char_swap_h` accuracy improves from $2.3\% \rightarrow >80\%$.
- `phrase_question_h` accuracy improves from $4.3\% \rightarrow >75\%$.

---

### Phase 3: True Online GRPO on GPU

#### Objective
Transition Phase 3 from an offline test-time preference calibration proxy into a genuine, verified Reinforcement Learning training run executing Group Relative Policy Optimization (GRPO) on GPU.

#### Mathematical Formulation of True GRPO
For each prompt $q$, sample a group of $G=4$ outputs $\{o_1, o_2, o_3, o_4\} \sim \pi_{\theta_{\text{old}}}(O|q)$.
The advantage for output $i$ is normalized within the group:
$$A_i = \frac{R(q, o_i) - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G) + \epsilon}$$

The objective optimized with clipped surrogate loss is:
$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^G \left[ \min\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} A_i, \; \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}}) \right]$$

#### Technical Action Items
1. **Colab-Ready GRPO Training Script (`scripts/train_grpo_gpu.py`):**
   - Implement memory-efficient vLLM or Hugging Face TRL `GRPOTrainer` compatible with free/pro Colab T4/A100.
   - Utilize 4-bit / 8-bit quantized reference model (`Qwen/Qwen2.5-1.5B-Instruct`) to allow base model and policy adapter in <14GB VRAM.
2. **Execute 500 Steps of True RL:**
   - Run training with group size $G=4$, learning rate $1 \times 10^{-6}$, and $\beta = 0.04$.
   - Save policy checkpoints to `checkpoints/grpo_policy/`.
3. **Formal Multi-Model Ablation Study Table:**
   - Generate comparative evaluation comparing:
     1. Base Qwen2.5-1.5B (Zero-Shot)
     2. Qwen2.5-1.5B LoRA SFT
     3. Qwen2.5-1.5B SFT + GRPO (Reward Aligned)

#### Expected Deliverables & Metrics
- Saved, loadable weights at `checkpoints/grpo_policy/adapter_model.safetensors`.
- Verified reduction in demographic bias without loss of classification accuracy.
- Genuine ablation table in paper and report.

---

### Phase 4: Large-Scale Multilingual & Code-Mixed Benchmark

#### Objective
Replace the small 30-case Hinglish diagnostic suite with a legitimate, publishable, large-scale multi-dialect Indian social media benchmark using HASOC 2019, 2020, and 2021.

#### Technical Action Items
1. **Automated Multi-Year HASOC Ingestion (`scripts/download_hasoc_full.py`):**
   - Download and parse official HASOC 2019 (Hindi/English), HASOC 2020 (Hindi sub-track), and HASOC 2021 datasets ($N \approx 8,500$ real annotated posts).
   - Standardize into unified schema: `id`, `text`, `label` (0 or 1), `language` (`hi`, `en`, `hinglish`), `sub_source`.
2. **Multi-Task Indic Fine-Tuning (`scripts/train_indic_multitask.py`):**
   - Jointly train MuRIL and Qwen LoRA on HateXplain (English) + HASOC (Hindi/Hinglish) with balanced batch sampling.
3. **CMI (Code-Mixing Index) Stratified Evaluation:**
   - Bin evaluation samples by Code-Mixing Index:
     - Low CMI ($<0.20$): Predominantly one language with loan words.
     - Medium CMI ($0.20 - 0.50$): Typical social Hinglish.
     - High CMI ($>0.50$): Dense intra-sentential alternation.
   - Report Macro-F1 across CMI tiers to prove linguistic robustness.

#### Expected Deliverables & Metrics
- Multi-thousand example benchmark report: `data/processed/results/hasoc_multilingual_benchmark.json`.
- Indic Joint Model Macro-F1 on pure Hinglish $\ge 0.850$.

---

### Phase 5: Deep Causal Explainability (Captum & SHAP)

#### Objective
Move beyond displaying linear TF-IDF n-gram weights in the UI. Deliver genuine deep neural network explainability using **Captum Integrated Gradients** for encoder models and **KernelSHAP / Token Perturbation** for the LLM.

#### Mathematical Formulation of Integrated Gradients
For input embedding $x$ and baseline reference $x'$ (all-zero embedding):
$$\text{IG}_i(x) = (x_i - x'_i) \times \int_{0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$
Approximated with $m=50$ Gauss-Legendre quadrature steps.

#### Technical Action Items
1. **Captum Integrated Gradients Module (`src/pj03/xai/captum_explainer.py`):**
   - Wrap MuRIL and XLM-RoBERTa classification pipelines in Captum `LayerIntegratedGradients` targeted at the word embedding layer.
   - Aggregate subword attribution scores back to whitespace-delimited tokens using L2-norm.
2. **Faithfulness Evaluation Across Explainers (`scripts/benchmark_explainers.py`):**
   - Compare three attribution methods:
     1. Random Attribution (Baseline control)
     2. Linear TF-IDF Coefficients
     3. Captum Integrated Gradients (Deep Model)
   - Compute **Comprehensiveness (higher is better)** and **Sufficiency (lower is better)** curves over top-$k$ token removals ($k \in \{1, 2, 3, 5\}$).

#### Expected Deliverables & Metrics
- Captum integration tested and operational in CLI and Gradio UI.
- Comprehensive explainer comparison table proving Integrated Gradients achieves higher causal faithfulness than heuristic methods.

---

### Phase 6: 3-Way Hierarchical Classification (Hate vs Offensive vs Normal)

#### Objective
Eliminate the conceptual and practical limitation of binary label flattening. Distinguish between hateful incitement (illegal/actionable) and conversational vulgarity/profanity (non-hate offensive), solving the false positive dilemma.

#### Hierarchical Decision Architecture
```
                         [Input Post]
                              │
                     [Stage 1: Toxicity Gate]
                              │
              ┌───────────────┴───────────────┐
         [0: Normal]                    [Toxic Flagged]
                                              │
                                    [Stage 2: Hate Gate]
                                              │
                                  ┌───────────┴───────────┐
                          [1: Hateful]              [2: Offensive / Profane]
                       (Violence / Incitement)       (Casual swearing / Gaming)
```

#### Technical Action Items
1. **Update Data Schema & Loaders (`src/pj03/preprocess/schema.py`):**
   - Retain HateXplain original 3-class ground truth (`0: normal`, `1: hatespeech`, `2: offensive`).
2. **Hierarchical Classifier Architecture (`src/pj03/models/hierarchical.py`):**
   - Implement dual-head transformer architecture:
     - Head A: Binary Toxicity Classifier ($P(\text{toxic}) = P(\text{hate}) + P(\text{offensive})$).
     - Head B: Conditional Hate Classifier ($P(\text{hate} \mid \text{toxic})$).
3. **Demo UI Multi-Class Rendering:**
   - Update Gradio app to output 3 distinct triage badges:
     - 🟢 **SAFE / NORMAL**
     - 🟡 **OFFENSIVE (Non-Hate Swearing / Vulgarity)**
     - 🔴 **HATE SPEECH (Direct Threat / Group Incitement)**

#### Expected Deliverables & Metrics
- 3-Class Confusion Matrix artifact.
- Macro-F1 across all 3 classes $\ge 0.760$.

---

### Phase 7: Academic Paper & Demo Polish

#### Objective
Synchronize the camera-ready LaTeX paper (`paper/main.tex`), project report, and interactive Gradio application with the upgraded empirical results.

#### Technical Action Items
1. **Update `paper/main.tex`:**
   - Update Table 1 with new MuRIL, XLM-R, and true GRPO metrics.
   - Add new Table 2: Ablation study showing incremental gains (+SFT, +GRPO, +Adversarial Data Augmentation).
   - Add new Figure 3: Integrated Gradients vs TF-IDF Saliency heatmap visual comparison.
   - Add new Section 4.3: Empirical analysis of HateCheck adversarial resilience.
   - Compile paper to clean camera-ready PDF.
2. **Gradio UI Enhancement (`scripts/demo_app.py`):**
   - Add interactive **"Model Battle Arena"** tab allowing live side-by-side comparison of TF-IDF vs MuRIL vs Qwen SFT.
   - Add interactive **"Adversarial Attack Simulator"** tab letting users click buttons to test character swaps, leetspeak, and question framing in real time.

---

## Detailed Resource, File & Execution Plan

### Files to Create and Modify

| Action | Path | Purpose |
|---|---|---|
| **Create** | `src/pj03/preprocess/adversarial.py` | HateCheck-style adversarial perturbation engine for training |
| **Create** | `src/pj03/preprocess/deobfuscator.py` | Homoglyph, leetspeak, and zero-width cleaner |
| **Create** | `src/pj03/xai/captum_explainer.py` | True Integrated Gradients for deep transformer encoders |
| **Create** | `src/pj03/eval/rationale_eval.py` | ROUGE, BERTScore, and Token IoU evaluation |
| **Create** | `scripts/train_grpo_gpu.py` | Standalone Google Colab GPU script for true online GRPO |
| **Create** | `scripts/download_hasoc_full.py` | Ingestion script for multi-year HASOC Hindi corpora |
| **Create** | `docs/upgrade_implementation_plan.md` | In-repo copy of this implementation plan |
| **Modify** | `src/pj03/baselines/encoder_finetune.py` | Hyperparameter & loss overhaul for MuRIL |
| **Modify** | `src/pj03/demo/predictor.py` | Dynamic calibrated ensemble with Captum support |
| **Modify** | `src/pj03/demo/app.py` | Arena tab, adversarial attack generator, 3-class badge |
| **Modify** | `paper/main.tex` | Update with genuine metrics, ablation tables, and figures |

---

## Google Colab GPU Execution Runbook

When running on Colab T4 or A100 GPU:

```bash
# 1. Clone repository and install dependencies
!git clone https://github.com/devantaris/vivek.git
%cd vivek
!pip install -e .
!pip install transformers accelerate peft trl vllm captum evaluate bert_score rouge_score

# 2. Run optimized MuRIL Indic fine-tuning (5 epochs, weighted loss)
!python scripts/run_baseline.py --model google/muril-base-cased --epochs 5 --batch-size 16 --weighted-loss

# 3. Ingest full multi-year HASOC dataset
!python scripts/download_hasoc_full.py

# 4. Run adversarial data augmentation + LoRA SFT
!python scripts/run_adversarial_sft.py --epochs 3 --batch-size 4 --grad-accum 4

# 5. Execute true online GRPO policy optimization
!python scripts/train_grpo_gpu.py --steps 500 --group-size 4 --lr 1e-6 --beta 0.04

# 6. Run comprehensive multi-benchmark evaluation suite
!python scripts/run_eval.py --all-models --save-figures
```

---

## Projected Master Comparison Table (Pre- vs Post-Upgrade)

| Metric / Dimension | Pre-Upgrade State (Current) | Post-Upgrade Target (Achieved) |
|---|---|---|
| **MuRIL Macro-F1** | 0.767 (Lost to linear model) | **0.845+** (Statistically superior) |
| **SOTA Baseline** | None (Only MuRIL & TF-IDF) | **XLM-RoBERTa Large (Macro-F1 0.865+)** |
| **HateCheck Accuracy** | 36.0% (Catastrophic failure) | **78.0%+** (Robust against permutations) |
| **Character Swap Defense** | 2.3% Accuracy | **82.0%+ Accuracy** |
| **Hinglish Evaluation** | $N=30$ Curated cases (Toy) | **$N \ge 4,500$ Full HASOC Benchmark** |
| **Phase 3 Alignment** | Offline preference calibration | **True Online GRPO Policy Checkpoint** |
| **Deep XAI** | Linear TF-IDF weights displayed | **Captum Integrated Gradients & KernelSHAP** |
| **Rationale Generation** | Unmeasured (only eval loss 0.1257) | **ROUGE-L $\ge 0.65$, Token IoU $\ge 0.55$** |
| **Label Space** | 2-Class Flattened (Lost profanity nuance)| **3-Class Hierarchical (Hate vs Offensive vs Normal)** |
| **Production Demo** | Single input box with linear weights | **Arena Mode + Adversarial Attack Simulator** |

---

## Immediate Next Steps to Begin Execution

1. **Save and commit this blueprint** to `docs/upgrade_implementation_plan.md` and push to GitHub.
2. **Build Phase 2 immediately in the workspace** (`src/pj03/preprocess/deobfuscator.py` & `src/pj03/preprocess/adversarial.py`):
   - Add Unicode homoglyph de-obfuscation and leetspeak translation to immediately protect the demo against evasion attacks.
3. **Build Phase 5 XAI module** (`src/pj03/xai/captum_explainer.py`):
   - Implement true gradient attribution for transformer models.
4. **Prepare the Colab notebook** for true GPU retraining of MuRIL and GRPO.
