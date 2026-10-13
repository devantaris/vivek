"""PJ-03: Hate Speech Detection through Explainable AI and Reward-Optimized LLMs.

Bennett University NLP research project. Classifies English and Hinglish
(Hindi-English code-mixed) social media text as hateful / non-hateful and
outputs a rationale (the words or reason behind the decision).

Pipeline phases:
    1. Data prep + baselines (TF-IDF+LogReg, mBERT, XLM-R, MuRIL).
    2. LoRA SFT of a small instruction-tuned LLM (1-3B) -> label + rationale.
    3. Reward-optimized post-training (GRPO/DPO via TRL) with composite reward
       R = alpha*R_label + beta*R_faith - gamma*R_bias.
    4. XAI (SHAP / Integrated Gradients via Captum), faithfulness metrics,
       bias audit, HateCheck tests, out-of-distribution evaluation.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
