"""Generate a modern 16:9 PowerPoint presentation (docs/PJ03_Milestone2_Presentation.pptx).

Includes:
  - 3 custom high-DPI Matplotlib visualizations saved to artifacts/ppt_charts/
  - 12 professionally laid-out widescreen slides with zero overlapping elements
  - Speaker notes on every slide tailored for the CSET-346 Milestone-2 evaluation
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
CHARTS_DIR = REPO_ROOT / "artifacts" / "ppt_charts"

# ---------------------------------------------------------------------------
# Color Palette (Modern Academic / Executive Dark-Navy & Electric Blue)
# ---------------------------------------------------------------------------
NAVY_DARK = RGBColor(15, 23, 42)       # #0F172A
NAVY_MID = RGBColor(30, 41, 59)        # #1E293B
BLUE_ACCENT = RGBColor(37, 99, 235)    # #2563EB
BLUE_LIGHT = RGBColor(219, 234, 254)   # #DBEAFE
TEAL = RGBColor(13, 148, 136)          # #0D9488
TEAL_LIGHT = RGBColor(204, 251, 241)   # #CCFBF1
CORAL = RGBColor(225, 29, 72)          # #E11D48
CORAL_LIGHT = RGBColor(255, 228, 230)  # #FFE4E6
AMBER = RGBColor(217, 119, 6)          # #D97706
AMBER_LIGHT = RGBColor(254, 243, 199)  # #FEF3C7
BG_LIGHT = RGBColor(248, 250, 252)     # #F8FAFC
WHITE = RGBColor(255, 255, 255)
BORDER_GRAY = RGBColor(203, 213, 225)  # #CBD5E1
TEXT_MAIN = RGBColor(30, 41, 59)       # #1E293B
TEXT_MUTED = RGBColor(100, 116, 139)   # #64748B

FONT_MAIN = "Segoe UI"
FONT_CODE = "Consolas"


# ---------------------------------------------------------------------------
# Matplotlib Charts
# ---------------------------------------------------------------------------


def generate_charts() -> dict[str, Path]:
    """Generate clean, high-resolution charts for embedding into slides."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.linewidth"] = 0.8

    paths: dict[str, Path] = {}

    # 1. Model Performance Chart (Macro-F1 & ROC-AUC)
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=300)
    models = [
        "TF-IDF\n(word)",
        "TF-IDF\n(word+char)",
        "Google\nMuRIL",
        "Qwen2.5\nLoRA SFT",
        "Code-Mixed\nUnified",
    ]
    f1_scores = [0.799, 0.810, 0.767, 0.785, 0.933]
    auc_scores = [0.891, 0.899, 0.849, 0.862, 0.916]

    x = np.arange(len(models))
    width = 0.36
    bars1 = ax.bar(x - width / 2, f1_scores, width, label="Macro-F1", color="#2563EB", zorder=3)
    bars2 = ax.bar(x + width / 2, auc_scores, width, label="ROC-AUC", color="#0D9488", zorder=3)

    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Score", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_title("Model Benchmark Comparison (Test Splits)", fontsize=11, fontweight="bold", color="#0F172A", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, color="#1E293B")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=2)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1E3A8A")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#115E59")

    plt.tight_layout()
    p_models = CHARTS_DIR / "chart_models.png"
    fig.savefig(p_models, dpi=300, facecolor="white")
    plt.close(fig)
    paths["models"] = p_models

    # 2. Hinglish Recovery Chart
    fig, ax = plt.subplots(figsize=(5.8, 3.6), dpi=300)
    configs = [
        "Standard English\nBaseline",
        "PJ-03 Hinglish\nDiagnostic (N=20)",
        "Unified HASOC +\nHinglish (N=30)",
    ]
    acc_vals = [55.0, 90.0, 93.3]
    colors = ["#E11D48", "#2563EB", "#0D9488"]

    bars = ax.bar(configs, acc_vals, width=0.52, color=colors, zorder=3)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_title("Overcoming Hinglish Vocabulary Collapse", fontsize=11, fontweight="bold", color="#0F172A", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#0F172A")

    plt.tight_layout()
    p_hinglish = CHARTS_DIR / "chart_hinglish.png"
    fig.savefig(p_hinglish, dpi=300, facecolor="white")
    plt.close(fig)
    paths["hinglish"] = p_hinglish

    # 3. Fairness Audit (BPSN AUC across 7 groups)
    fig, ax = plt.subplots(figsize=(6.4, 4.1), dpi=300)
    groups = ["African", "Jewish", "Arab", "Islam", "Homosexual", "Women", "Refugee"]
    raw_bpsn = [0.696, 0.700, 0.717, 0.778, 0.868, 0.934, 0.941]
    cal_bpsn = [0.962, 0.970, 0.959, 0.971, 0.989, 0.998, 1.000]

    y = np.arange(len(groups))
    h = 0.36
    ax.barh(y - h / 2, raw_bpsn, h, label="Baseline Raw BPSN", color="#94A3B8", zorder=3)
    ax.barh(y + h / 2, cal_bpsn, h, label="Reward-Calibrated BPSN (Offline)", color="#2563EB", zorder=3)

    ax.set_xlim(0.5, 1.12)
    ax.set_xlabel("BPSN AUC (Higher = Fewer Identity False Alarms)", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.set_title("Demographic Fairness Audit Across 7 Protected Groups", fontsize=11, fontweight="bold", color="#0F172A", pad=26)
    ax.set_yticks(y)
    ax.set_yticklabels(groups, fontsize=9, fontweight="bold", color="#1E293B")
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=8.5)

    for idx, (r, c) in enumerate(zip(raw_bpsn, cal_bpsn, strict=False)):
        ax.text(c + 0.01, idx + h / 2, f"{c:.3f} (+{c - r:.2f})", va="center", fontsize=7.5, fontweight="bold", color="#1E3A8A")

    plt.tight_layout()
    p_fair = CHARTS_DIR / "chart_fairness.png"
    fig.savefig(p_fair, dpi=300, facecolor="white")
    plt.close(fig)
    paths["fairness"] = p_fair

    return paths


# ---------------------------------------------------------------------------
# Slide Building Primitives
# ---------------------------------------------------------------------------


def set_slide_bg(slide, color: RGBColor) -> None:
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_header(slide, title: str, subtitle: str = "", page_num: int = 1) -> None:
    """Standard clean light header with top navy strip and slide number."""
    set_slide_bg(slide, BG_LIGHT)

    # Top thin accent line
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.08))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE_ACCENT
    bar.line.fill.background()

    # Title
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.25), Inches(11.5), Inches(0.55))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title
    r.font.name = FONT_MAIN
    r.font.size = Pt(24)
    r.font.bold = True
    r.font.color.rgb = NAVY_DARK

    if subtitle:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = subtitle
        r2.font.name = FONT_MAIN
        r2.font.size = Pt(12)
        r2.font.color.rgb = TEXT_MUTED

    # Divider line
    div = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.02), Inches(12.133), Inches(0.02))
    div.fill.solid()
    div.fill.fore_color.rgb = BORDER_GRAY
    div.line.fill.background()

    # Footer
    ft = slide.shapes.add_textbox(Inches(0.6), Inches(7.12), Inches(9.5), Inches(0.3))
    fp = ft.text_frame.paragraphs[0]
    fr = fp.add_run()
    fr.text = "PJ-03  |  CSET-346 NLP Milestone-2 Evaluation  |  Bennett University (SCSET)"
    fr.font.name = FONT_MAIN
    fr.font.size = Pt(9)
    fr.font.color.rgb = TEXT_MUTED

    ptb = slide.shapes.add_textbox(Inches(11.5), Inches(7.12), Inches(1.2), Inches(0.3))
    pp = ptb.text_frame.paragraphs[0]
    pp.alignment = PP_ALIGN.RIGHT
    pr = pp.add_run()
    pr.text = f"{page_num} / 12"
    pr.font.name = FONT_MAIN
    pr.font.size = Pt(9)
    pr.font.bold = True
    pr.font.color.rgb = TEXT_MUTED


def add_card(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    title: str = "",
    accent_color: RGBColor = BLUE_ACCENT,
    bg_color: RGBColor = WHITE,
):
    """Draw a rounded card with a top colored accent border."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.adjustments[0] = 0.03
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = BORDER_GRAY
    card.line.width = Pt(1)

    # Top accent strip inside card
    strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(0.07))
    strip.fill.solid()
    strip.fill.fore_color.rgb = accent_color
    strip.line.fill.background()

    if title:
        tb = slide.shapes.add_textbox(Inches(left + 0.18), Inches(top + 0.15), Inches(width - 0.36), Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = title
        r.font.name = FONT_MAIN
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = NAVY_DARK
    return card


def add_bullets(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    items: list[tuple[str, str]],
    font_size: int = 11,
    space_after: int = 6,
) -> None:
    """Add formatted bold-prefix + description bullets."""
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    for idx, (prefix, body) in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        if prefix:
            r1 = p.add_run()
            r1.text = prefix + (" " if body else "")
            r1.font.name = FONT_MAIN
            r1.font.size = Pt(font_size)
            r1.font.bold = True
            r1.font.color.rgb = NAVY_DARK
        if body:
            r2 = p.add_run()
            r2.text = body
            r2.font.name = FONT_MAIN
            r2.font.size = Pt(font_size)
            r2.font.color.rgb = TEXT_MAIN


def add_pill(slide, left: float, top: float, width: float, height: float, text: str, bg: RGBColor, fg: RGBColor, font_size: int = 10) -> None:
    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    pill.adjustments[0] = 0.5
    pill.fill.solid()
    pill.fill.fore_color.rgb = bg
    pill.line.fill.background()
    tf = pill.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Pt(6)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.name = FONT_MAIN
    r.font.size = Pt(font_size)
    r.font.bold = True
    r.font.color.rgb = fg


def add_notes(slide, text: str) -> None:
    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = text


def style_table_cell(cell, text: str, bg: RGBColor, fg: RGBColor, bold: bool = False, size: int = 10, align=PP_ALIGN.LEFT) -> None:
    cell.fill.solid()
    cell.fill.fore_color.rgb = bg
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = Inches(0.1)
    cell.margin_right = Inches(0.1)
    cell.margin_top = Inches(0.05)
    cell.margin_bottom = Inches(0.05)
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.name = FONT_MAIN
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = fg


# ---------------------------------------------------------------------------
# Build 12 Slides
# ---------------------------------------------------------------------------


def build_presentation(output_path: Path, charts: dict[str, Path]) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =======================================================================
    # SLIDE 1: TITLE SLIDE
    # =======================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s1, NAVY_DARK)

    # Decorative geometric blocks
    accent_side = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.28), Inches(7.5))
    accent_side.fill.solid()
    accent_side.fill.fore_color.rgb = BLUE_ACCENT
    accent_side.line.fill.background()

    add_pill(s1, 0.85, 0.9, 4.3, 0.42, "CSET-346 NLP  •  MILESTONE-2 EVALUATION", RGBColor(30, 58, 138), RGBColor(191, 219, 254), font_size=10)

    tb_t = s1.shapes.add_textbox(Inches(0.85), Inches(1.6), Inches(11.4), Inches(2.0))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    r_t = p_t.add_run()
    r_t.text = "Hate Speech Detection through Explainable AI\nand Reward-Optimized LLMs"
    r_t.font.name = FONT_MAIN
    r_t.font.size = Pt(38)
    r_t.font.bold = True
    r_t.font.color.rgb = WHITE

    p_sub = tf_t.add_paragraph()
    p_sub.space_before = Pt(14)
    r_sub = p_sub.add_run()
    r_sub.text = "Project PJ-03: End-to-End English & Hinglish Moderation with Causal Attributions, Adversarial Defense & Live Serving"
    r_sub.font.name = FONT_MAIN
    r_sub.font.size = Pt(16)
    r_sub.font.color.rgb = RGBColor(148, 163, 184)

    # Bottom Metadata Box
    meta_box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.85), Inches(4.55), Inches(11.6), Inches(2.1))
    meta_box.adjustments[0] = 0.04
    meta_box.fill.solid()
    meta_box.fill.fore_color.rgb = NAVY_MID
    meta_box.line.color.rgb = RGBColor(51, 65, 85)
    meta_box.line.width = Pt(1)

    add_bullets(
        s1, 1.15, 4.78, 5.5, 1.7,
        [
            ("Team Members:", "Devansh Kumar, Ayush, Saksham Tyagi, Mudit Agrawal"),
            ("Supervisor:", "Dr. Pradeepika Verma"),
            ("Institution:", "Bennett University — SCSET (Nov 25–26, 2026)"),
        ],
        font_size=13,
        space_after=8,
    )
    # Fix colors for dark card
    for p in s1.shapes[-1].text_frame.paragraphs:
        p.runs[0].font.color.rgb = RGBColor(96, 165, 250)
        p.runs[1].font.color.rgb = WHITE

    add_bullets(
        s1, 7.0, 4.78, 5.2, 1.7,
        [
            ("GitHub Repository:", "github.com/devantaris/vivek (main branch)"),
            ("Verification:", "138 / 138 Unit Tests Passing  |  Ruff Clean"),
            ("Live Stack:", "FastAPI (56ms) + Gradio Web App (Port 7860)"),
        ],
        font_size=13,
        space_after=8,
    )
    for p in s1.shapes[-1].text_frame.paragraphs:
        p.runs[0].font.color.rgb = RGBColor(45, 212, 191)
        p.runs[1].font.color.rgb = WHITE

    add_notes(
        s1,
        "Good afternoon ma'am/sir. We are team PJ-03 presenting our Milestone-2 progress on Explainable Hate Speech Detection for English and Hinglish code-mixed text."
    )

    # =======================================================================
    # SLIDE 2: PROBLEM STATEMENT & MOTIVATION
    # =======================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Problem Statement: Why Standard Moderation Fails", "Three real-world vulnerabilities preventing trustworthy automated moderation", 2)

    cards_s2 = [
        ("1. The Black-Box & Bias Trap", CORAL, [
            ("Opaque Scores:", "Standard neural models output a bare probability (e.g., P = 0.94) with zero rationale."),
            ("Identity Shortcuts:", "Models learn spurious links between demographic terms ('Muslim', 'Jewish', 'African') and toxicity."),
            ("Real Failure:", "Benign affirmation ('I am a proud Muslim') gets flagged as hate speech (high False Positives)."),
        ]),
        ("2. Hinglish Vocabulary Collapse", BLUE_ACCENT, [
            ("Code-Mixed Reality:", "Indian users write Romanized Hindi + English without fixed spelling rules."),
            ("Subword Shattering:", "English tokenizers split insults ('chutiye') into meaningless chunks: ['chu', '##ti', '##ye']."),
            ("Deadly Blindspot:", "English baselines fall to 55.0% accuracy—missing death wishes like 'modi ko mar jana chahliye'."),
        ]),
        ("3. Adversarial Evasion", AMBER, [
            ("Character Disguises:", "Trolls swap letters, insert dots, or use leetspeak ('f@ggot', 's.l.u.t', 'I h a t e')."),
            ("OOV Collapse:", "Perturbed tokens fall outside fixed word vocabularies—standard TF-IDF fails >90% on spelling attacks."),
            ("Label Ambiguity:", "Conflating casual swearing ('offensive') with violent incitement ('hatespeech') hurts precision."),
        ]),
    ]
    for i, (ctitle, ccol, citems) in enumerate(cards_s2):
        left = 0.6 + i * 4.15
        add_card(s2, left, 1.3, 3.85, 4.75, ctitle, ccol)
        add_bullets(s2, left + 0.18, 1.95, 3.49, 3.9, citems, font_size=11.5, space_after=12)

    # Bottom takeaway banner
    add_pill(
        s2, 0.6, 6.28, 12.133, 0.58,
        "GOAL: Build an end-to-end bilingual detector that classifies accurately, resists disguises, explains WHY causally, and audits fairness.",
        NAVY_DARK, WHITE, font_size=12
    )
    add_notes(s2, "We address 3 concrete flaws: lack of causal explainability causing demographic false positives, Hinglish vocabulary collapse dropping English models to 55% accuracy, and adversarial obfuscation attacks.")

    # =======================================================================
    # SLIDE 3: DATASETS & BENCHMARK SUITE
    # =======================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Datasets: Multi-Lingual, Rationale-Annotated & Stress-Tested", "Justification, reproducible splits, and class distribution across 3 complementary corpora", 3)

    # Table of Datasets
    t_shape = s3.shapes.add_table(4, 5, Inches(0.6), Inches(1.25), Inches(12.133), Inches(2.8))
    tbl = t_shape.table
    col_w = [2.1, 2.2, 2.4, 2.4, 3.033]
    for idx, w in enumerate(col_w):
        tbl.columns[idx].width = Inches(w)

    headers = ["Dataset", "Source / Size", "Splits", "Class Distribution", "Why We Use It"]
    for c, h in enumerate(headers):
        style_table_cell(tbl.cell(0, c), h, NAVY_DARK, WHITE, bold=True, size=10.5)

    rows_s3 = [
        ["HateXplain\n(Mathew et al., 2021)", "Twitter & Gab\n20,148 total posts", "Train: 15,383 (80%)\nVal: 1,922 (10%)\nTest: 1,924 (10%)", "Normal: 40.5%\nHatespeech: 29.8%\nOffensive: 29.7%", "Provides gold word-level human rationale masks & 8 target groups"],
        ["HASOC 2019–2021 &\nHinglish Diagnostics", "Twitter Social\n8,500+ shared-task;\n30 diagnostic cases", "Evaluated on\ncode-mixed test\ntracks", "Binary: HOF (Hate /\nOffensive) vs NOT", "Captures Romanized Hindi slang, threats & varying Code-Mixing Index"],
        ["HateCheck\n(Röttger et al., 2021)", "Synthetic Diagnostic\n3,728 test cases", "100% Held-Out Test\n(Zero leakage)", "29 functional suites\n(Hateful vs Non-Hate)", "Audits char swaps, leetspeak, homoglyphs, questions & counter-speech"],
    ]
    for r_i, row in enumerate(rows_s3):
        bg = WHITE if r_i % 2 == 0 else BG_LIGHT
        for c_i, val in enumerate(row):
            style_table_cell(tbl.cell(r_i + 1, c_i), val, bg, TEXT_MAIN, bold=(c_i == 0), size=9.5)

    # Bottom 2 cards explaining key schema & data engineering nuance
    add_card(s3, 0.6, 4.3, 5.9, 2.55, "Unified Schema & Split Integrity", BLUE_ACCENT)
    add_bullets(s3, 0.78, 4.92, 5.5, 1.8, [
        ("Standardized Schema (schema.py):", "Every dataset normalizes into (id, text, label, language, target_group, rationale_mask)."),
        ("Majority-Vote Consensus:", "HateXplain ties broken toward higher harm; token rationales require >= 2/3 annotator agreement."),
    ], font_size=11, space_after=8)

    add_card(s3, 6.8, 4.3, 5.933, 2.55, "Critical Data Discovery: 'Strict Hate' vs 'Offensive'", CORAL)
    add_bullets(s3, 6.98, 4.92, 5.55, 1.8, [
        ("Why It Matters:", "Under HateXplain's 'strict_hate' mode, 4,514 'offensive' posts (e.g., 'I hate women') map to class 0."),
        ("Impact Observed:", "Models trained strictly on label 0 treat casual profanity as safe—explaining low HateCheck scores."),
    ], font_size=11, space_after=8)

    add_notes(s3, "Criterion 1: Here are our 3 datasets—HateXplain gives us gold human rationale masks, HASOC covers Hinglish code-mixing, and HateCheck stress-tests 29 attack types without data leakage.")

    # =======================================================================
    # SLIDE 4: END-TO-END ARCHITECTURE
    # =======================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "End-to-End Architecture: From Raw Post to Audited Moderation", "Modular 5-layer pipeline built in src/pj03/ with strict privacy & CI safeguards", 4)

    # Layer 1 -> Layer 2 -> Layer 3 -> Layer 4 -> Layer 5 flow boxes
    boxes = [
        ("1. Preprocessing &\nDefense", "src/pj03/preprocess/", [
            "• Zero-Leak PII Scrubber",
            "• Unicode & Leetspeak\n  De-obfuscator",
            "• Hinglish Phonetic\n  Canonicalizer",
        ], BLUE_ACCENT),
        ("2. Multi-Model\nEngine", "baselines/ & sft/", [
            "• TF-IDF + Char N-Gram\n  LogReg (18ms CPU)",
            "• Google MuRIL (236M)\n  WeightedTrainer",
            "• Qwen2.5-1.5B LoRA",
        ], NAVY_DARK),
        ("3. Calibrated\nEnsemble & RL", "reward/ & demo/", [
            "• 65% LLM + 35% TF-IDF\n  Ensemble Guard",
            "• Hinglish Threat Override",
            "• Composite Reward\n  (Label + Faith - Bias)",
        ], TEAL),
        ("4. XAI & Audit\nSuite", "xai/ & eval/", [
            "• Captum Integrated\n  Gradients + SHAP",
            "• ERASER Comp / Suff",
            "• Borkan 7-Group Bias",
            "• HateCheck & CI Gate",
        ], AMBER),
        ("5. Serving &\nActive Learning", "service/ & scripts/", [
            "• FastAPI REST (/predict)",
            "• Gradio UI (Port 7860)",
            "• PII-Safe JSONL Logs",
            "• Human Review Queue",
        ], CORAL),
    ]

    for i, (btitle, bmod, bpts, bcol) in enumerate(boxes):
        x = 0.5 + i * 2.52
        add_card(s4, x, 1.35, 2.25, 4.3, "", bcol)
        # Header badge inside card
        hdr = s4.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(1.35), Inches(2.25), Inches(0.82))
        hdr.fill.solid()
        hdr.fill.fore_color.rgb = bcol
        hdr.line.fill.background()
        htf = hdr.text_frame
        htf.vertical_anchor = MSO_ANCHOR.MIDDLE
        hp = htf.paragraphs[0]
        hp.alignment = PP_ALIGN.CENTER
        hr = hp.add_run()
        hr.text = btitle
        hr.font.name = FONT_MAIN
        hr.font.size = Pt(12)
        hr.font.bold = True
        hr.font.color.rgb = WHITE

        # Module tag
        add_pill(s4, x + 0.15, 2.28, 1.95, 0.28, bmod, BG_LIGHT, TEXT_MUTED, font_size=8.5)

        # Bullet text
        tb = s4.shapes.add_textbox(Inches(x + 0.12), Inches(2.68), Inches(2.0), Inches(2.8))
        tf = tb.text_frame
        tf.word_wrap = True
        for j, pt in enumerate(bpts):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            p.space_after = Pt(7)
            r = p.add_run()
            r.text = pt
            r.font.name = FONT_MAIN
            r.font.size = Pt(10)
            r.font.color.rgb = TEXT_MAIN

        # Arrow to next box
        if i < len(boxes) - 1:
            arr = s4.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 2.29), Inches(3.3), Inches(0.19), Inches(0.26))
            arr.fill.solid()
            arr.fill.fore_color.rgb = TEXT_MUTED
            arr.line.fill.background()

    # Bottom note
    add_card(s4, 0.5, 5.88, 12.333, 1.0, "", TEAL, TEAL_LIGHT)
    add_bullets(s4, 0.72, 6.03, 11.9, 0.75, [
        ("Key Architectural Strength:", "Unlike standalone notebooks, every stage is modular, unit-tested (138 tests), config-driven, and connected to an automated CI Release Gate (Macro-F1 ≥ 0.78, AUC ≥ 0.85, BPSN ≥ 0.60) before serving."),
    ], font_size=11.5, space_after=0)

    add_notes(s4, "Criterion 2: Walk through the 5 stages left-to-right. Notice how privacy, de-obfuscation, multi-model inference, XAI, and human feedback form a complete closed loop.")

    # =======================================================================
    # SLIDE 5: MULTI-MODEL STACK
    # =======================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Methodology: Progressive Multi-Model Modeling Stack", "Combining ultra-fast linear interpretability, Indic encoders, and instruction-tuned LLMs", 5)

    models_s5 = [
        ("Model 1: TF-IDF + Logistic Regression", BLUE_ACCENT, [
            ("Features:", "Word (1,2)-grams + Subword Char_wb (2,5)-grams (~150k total features)."),
            ("Exact Attribution:", "Feature impact = tfidf_value × coefficient (zero approximation error)."),
            ("Performance:", "Macro-F1: 0.810 | ROC-AUC: 0.899 | CPU Latency: ~18.4 ms."),
            ("Role:", "Fast production floor & ensemble anchor against LLM hallucinations."),
        ]),
        ("Model 2: Google MuRIL (Indic BERT)", TEAL, [
            ("Architecture:", "236M-param Transformer pre-trained on 17 Indian languages + Romanized Hindi."),
            ("Contextual Attention:", "Bidirectional self-attention distinguishes literal threats from benign idioms."),
            ("Initial Run vs Fix:", "3-epoch unweighted run scored 0.767 F1. Upgraded with WeightedTrainer (w_c = N/2N_c) + 10% linear warmup."),
            ("XAI Hook:", "Directly supports Captum Integrated Gradients on word embeddings."),
        ]),
        ("Model 3: Qwen2.5-1.5B + LoRA SFT", NAVY_DARK, [
            ("Parameter Efficiency:", "LoRA rank r=16, alpha=32 updates attention projections; saved adapter is only 17.5 MB."),
            ("Structured Prompting:", "Completion-only loss masking trains model to emit: Label: <0|1> | Rationale: <text>."),
            ("Convergence:", "Achieved Eval Loss: 0.1257 across 2 epochs on Colab T4 GPU."),
            ("Ensemble Guard:", "Blends 65% LLM logit + 35% TF-IDF + hard Hinglish threat override."),
        ]),
    ]

    for i, (mtitle, mcol, mitems) in enumerate(models_s5):
        left = 0.6 + i * 4.15
        add_card(s5, left, 1.3, 3.85, 5.5, mtitle, mcol)
        add_bullets(s5, left + 0.18, 1.98, 3.49, 4.65, mitems, font_size=11, space_after=11)

    add_notes(s5, "Explain why we have 3 models: TF-IDF for speed and exact weights, MuRIL for Indian bilingual context, and Qwen2.5-1.5B LoRA SFT to generate natural-language explanations.")

    # =======================================================================
    # SLIDE 6: HINGLISH & ADVERSARIAL ROBUSTNESS
    # =======================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Breakthroughs: Hinglish Engine & Adversarial De-Obfuscation", "Resolving Romanized vocabulary collapse and character-level evasion attacks", 6)

    # Left: Chart
    add_card(s6, 0.6, 1.28, 6.0, 4.1, "", TEAL)
    s6.shapes.add_picture(str(charts["hinglish"]), Inches(0.75), Inches(1.42), width=Inches(5.7))

    # Left Bottom: Live Audit Examples Table
    t6 = s6.shapes.add_table(4, 3, Inches(0.6), Inches(5.52), Inches(6.0), Inches(1.45)).table
    t6.columns[0].width = Inches(2.85)
    t6.columns[1].width = Inches(1.6)
    t6.columns[2].width = Inches(1.55)
    style_table_cell(t6.cell(0, 0), "Live Audit Test Input", NAVY_DARK, WHITE, bold=True, size=9)
    style_table_cell(t6.cell(0, 1), "Prediction", NAVY_DARK, WHITE, bold=True, size=9, align=PP_ALIGN.CENTER)
    style_table_cell(t6.cell(0, 2), "P(hateful)", NAVY_DARK, WHITE, bold=True, size=9, align=PP_ALIGN.CENTER)

    audit_rows = [
        ['"modi ko mar jana chahliye"', "HATEFUL", "0.920", CORAL_LIGHT, CORAL],
        ['"f@ggot trash" (Leetspeak)', "HATEFUL", "0.789", CORAL_LIGHT, CORAL],
        ['"modi visited university today"', "NON-HATEFUL", "0.129", TEAL_LIGHT, TEAL],
    ]
    for r_i, (txt, pred, prob, bg, fg) in enumerate(audit_rows):
        style_table_cell(t6.cell(r_i + 1, 0), txt, WHITE, TEXT_MAIN, size=9)
        style_table_cell(t6.cell(r_i + 1, 1), pred, bg, fg, bold=True, size=9, align=PP_ALIGN.CENTER)
        style_table_cell(t6.cell(r_i + 1, 2), prob, bg, fg, bold=True, size=9, align=PP_ALIGN.CENTER)

    # Right: Two Cards explaining Hinglish + Deobfuscator
    add_card(s6, 6.85, 1.28, 5.88, 2.7, "Hinglish Normalizer (src/pj03/preprocess/hinglish.py)", BLUE_ACCENT)
    add_bullets(s6, 7.05, 1.9, 5.48, 1.95, [
        ("Phonetic Canonicalization:", "Maps noisy variants (madharchod, maderchod, mc) to canonical roots."),
        ("Multi-Word Threat Regex:", "Detects split-word violence ('mar jana chah[il]*iye', 'goli maar', 'khatam kar[oa]?')."),
        ("Impact:", "Lifted code-mixed Macro-F1 from 0.436 (55% acc) to 0.933 (93.3% acc)."),
    ], font_size=11, space_after=8)

    add_card(s6, 6.85, 4.18, 5.88, 2.79, "Adversarial De-Obfuscator (deobfuscator.py + adversarial.py)", AMBER)
    add_bullets(s6, 7.05, 4.8, 5.48, 2.05, [
        ("Unicode & Zero-Width:", "Maps Cyrillic/Greek lookalikes to Latin; strips invisible \\u200B characters."),
        ("Spaced & Punctuated:", "Collapses 's.l.u.t' -> 'slut' and 'I h a t e' -> 'I hate'."),
        ("Typos & Leetspeak:", "Resolves swaps ('haet' -> 'hate', 'sikc' -> 'sick') and symbols ('s1ut', 'f@ggot')."),
        ("Perturbation Engine:", "adversarial.py generates synthetic char swaps & questions for training."),
    ], font_size=10.5, space_after=6)

    add_notes(s6, "Show how our Hinglish engine fixed our exact demo failures (like 'modi ko mar jana chahliye') while avoiding false alarms on benign posts like 'modi visited university today'.")

    # =======================================================================
    # SLIDE 7: XAI & CAUSAL FAITHFULNESS
    # =======================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Explainable AI: Causal Attributions & Faithfulness Metrics", "Moving from unverified attention maps to mathematically grounded explanations", 7)

    add_card(s7, 0.6, 1.3, 6.0, 5.55, "1. Token Attribution Backends (xai/attribution.py)", BLUE_ACCENT)
    add_bullets(s7, 0.8, 1.95, 5.6, 4.7, [
        ("Linear Exact Weights (TF-IDF):", "Computes tfidf(token) × coefficient directly from the decision hyperplane."),
        ("Captum Integrated Gradients:", "For deep transformers (MuRIL/BERT), integrates embedding gradients along 30 path steps from a zero baseline to satisfy the Axiom of Completeness."),
        ("Token-Level KernelSHAP:", "Model-agnostic Shapley attribution: removes single tokens (n <= 12) or samples Monte Carlo coalitions (n > 12) to measure true marginal ΔP."),
        ("Gradio Dual-Color Highlighting:", "Red tokens push toward HATEFUL (+); Green tokens push toward NON-HATEFUL (-)."),
    ], font_size=11.5, space_after=13)

    add_card(s7, 6.85, 1.3, 5.88, 3.1, "2. ERASER Faithfulness Verification (faithfulness.py)", TEAL)
    add_bullets(s7, 7.05, 1.95, 5.48, 2.3, [
        ("Comprehensiveness = P(full) − P(without top-k):", "Higher is better. Removing top-attributed words must collapse hateful confidence."),
        ("Sufficiency = P(full) − P(only top-k):", "Closer to 0 is better. Top rationale tokens alone should retain the decision."),
        ("Why It Beats Attention:", "Prevents 'decorative' explanations—rationales must causally control the output."),
    ], font_size=11, space_after=10)

    add_card(s7, 6.85, 4.6, 5.88, 2.25, "3. Rationale Quality Evaluation (eval/rationale_eval.py)", NAVY_DARK)
    add_bullets(s7, 7.05, 5.22, 5.48, 1.5, [
        ("Token IoU & Token F1:", "Measures exact Jaccard overlap between model highlights and HateXplain human rationale spans."),
        ("ROUGE-1 / ROUGE-2 / ROUGE-L:", "Evaluates natural-language rationales emitted by the Qwen LoRA LLM."),
    ], font_size=11, space_after=8)

    add_notes(s7, "Highlight that we don't just display pretty colors: Captum Integrated Gradients gives deep attributions, and ERASER Comprehensiveness & Sufficiency mathematically verify that highlighted words drive the decision.")

    # =======================================================================
    # SLIDE 8: REWARD DESIGN & FAIRNESS AUDIT
    # =======================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Composite Reward & Borkan Demographic Fairness Audit", "Penalizing identity shortcuts across 7 protected groups (African, Arab, Islam, Jewish, etc.)", 8)

    # Left: Formula & Reward Breakdown
    add_card(s8, 0.6, 1.28, 5.5, 2.35, "Composite Reward Contract (reward/composite.py)", NAVY_DARK)
    add_pill(s8, 0.8, 1.88, 5.1, 0.45, "R  =  α·R_label  +  β·R_faith  −  γ·R_bias   (1.0, 0.5, 0.3)", BLUE_LIGHT, NAVY_DARK, font_size=11)
    add_bullets(s8, 0.8, 2.45, 5.1, 1.1, [
        ("R_label:", "Binary match + strict format check (Label: | Rationale:)."),
        ("R_faith:", "|ΔP| when masking cited rationale words."),
        ("R_bias:", "Penalty if rationale cites protected identity lexicons."),
    ], font_size=10, space_after=4)

    add_card(s8, 0.6, 3.82, 5.5, 3.05, "Borkan et al. (2019) Audit & Honesty Note", CORAL)
    add_bullets(s8, 0.8, 4.44, 5.1, 2.3, [
        ("Subgroup AUC:", "Separating hate vs benign inside one group (0.717 Arab to 0.865 Women)."),
        ("BPSN AUC (False Positives):", "Benign group mentions vs hateful background (baseline low on African 0.696 & Jewish 0.700)."),
        ("BNSP AUC (Missed Hate):", "Group hate vs benign background (0.743 Refugee to 0.943 Jewish)."),
        ("Status Note:", "Chart shows offline calibration margin (400 samples); scripts/train_grpo_gpu.py is built for live 500-step GPU RL."),
    ], font_size=10, space_after=6)

    # Right: Chart
    add_card(s8, 6.35, 1.28, 6.38, 5.59, "", BLUE_ACCENT)
    s8.shapes.add_picture(str(charts["fairness"]), Inches(6.48), Inches(1.85), width=Inches(6.12))

    add_notes(s8, "Explain BPSN and BNSP clearly, and proactively note that online GPU GRPO training is ready via train_grpo_gpu.py while the chart compares baseline raw BPSN vs our offline reward-margin calibration.")

    # =======================================================================
    # SLIDE 9: BENCHMARK RESULTS & ERROR ANALYSIS
    # =======================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Empirical Results & Honest Error Analysis", "Comprehensive evaluation on HateXplain (1,924), Unified Code-Mixed (30) & HateCheck (3,728)", 9)

    # Top: Chart + Summary Table
    add_card(s9, 0.6, 1.25, 6.2, 3.9, "", BLUE_ACCENT)
    s9.shapes.add_picture(str(charts["models"]), Inches(0.72), Inches(1.36), width=Inches(5.96))

    t9 = s9.shapes.add_table(6, 4, Inches(7.0), Inches(1.25), Inches(5.733), Inches(3.15)).table
    t9.columns[0].width = Inches(2.25)
    t9.columns[1].width = Inches(1.1)
    t9.columns[2].width = Inches(1.1)
    t9.columns[3].width = Inches(1.283)
    for c, h in enumerate(["Model / Track", "Macro-F1", "ROC-AUC", "Notes"]):
        style_table_cell(t9.cell(0, c), h, NAVY_DARK, WHITE, bold=True, size=9.5, align=(PP_ALIGN.CENTER if c in (1, 2) else PP_ALIGN.LEFT))

    res_rows = [
        ["TF-IDF (word) + LogReg", "0.799", "0.891", "50k features"],
        ["TF-IDF (word+char)", "0.810", "0.899", "83.3% Accuracy"],
        ["Google MuRIL (3 epochs)", "0.767", "0.849", "Eval loss: 0.520"],
        ["Qwen2.5-1.5B LoRA SFT", "0.785", "0.862", "Eval loss: 0.1257"],
        ["Unified HASOC+Hinglish", "0.933", "0.916", "93.3% Accuracy"],
    ]
    for r_i, row in enumerate(res_rows):
        bg = WHITE if r_i % 2 == 0 else BG_LIGHT
        for c_i, val in enumerate(row):
            style_table_cell(t9.cell(r_i + 1, c_i), val, bg, TEXT_MAIN, bold=(c_i == 0 or r_i == 4), size=9, align=(PP_ALIGN.CENTER if c_i in (1, 2) else PP_ALIGN.LEFT))

    # Bottom Right & Left: Honest Error Analysis
    add_card(s9, 7.0, 4.55, 5.733, 2.38, "Why TF-IDF (0.810) Beat Initial MuRIL (0.767)", TEAL)
    add_bullets(s9, 7.18, 5.15, 5.38, 1.65, [
        ("Key Finding:", "Character n-grams tightly capture explicit Twitter slurs. MuRIL's initial 3-epoch unweighted run suffered from class imbalance."),
        ("Code Upgrade:", "Added class weights + warmup in WeightedTrainer."),
    ], font_size=10, space_after=6)

    add_card(s9, 0.6, 5.3, 6.2, 1.63, "HateCheck Gap Analysis (36.0% Raw Accuracy Across 3,728 Tests)", CORAL)
    add_bullets(s9, 0.78, 5.88, 5.84, 0.98, [
        ("Root Causes:", "(1) strict_hate mapped 'offensive' posts ('I hate women') to label 0; (2) char swaps (2.3% pass) & questions (4.3% pass)."),
        ("Mitigation:", "De-obfuscator lifted live adversarial audit to 69.2% (9/13)."),
    ], font_size=10, space_after=4)

    add_notes(s9, "Present our numbers transparently: TF-IDF leads in-domain at 0.810 F1, code-mixed reached 0.933, and we clearly diagnosed why MuRIL trailed and why HateCheck raw scored 36%.")

    # =======================================================================
    # SLIDE 10: PRODUCTION SERVING, PRIVACY & TESTING
    # =======================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Production Engineering: FastAPI, Gradio, Privacy & Testing", "Enterprise-ready deployment verified end-to-end with zero PII leaks", 10)

    prod_cards = [
        ("FastAPI Microservice", BLUE_ACCENT, [
            ("Endpoints:", "GET /health, POST /predict, review queue endpoints."),
            ("Performance:", "56.19 ms average E2E latency across verified integration suite."),
            ("Configurability:", "Per-request sensitivity threshold τ and model routing (auto / llm / tfidf)."),
        ]),
        ("Gradio Interactive UI (Port 7860)", TEAL, [
            ("Classify & Explain:", "Dual-color token highlighting (red hateful, green non-hateful) + live faithfulness."),
            ("Natural Rationale:", "Displays LLM structured explanation alongside token impacts."),
            ("Model Card Tab:", "Shows live architecture leaderboard & Borkan 7-group fairness audit."),
        ]),
        ("Privacy & Active Learning", CORAL, [
            ("Zero-Leak PII Scrubber:", "Emails, phones, URLs, & @handles scrubbed before writing logs/decisions.jsonl."),
            ("Review Queue:", "Moderators review borderline posts; corrections become hard negatives (retrain_from_feedback.py)."),
        ]),
        ("Engineering Verification", NAVY_DARK, [
            ("138 / 138 Unit Tests:", "100% pass rate via pytest across loaders, schemas, XAI, Hinglish, & API."),
            ("0 Linter Warnings:", "Strict PEP 8 compliance via ruff check."),
            ("E2E Report:", "test_production_e2e.py -> ALL_TESTS_PASSED."),
        ]),
    ]

    positions = [(0.6, 1.3), (6.8, 1.3), (0.6, 4.18), (6.8, 4.18)]
    for (ptitle, pcol, pitems), (px, py) in zip(prod_cards, positions, strict=False):
        add_card(s10, px, py, 5.933, 2.65, ptitle, pcol)
        add_bullets(s10, px + 0.18, py + 0.62, 5.56, 1.9, pitems, font_size=11, space_after=8)

    add_notes(s10, "Show that this is executable, tested software: 138 passing tests, sub-60ms API latency, write-time PII scrubbing, and active learning.")

    # =======================================================================
    # SLIDE 11: INDIVIDUAL CONTRIBUTIONS
    # =======================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Team Contributions & Ownership Matrix", "Balanced engineering ownership across all 4 team members (Criterion 5)", 11)

    team_s11 = [
        ("Devansh Kumar", "Team Lead • Architecture & Serving", BLUE_ACCENT, [
            ("Owned:", "End-to-end architecture, FastAPI REST service, Gradio interactive demo."),
            ("Modeling:", "Qwen2.5-1.5B LoRA SFT pipeline & 65/35 LLM+TF-IDF ensemble guard."),
            ("Files:", "service/api.py, demo/app.py, demo/predictor.py, sft/lora_sft.py."),
        ]),
        ("Ayush", "Data, Baselines & Privacy", TEAL, [
            ("Owned:", "Dataset standardization (HateXplain 20k, HASOC), split builder & PII scrubber."),
            ("Modeling:", "TF-IDF word+char baseline (0.810) & MuRIL WeightedTrainer setup."),
            ("Files:", "preprocess/schema.py, loaders_hatexplain.py, pii.py, encoder_finetune.py."),
        ]),
        ("Saksham Tyagi", "Hinglish & Adversarial Defense", AMBER, [
            ("Owned:", "Hinglish phonetic canonicalizer, death-threat regexes & HateCheck eval."),
            ("Defense:", "Built deobfuscator.py (leetspeak/homoglyphs) & adversarial.py."),
            ("Files:", "preprocess/hinglish.py, deobfuscator.py, adversarial.py, eval/hatecheck.py."),
        ]),
        ("Mudit Agrawal", "XAI, Faithfulness & Fairness", CORAL, [
            ("Owned:", "Captum Integrated Gradients, Token KernelSHAP & Rationale IoU/ROUGE."),
            ("Auditing:", "ERASER Comprehensiveness/Sufficiency & Borkan 7-group fairness audit."),
            ("Files:", "xai/attribution.py, xai/faithfulness.py, eval/fairness.py, reward/r_bias.py."),
        ]),
    ]

    for i, (name, role, col, items) in enumerate(team_s11):
        x = 0.55 + i * 3.1
        add_card(s11, x, 1.32, 2.93, 5.45, "", col)
        add_pill(s11, x + 0.16, 1.52, 2.61, 0.42, name, col, WHITE, font_size=13)
        tb = s11.shapes.add_textbox(Inches(x + 0.16), Inches(2.02), Inches(2.61), Inches(0.42))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = role
        r.font.name = FONT_MAIN
        r.font.size = Pt(10)
        r.font.bold = True
        r.font.color.rgb = TEXT_MUTED

        add_bullets(s11, x + 0.16, 2.55, 2.61, 4.05, items, font_size=10.5, space_after=10)

    add_notes(s11, "Each team member will speak for ~1 minute covering their respective card and code modules.")

    # =======================================================================
    # SLIDE 12: FUTURE ROADMAP & LIVE DEMO
    # =======================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Roadmap to Final Evaluation & Live Demo", "Planned GPU scale-up, 3-class hierarchy, and immediate live demonstration", 12)

    add_card(s12, 0.6, 1.3, 6.2, 5.4, "Future Work (Before Final Submission)", BLUE_ACCENT)
    add_bullets(s12, 0.8, 1.95, 5.8, 4.6, [
        ("1. True Online GPU GRPO Training:", "Execute 500 policy steps using scripts/train_grpo_gpu.py (G=4, β=0.04) on Colab A100/T4 to verify converged RL weights."),
        ("2. MuRIL Retraining & XLM-R Large:", "Run WeightedTrainer for 5 epochs on joint HateXplain + full multi-year HASOC corpora (download_hasoc_full.py)."),
        ("3. 3-Class Hierarchical Triage:", "Separate Benign vs Toxic, then Hate Speech (incitement) vs Casual Profanity—fixing the HateCheck 'offensive' conflation."),
        ("4. Adversarial SFT Augmentation:", "Retrain Qwen2.5-1.5B with 20% perturbed HateCheck-style variants from adversarial.py."),
    ], font_size=11, space_after=12)

    add_card(s12, 7.05, 1.3, 5.68, 5.4, "Live Demo Sequence (What We Will Show Now)", TEAL, BG_LIGHT)
    add_bullets(s12, 7.25, 1.95, 5.28, 3.1, [
        ("Terminal Verification:", "pytest tests/ -q (138 passed) & test_production_e2e.py (56ms)."),
        ("Demo Case 1 (Hinglish Threat):", "'modi ko mar jana chahliye' -> HATEFUL (0.920)."),
        ("Demo Case 2 (Hinglish Abuse):", "'modi madharchod hai' -> HATEFUL (0.920)."),
        ("Demo Case 3 (Leetspeak):", "'f@ggot trash' -> HATEFUL (0.789)."),
        ("Demo Case 4 (Benign Entity):", "'modi visited the university today' -> NON-HATEFUL (0.129)."),
    ], font_size=11, space_after=9)

    add_pill(
        s12, 7.3, 5.45, 5.18, 0.9,
        "Thank You!\nQuestions & Live Code Demonstration",
        NAVY_DARK, WHITE, font_size=15
    )

    add_notes(s12, "Wrap up with our clear final evaluation roadmap and immediately switch screens to run pytest, test_production_e2e.py, and demo_app.py at localhost:7860.")

    prs.save(str(output_path))
    print(f"Saved PowerPoint: {output_path}")


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    charts = generate_charts()
    out_pptx = DOCS_DIR / "PJ03_Milestone2_Presentation.pptx"
    build_presentation(out_pptx, charts)


if __name__ == "__main__":
    main()
