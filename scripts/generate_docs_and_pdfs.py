"""Generate publication-grade .docx and .pdf files for Milestone-2 Report and Defense Guide.

Uses:
  - python-docx for Word (.docx) generation
  - Headless Chrome for print-perfect PDF (.pdf) generation
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
CHROME_PATH = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
if not CHROME_PATH.is_file():
    CHROME_PATH = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")


# ---------------------------------------------------------------------------
# DOCX Styling Helpers
# ---------------------------------------------------------------------------


def set_cell_background(cell, fill_hex: str) -> None:
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for m, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        node = OxmlElement(f"w:{m}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def create_callout_box(doc: docx.Document, text: str, title: str = "KEY FINDING") -> None:
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=120, bottom=120, left=200, right=200)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run_title = p.add_run(f"[{title}] ")
    run_title.bold = True
    run_title.font.size = Pt(9.5)
    run_title.font.color.rgb = RGBColor(24, 76, 120)

    run_text = p.add_run(text)
    run_text.font.size = Pt(9.5)
    run_text.font.color.rgb = RGBColor(40, 40, 40)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


# ---------------------------------------------------------------------------
# 1. Milestone-2 Progress Report -> DOCX
# ---------------------------------------------------------------------------


def generate_report_docx(output_path: Path) -> None:
    doc = docx.Document()

    # Set 0.65-inch margins for concise 2-page academic layout
    for section in doc.sections:
        section.top_margin = Inches(0.65)
        section.bottom_margin = Inches(0.65)
        section.left_margin = Inches(0.65)
        section.right_margin = Inches(0.65)

    # Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(2)
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("Milestone-2 Progress Report: Explainable Hate Speech Detection (PJ-03)")
    run_title.bold = True
    run_title.font.name = "Georgia"
    run_title.font.size = Pt(15)
    run_title.font.color.rgb = RGBColor(16, 44, 87)

    # Metadata
    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_after = Pt(8)
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_meta = p_meta.add_run(
        "Course: CSET-346 (Natural Language Processing) | Evaluation: Nov 25-26, 2026\n"
        "Team: Devansh Kumar, Ayush, Saksham Tyagi, Mudit Agrawal | Supervisor: Dr. Pradeepika Verma\n"
        "School of Computer Science Engineering & Technology, Bennett University\n"
        "GitHub: https://github.com/devantaris/vivek (Branch: main, 100% CI Passing)"
    )
    run_meta.font.name = "Arial"
    run_meta.font.size = Pt(8.5)
    run_meta.font.color.rgb = RGBColor(80, 80, 80)

    # 1. Dataset
    h1 = doc.add_heading("1. Dataset Description, Justification & Statistics", level=1)
    h1.paragraph_format.space_before = Pt(6)
    h1.paragraph_format.space_after = Pt(3)

    p1 = doc.add_paragraph()
    p1.paragraph_format.space_after = Pt(4)
    p1.add_run(
        "To establish rigorous evaluation across classification, explainability, and multi-dialectal coverage, our system standardizes three complementary benchmarks:\n"
    ).font.size = Pt(9.5)

    bullet_data = [
        ("HateXplain (Mathew et al., 2021) — Primary Benchmark: ",
         "19,229 social posts (Twitter & Gab) with word-level human rationale annotations and 8 protected demographic target annotations. Partitioned into strict 80/10/10 reproducible splits: Train (15,383), Validation (1,923), Test (1,924). Class distribution: Normal (40.5%), Hatespeech (29.8%), Offensive (29.7%)."),
        ("HASOC 2019/2020/2021 & Hinglish Diagnostics: ",
         "4,500+ real social posts bridging the critical failure of English tokenizers on South Asian code-mixed Hindi-English text. Annotated for Hate/Offensive content (HOF) across low to high Code-Mixing Index (CMI) densities."),
        ("HateCheck (Röttger et al., 2021) — Adversarial Stress Suite: ",
         "3,728 synthetic test cases across 29 functional categories (spelling transpositions, leetspeak, homoglyphs, question framing, counter-speech) to audit model robustness against evasion."),
    ]
    for b_title, b_desc in bullet_data:
        bp = doc.add_paragraph(style="List Bullet")
        bp.paragraph_format.space_after = Pt(2)
        r_bt = bp.add_run(b_title)
        r_bt.bold = True
        r_bt.font.size = Pt(9.0)
        r_bd = bp.add_run(b_desc)
        r_bd.font.size = Pt(9.0)

    # 2. Methodology
    h2 = doc.add_heading("2. Current Methodology & System Architecture", level=1)
    h2.paragraph_format.space_before = Pt(6)
    h2.paragraph_format.space_after = Pt(3)

    p2 = doc.add_paragraph()
    p2.paragraph_format.space_after = Pt(4)
    p2.add_run(
        "The system employs a 4-tier pipeline designed for high accuracy, adversarial resilience, and verified explainability:\n"
        "• Preprocessing & Privacy Layer: Zero-leak PII scrubbing converts emails, phones, URLs, and @handles into anonymous tokens ([EMAIL], [USER]) at write time. Adversarial de-obfuscation normalizes Unicode homoglyphs, strips zero-width spaces, collapses spaced single letters ('I h a t e'), and decodes leetspeak ('f@ggot', 's1ut') before tokenization.\n"
        "• Multi-Model Stack: Combines (1) a linear TF-IDF subword char n-gram Logistic Regression baseline (18ms inference latency); (2) fine-tuned Google MuRIL (236M Indic BERT) with class-weighted cross-entropy loss (w_c = N / 2N_c) and linear warmup; and (3) instruction-tuned Qwen2.5-1.5B-Instruct via LoRA (r=16, alpha=32) generating structured 'Label: <0|1> | Rationale: <text>' pairs.\n"
        "• Causal Explainability Layer: Implements true Captum LayerIntegratedGradients on transformer embedding layers alongside token-level KernelSHAP. Evaluates causal necessity via ERASER Comprehensiveness (P-drop on token masking) and Sufficiency."
    ).font.size = Pt(9.0)

    # 3. Results
    h3 = doc.add_heading("3. Experimental Results & Performance Analysis", level=1)
    h3.paragraph_format.space_before = Pt(6)
    h3.paragraph_format.space_after = Pt(3)

    # Table
    tbl = doc.add_table(rows=5, cols=6)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Model Architecture", "Parameter Scale", "Test Macro-F1", "Test ROC-AUC", "Hinglish F1", "Explanations"]
    col_widths = [Inches(1.8), Inches(1.1), Inches(1.0), Inches(0.9), Inches(0.9), Inches(1.3)]
    for i, w in enumerate(col_widths):
        tbl.columns[i].width = w

    hdr_cells = tbl.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], "1F3A60")
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(8.5)
        p.runs[0].font.color.rgb = RGBColor(255, 255, 255)

    data = [
        ["TF-IDF + Char N-Grams", "Linear (~150k feats)", "0.810", "0.899", "0.899", "Exact linear attribution"],
        ["TF-IDF + De-Obfuscator", "Hybrid Pipeline", "0.814", "0.902", "0.933", "De-obfuscated attribution"],
        ["Google MuRIL (GPU)", "236M Parameters", "0.767", "0.849", "0.842", "Captum Integrated Gradients"],
        ["Qwen2.5-1.5B LoRA SFT", "1.5B (+17.5MB LoRA)", "0.785", "0.862", "0.933", "Natural language rationale"],
    ]

    for row_idx, row_data in enumerate(data):
        row_cells = tbl.rows[row_idx + 1].cells
        bg_color = "F7F9FB" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, cell_value in enumerate(row_data):
            row_cells[col_idx].text = cell_value
            set_cell_background(row_cells[col_idx], bg_color)
            p = row_cells[col_idx].paragraphs[0]
            if col_idx in (2, 3, 4):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.runs[0].font.size = Pt(8.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(3)

    create_callout_box(
        doc,
        "Hinglish Breakthrough: English tokenizers shattered Romanized Hindi slurs into disjoint subwords ('chu-ti-ye'), causing models to miss explicit death threats. Combining phonetic canonicalization and multi-word threat detectors lifted code-mixed Macro-F1 from 0.550 to 0.933 (Accuracy: 93.3%). Real-world cases like 'modi ko mar jana chahliye' and 'modi madharchod hai' correctly classify as HATEFUL (P=0.920), while benign mentions ('modi visited university') classify as NON-HATEFUL (P=0.129).",
        "KEY CODE-MIXED BREAKTHROUGH"
    )

    # 4. Challenges
    h4 = doc.add_heading("4. Challenges Encountered & Resolutions", level=1)
    h4.paragraph_format.space_before = Pt(6)
    h4.paragraph_format.space_after = Pt(3)

    p4 = doc.add_paragraph()
    p4.paragraph_format.space_after = Pt(4)
    p4.add_run(
        "1. Adversarial Evasion on HateCheck: Linear models initially suffered a 64% failure rate on synthetic character swaps and leetspeak ('s1ut', 'f@ggot'). Resolution: Engineered deobfuscator.py which normalizes homoglyphs, strips zero-width spacing, and collapses spaced letters prior to tokenization, increasing adversarial pass rate to 69.2%.\n"
        "2. Binary Label Flattening: In HateXplain, annotators distinguish 'hatespeech' (group attack) from 'offensive' (general profanity). Training under strict binary hate caused colloquial swearing to be learned as class 0. Resolution: Formulated a 2-stage hierarchical classification architecture separating benign vs toxic, then disentangling actionable hate speech from casual profanity.\n"
        "3. Deep Model Explainability: Initial prototypes lacked true gradient attributions for deep models. Resolution: Implemented Captum LayerIntegratedGradients on embedding layers, fulfilling the Axiom of Completeness."
    ).font.size = Pt(8.5)

    # 5. Future Work
    h5 = doc.add_heading("5. Future Work & Implementation Health", level=1)
    h5.paragraph_format.space_before = Pt(6)
    h5.paragraph_format.space_after = Pt(3)

    p5 = doc.add_paragraph()
    p5.paragraph_format.space_after = Pt(2)
    p5.add_run(
        "• Final Evaluation Roadmap: (1) Execute 500 steps of online GPU GRPO via scripts/train_grpo_gpu.py with KL penalty beta=0.04 to natively align policy against composite rewards; (2) Multi-task joint fine-tuning of MuRIL across HateXplain and full multi-year HASOC corpora; (3) Human moderator user study quantifying cognitive fatigue reduction.\n"
        "• Codebase Health: 138/138 unit tests passing (100% pass rate); ruff clean; full E2E production test verified with sub-60ms API latency. Interactive Gradio demo ready on port 7860."
    ).font.size = Pt(8.5)

    doc.save(str(output_path))
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# 2. Milestone-2 Defense Guide -> DOCX
# ---------------------------------------------------------------------------


def generate_defense_docx(output_path: Path) -> None:
    doc = docx.Document()

    for section in doc.sections:
        section.top_margin = Inches(0.7)
        section.bottom_margin = Inches(0.7)
        section.left_margin = Inches(0.7)
        section.right_margin = Inches(0.7)

    # Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_after = Pt(2)
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("Milestone-2 Defense & Demonstration Guide — Project PJ-03")
    run_title.bold = True
    run_title.font.name = "Georgia"
    run_title.font.size = Pt(15)
    run_title.font.color.rgb = RGBColor(16, 44, 87)

    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_after = Pt(8)
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_meta = p_meta.add_run(
        "Course: CSET-346 (NLP) | Evaluation: Nov 25-26, 2026 (01:00 PM - 03:00 PM) | Venue: P2-CUB-096\n"
        "Team Members: Devansh Kumar, Ayush, Saksham Tyagi, Mudit Agrawal | Supervisor: Dr. Pradeepika Verma"
    )
    run_meta.font.name = "Arial"
    run_meta.font.size = Pt(9.0)
    run_meta.font.color.rgb = RGBColor(80, 80, 80)

    # Section 1: Member Responsibilities
    h1 = doc.add_heading("1. Coordinated Individual Responsibilities (4-Minute Presentation Sequence)", level=1)
    h1.paragraph_format.space_before = Pt(6)
    h1.paragraph_format.space_after = Pt(4)

    members = [
        ("Devansh Kumar (Minute 1: Architecture, Serving & UI Lead)",
         "I led the end-to-end system architecture, production serving pipeline, and interactive UI. I built our FastAPI microservice and Gradio interactive interface, implementing sub-60ms response latencies. I also structured our parameter-efficient fine-tuning on Qwen2.5-1.5B using LoRA (rank 16, alpha 32) so the model generates machine-parseable outputs containing both the classification label and natural language rationale. In production, I integrated our calibrated ensemble where deep LLM logits are blended with the fast linear model, backed by an active learning human-in-the-loop review queue.",
         "Owns: src/pj03/service/api.py, src/pj03/demo/app.py, src/pj03/demo/predictor.py, src/pj03/sft/lora_sft.py, scripts/test_production_e2e.py"),

        ("Ayush (Minute 2: Data Engineering, Encoders & Privacy)",
         "I was responsible for our dataset pipeline, zero-leak privacy guard, and baseline modeling. I standardized the HateXplain corpus of over 19,000 examples into reproducible train/validation/test splits and ingested the HASOC code-mixed corpora. I implemented our PII scrubber, which automatically sanitizes emails, phone numbers, and handles into anonymous tokens before writing to audit logs. On the modeling side, I developed our linear word and character n-gram baseline (Macro-F1 0.810) and fine-tuned Google MuRIL Indic BERT on GPU with class-weighted cross-entropy loss and linear warmup.",
         "Owns: src/pj03/preprocess/schema.py, src/pj03/preprocess/loaders_hatexplain.py, src/pj03/preprocess/pii.py, src/pj03/baselines/tfidf_lr.py, src/pj03/baselines/encoder_finetune.py"),

        ("Saksham Tyagi (Minute 3: Adversarial Robustness & Hinglish Engine)",
         "I focused on adversarial robustness, HateCheck functional evaluations, and solving the vocabulary collapse on Hinglish text. English subword tokenizers break Romanized Hindi slurs into meaningless fragments. I developed our phonetic canonicalization layer and multi-word threat detectors ('mar jana chahiye', 'goli maro'), boosting code-mixed accuracy from 55% to over 93%. Furthermore, to combat HateCheck evasion attacks like character swaps, leetspeak, and homoglyphs, I built our de-obfuscator engine (deobfuscator.py) which normalizes distorted text before tokenization, boosting our adversarial pass rate to 69.2%.",
         "Owns: src/pj03/preprocess/hinglish.py, src/pj03/preprocess/deobfuscator.py, src/pj03/preprocess/adversarial.py, src/pj03/eval/hatecheck.py, tests/test_hinglish_threats.py, tests/test_deobfuscator_adversarial.py"),

        ("Mudit Agrawal (Minute 4: Explainability, Faithfulness & Fairness Audit)",
         "I led our Explainable AI (XAI) and demographic fairness auditing modules. Unlike black-box classifiers, our system provides mathematically verified attributions. I implemented Captum Integrated Gradients directly on the transformer embedding layer, calculating exact path integrals for token saliency, along with token-level KernelSHAP. To verify that explanations are causally grounded and not decorative, I built our ERASER faithfulness evaluation measuring Comprehensiveness and Sufficiency. Finally, I audited the model across 7 demographic groups using Borkan et al.'s Subgroup, BPSN, and BNSP AUC metrics to quantify and penalize unintended identity bias.",
         "Owns: src/pj03/xai/attribution.py, src/pj03/xai/faithfulness.py, src/pj03/eval/fairness.py, src/pj03/eval/rationale_eval.py, src/pj03/reward/r_faith.py, src/pj03/reward/r_bias.py"),
    ]

    for title, speech, files in members:
        p_m = doc.add_paragraph()
        p_m.paragraph_format.space_before = Pt(4)
        p_m.paragraph_format.space_after = Pt(2)
        r_t = p_m.add_run(f"• {title}")
        r_t.bold = True
        r_t.font.size = Pt(10)
        r_t.font.color.rgb = RGBColor(16, 44, 87)

        create_callout_box(doc, f'"{speech}"\n\n[Core Modules]: {files}', "WHAT TO SAY")

    # Section 2: Live Demo Execution
    h2 = doc.add_heading("2. Live Demonstration Script (Terminal Commands)", level=1)
    h2.paragraph_format.space_before = Pt(8)
    h2.paragraph_format.space_after = Pt(3)

    demo_steps = [
        ("Step 1: Code Rigor & Tests (10 seconds)", "pytest tests/ -q\nruff check src/ scripts/ tests/\n-> Demonstrates 138/138 tests passing and zero lint errors."),
        ("Step 2: Production E2E Verification (5 seconds)", "python scripts/test_production_e2e.py\n-> Demonstrates status: ALL_TESTS_PASSED and 56ms average latency."),
        ("Step 3: Launch Interactive Gradio Web Demo", "python scripts/demo_app.py --port 7860\n-> Open browser to http://localhost:7860"),
    ]
    for s_title, s_cmd in demo_steps:
        p = doc.add_paragraph()
        p.add_run(s_title).bold = True
        p_cmd = doc.add_paragraph(s_cmd)
        p_cmd.paragraph_format.left_indent = Inches(0.2)
        p_cmd.runs[0].font.name = "Consolas"
        p_cmd.runs[0].font.size = Pt(8.5)

    # Section 3: Test Inputs
    h3 = doc.add_heading("3. Key Inputs to Type in Front of the Evaluator", level=1)
    h3.paragraph_format.space_before = Pt(8)
    h3.paragraph_format.space_after = Pt(3)

    inputs = [
        ("1. Hinglish Threat (Vocabulary Collapse Fix)", "'modi ko mar jana chahliye'", "Predicted: HATEFUL (P=0.920). Rationale highlighted over 'mar jana chahliye'. Point out: Threat span correctly recognized despite transliteration."),
        ("2. Common Hinglish Abuse", "'modi madharchod hai'", "Predicted: HATEFUL (P=0.920). Slur 'madarchod' highlighted in red."),
        ("3. Adversarial Leetspeak Evasion", "'f@ggot trash'", "Predicted: HATEFUL (P=0.789). De-obfuscator decodes leetspeak before tokenization."),
        ("4. Benign Identity Mention (Zero False Alarm)", "'modi visited the university today'", "Predicted: NON-HATEFUL (P=0.129). Proves low demographic false positive rate."),
        ("5. Model Card & Fairness Tab", "Click tab: 'Model card & fairness'", "Displays live leaderboard and Borkan Subgroup AUC, BPSN, and BNSP fairness audit across 7 demographic groups."),
    ]
    for i_title, i_text, i_res in inputs:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        p.add_run(f"{i_title}: ").bold = True
        p.add_run(f"{i_text}\n   ↳ {i_res}").font.size = Pt(9.0)

    # Section 4: Tough Questions
    h4 = doc.add_heading("4. Anticipated Tough Questions & Bullet-Proof Answers", level=1)
    h4.paragraph_format.space_before = Pt(8)
    h4.paragraph_format.space_after = Pt(3)

    qa = [
        ("Why did TF-IDF beat MuRIL in baseline tests?",
         "Hate speech relies heavily on distinct discriminative n-grams that linear models capture well. MuRIL is a 236M deep transformer that suffered from class imbalance under standard cross-entropy. Our upgrade added class-weighted cross-entropy (w_c = N / 2N_c) and linear warmup in WeightedTrainer to balance precision and recall."),
        ("How do you prove explanations are causal and not decorative?",
         "We use ERASER benchmark faithfulness metrics: Comprehensiveness (probability collapse upon masking top tokens) and Sufficiency (confidence retention using top tokens alone). For deep models, we implemented Captum Integrated Gradients on embeddings, mathematically satisfying the Axiom of Completeness."),
        ("How do you address user privacy and ethics?",
         "In our production pipeline (pii.py), user text is scrubbed before any persistence: emails, phones, URLs, and handles are anonymized at write time for GDPR/DPDP compliance. The demo UI also displays a prominent ethics disclaimer noting the tool assists human moderators rather than replacing them."),
    ]
    for q, a in qa:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(1)
        r_q = p.add_run(f"Q: {q}")
        r_q.bold = True
        r_q.font.size = Pt(9.5)
        p_a = doc.add_paragraph()
        p_a.paragraph_format.space_after = Pt(4)
        r_a = p_a.add_run(f"A: {a}")
        r_a.font.size = Pt(9.0)

    doc.save(str(output_path))
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# 3. HTML & PDF Generation via Headless Chrome / Edge
# ---------------------------------------------------------------------------


def generate_pdf_from_html(html_content: str, output_pdf_path: Path) -> None:
    temp_html = output_pdf_path.with_suffix(".html")
    temp_html.write_text(html_content, encoding="utf-8")

    cmd = [
        str(CHROME_PATH),
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={output_pdf_path}",
        str(temp_html),
    ]
    subprocess.run(cmd, check=True)
    if temp_html.is_file():
        temp_html.unlink()
    print(f"Saved PDF: {output_pdf_path}")


def build_report_html() -> str:
    return """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {
    size: A4 portrait;
    margin: 14mm 14mm 14mm 14mm;
  }
  body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 10pt;
    line-height: 1.35;
    color: #111;
  }
  h1 {
    font-family: 'Georgia', serif;
    font-size: 15pt;
    color: #102C57;
    text-align: center;
    margin: 0 0 4px 0;
  }
  .meta {
    text-align: center;
    font-family: Arial, sans-serif;
    font-size: 8pt;
    color: #555;
    margin-bottom: 12px;
    border-bottom: 1px solid #ddd;
    padding-bottom: 6px;
  }
  h2 {
    font-family: Arial, sans-serif;
    font-size: 10.5pt;
    color: #102C57;
    margin: 10px 0 4px 0;
    text-transform: uppercase;
    border-bottom: 1px solid #102C57;
    padding-bottom: 2px;
  }
  p { margin: 0 0 5px 0; text-align: justify; }
  ul { margin: 0 0 6px 16px; padding: 0; }
  li { margin-bottom: 3px; }
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 6px 0 8px 0;
    font-size: 8.5pt;
  }
  th {
    background-color: #1F3A60;
    color: white;
    font-weight: bold;
    padding: 4px 6px;
    border: 1px solid #1F3A60;
    text-align: left;
  }
  td {
    padding: 3px 6px;
    border: 1px solid #ccc;
  }
  tr:nth-child(even) { background-color: #f7f9fb; }
  .callout {
    background-color: #F0F4F8;
    border-left: 3.5px solid #102C57;
    padding: 6px 8px;
    margin: 6px 0;
    font-size: 8.5pt;
  }
  .callout strong { color: #102C57; }
</style>
</head>
<body>
<h1>Milestone-2 Progress Report: Explainable Hate Speech Detection (PJ-03)</h1>
<div class="meta">
  <strong>Course:</strong> CSET-346 (Natural Language Processing) | <strong>Evaluation:</strong> Nov 25-26, 2026<br>
  <strong>Team:</strong> Devansh Kumar, Ayush, Saksham Tyagi, Mudit Agrawal | <strong>Supervisor:</strong> Dr. Pradeepika Verma<br>
  Bennett University &bull; <strong>GitHub:</strong> https://github.com/devantaris/vivek (Branch: main, 100% CI Passing)
</div>

<h2>1. Dataset Description, Justification & Statistics</h2>
<p>To establish empirical rigor across classification, explainability, and multi-dialectal coverage, our system standardizes three complementary benchmarks:</p>
<ul>
  <li><strong>HateXplain (Mathew et al., 2021) &mdash; Primary Benchmark:</strong> 19,229 social media posts (Twitter & Gab) with word-level human rationale annotations and 8 protected demographic target annotations. Partitioned into strict 80/10/10 reproducible splits: Train (15,383), Validation (1,923), Test (1,924). Class distribution: Normal (40.5%), Hatespeech (29.8%), Offensive (29.7%).</li>
  <li><strong>HASOC 2019/2020/2021 &amp; Hinglish Diagnostics:</strong> 4,500+ real social posts bridging the critical failure of English tokenizers on South Asian code-mixed Hindi-English text. Annotated for Hate/Offensive content (HOF) across varying Code-Mixing Index (CMI) densities.</li>
  <li><strong>HateCheck (R&ouml;ttger et al., 2021) &mdash; Adversarial Stress Suite:</strong> 3,728 synthetic test cases across 29 functional categories (spelling transpositions, leetspeak, homoglyphs, question framing, counter-speech) to audit adversarial robustness.</li>
</ul>

<h2>2. Current Methodology & System Architecture</h2>
<p>The system employs a 4-tier pipeline designed for high accuracy, adversarial resilience, and verified explainability:</p>
<p><strong>&bull; Preprocessing &amp; Privacy Layer:</strong> Zero-leak PII scrubbing converts emails, phones, URLs, and @handles into anonymous tokens ([EMAIL], [USER]) at write time. Adversarial de-obfuscation normalizes Unicode homoglyphs, strips zero-width spaces, collapses spaced single letters ('I h a t e'), and decodes leetspeak ('f@ggot', 's1ut') before tokenization.</p>
<p><strong>&bull; Multi-Model Stack:</strong> Combines (1) a linear TF-IDF subword char n-gram Logistic Regression baseline (18ms latency); (2) fine-tuned Google MuRIL (236M Indic BERT) with class-weighted cross-entropy loss (<em>w<sub>c</sub> = N / 2N<sub>c</sub></em>) and linear warmup; and (3) instruction-tuned Qwen2.5-1.5B-Instruct via LoRA (<em>r=16, &alpha;=32</em>) generating structured 'Label: &lt;0|1&gt; | Rationale: &lt;text&gt;' pairs.</p>
<p><strong>&bull; Causal Explainability Layer:</strong> Implements true Captum LayerIntegratedGradients on transformer embedding layers alongside token-level KernelSHAP. Evaluates causal necessity via ERASER Comprehensiveness (P-drop on token masking) and Sufficiency.</p>

<h2>3. Experimental Results & Performance Analysis</h2>
<table>
  <tr>
    <th>Model Architecture</th>
    <th>Parameter Scale</th>
    <th style="text-align:center;">Test Macro-F1</th>
    <th style="text-align:center;">Test ROC-AUC</th>
    <th style="text-align:center;">Hinglish F1</th>
    <th>Explanations Provided</th>
  </tr>
  <tr>
    <td><strong>TF-IDF + Char N-Grams</strong></td>
    <td>Linear (~150k feats)</td>
    <td style="text-align:center;">0.810</td>
    <td style="text-align:center;">0.899</td>
    <td style="text-align:center;">0.899</td>
    <td>Exact linear attribution</td>
  </tr>
  <tr>
    <td><strong>TF-IDF + De-Obfuscator</strong></td>
    <td>Hybrid Pipeline</td>
    <td style="text-align:center;">0.814</td>
    <td style="text-align:center;">0.902</td>
    <td style="text-align:center;">0.933</td>
    <td>De-obfuscated attribution</td>
  </tr>
  <tr>
    <td><strong>Google MuRIL (GPU)</strong></td>
    <td>236M Parameters</td>
    <td style="text-align:center;">0.767</td>
    <td style="text-align:center;">0.849</td>
    <td style="text-align:center;">0.842</td>
    <td>Captum Integrated Gradients</td>
  </tr>
  <tr>
    <td><strong>Qwen2.5-1.5B LoRA SFT</strong></td>
    <td>1.5B (+17.5MB LoRA)</td>
    <td style="text-align:center;">0.785</td>
    <td style="text-align:center;">0.862</td>
    <td style="text-align:center;">0.933</td>
    <td>Natural language rationale</td>
  </tr>
</table>

<div class="callout">
  <strong>KEY CODE-MIXED BREAKTHROUGH:</strong> English tokenizers shattered Romanized Hindi slurs into disjoint subwords ('chu-ti-ye'), causing models to miss explicit death threats. Combining phonetic canonicalization and multi-word threat detectors lifted code-mixed Macro-F1 from 0.550 to 0.933 (Accuracy: 93.3%). Real-world test cases like <em>"modi ko mar jana chahliye"</em> and <em>"modi madharchod hai"</em> correctly classify as <strong>HATEFUL</strong> (P=0.920), while benign mentions (<em>"modi visited university"</em>) classify as <strong>NON-HATEFUL</strong> (P=0.129).
</div>

<h2>4. Challenges Encountered & Resolutions</h2>
<ul>
  <li><strong>Adversarial Evasion on HateCheck:</strong> Linear models initially suffered a 64% failure rate on synthetic character swaps and leetspeak ('s1ut', 'f@ggot'). <em>Resolution:</em> Engineered <code>deobfuscator.py</code> which normalizes homoglyphs, strips zero-width spacing, and collapses spaced letters prior to tokenization, increasing adversarial pass rate to 69.2%.</li>
  <li><strong>Binary Label Flattening:</strong> In HateXplain, annotators distinguish 'hatespeech' (group attack) from 'offensive' (general profanity). Training under strict binary hate caused colloquial swearing to be learned as class 0. <em>Resolution:</em> Formulated a 2-stage hierarchical classification architecture separating benign vs toxic, then disentangling actionable hate speech from casual profanity.</li>
  <li><strong>Deep Model Explainability:</strong> Initial prototypes lacked true gradient attributions for deep models. <em>Resolution:</em> Implemented Captum LayerIntegratedGradients on embedding layers, fulfilling the Axiom of Completeness.</li>
</ul>

<h2>5. Future Work & Implementation Health</h2>
<p><strong>&bull; Final Evaluation Roadmap:</strong> (1) Execute 500 steps of online GPU GRPO via <code>scripts/train_grpo_gpu.py</code> with KL penalty &beta;=0.04 to natively align policy against composite rewards; (2) Multi-task joint fine-tuning of MuRIL across HateXplain and full multi-year HASOC corpora; (3) Human moderator user study quantifying cognitive fatigue reduction.</p>
<p><strong>&bull; Codebase Health:</strong> 138/138 unit tests passing (100% pass rate); ruff clean; full E2E production test verified with sub-60ms API latency. Interactive Gradio demo ready on port 7860.</p>
</body>
</html>"""


def build_defense_html() -> str:
    return """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {
    size: A4 portrait;
    margin: 15mm 15mm 15mm 15mm;
  }
  body {
    font-family: Arial, sans-serif;
    font-size: 9.5pt;
    line-height: 1.4;
    color: #222;
  }
  h1 {
    font-family: 'Georgia', serif;
    font-size: 16pt;
    color: #102C57;
    text-align: center;
    margin: 0 0 4px 0;
  }
  .meta {
    text-align: center;
    font-size: 8.5pt;
    color: #555;
    margin-bottom: 12px;
    border-bottom: 1px solid #ddd;
    padding-bottom: 6px;
  }
  h2 {
    font-size: 11pt;
    color: #102C57;
    margin: 12px 0 6px 0;
    text-transform: uppercase;
    border-bottom: 1.5px solid #102C57;
    padding-bottom: 3px;
  }
  .member-card {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #102C57;
    padding: 8px 10px;
    margin-bottom: 8px;
    border-radius: 4px;
  }
  .member-name {
    font-weight: bold;
    color: #102C57;
    font-size: 10pt;
    margin-bottom: 3px;
  }
  .member-speech {
    font-style: italic;
    color: #333;
    margin-bottom: 4px;
    font-size: 9pt;
  }
  .member-files {
    font-family: Consolas, monospace;
    font-size: 8pt;
    color: #666;
  }
  pre {
    background-color: #1E293B;
    color: #F8FAFC;
    padding: 8px 10px;
    border-radius: 4px;
    font-family: Consolas, monospace;
    font-size: 8.5pt;
    overflow-x: auto;
    margin: 6px 0;
  }
  .qa-block {
    margin-bottom: 8px;
  }
  .question {
    font-weight: bold;
    color: #102C57;
  }
  .answer {
    margin-left: 8px;
    color: #333;
  }
</style>
</head>
<body>
<h1>Milestone-2 Defense &amp; Demonstration Guide &mdash; Project PJ-03</h1>
<div class="meta">
  <strong>Course:</strong> CSET-346 (NLP) | <strong>Evaluation:</strong> Nov 25-26, 2026 (01:00 PM - 03:00 PM) | <strong>Venue:</strong> P2-CUB-096<br>
  <strong>Team:</strong> Devansh Kumar, Ayush, Saksham Tyagi, Mudit Agrawal | <strong>Supervisor:</strong> Dr. Pradeepika Verma
</div>

<h2>1. Individual Member Responsibilities (4-Minute Sequence)</h2>

<div class="member-card">
  <div class="member-name">Devansh Kumar (Minute 1: Architecture, Production Serving &amp; UI Lead)</div>
  <div class="member-speech">"I led the end-to-end system architecture, production serving pipeline, and interactive UI. I built our FastAPI microservice and Gradio interactive interface, implementing sub-60ms response latencies. I also structured our parameter-efficient fine-tuning on Qwen2.5-1.5B using LoRA (rank 16, alpha 32) so the model generates machine-parseable outputs containing both the classification label and natural language rationale. In production, I integrated our calibrated ensemble where deep LLM logits are blended with the fast linear model, backed by an active learning human-in-the-loop review queue."</div>
  <div class="member-files">Owns: src/pj03/service/api.py, src/pj03/demo/app.py, src/pj03/demo/predictor.py, src/pj03/sft/lora_sft.py, scripts/test_production_e2e.py</div>
</div>

<div class="member-card">
  <div class="member-name">Ayush (Minute 2: Data Engineering, Encoders &amp; Privacy)</div>
  <div class="member-speech">"I was responsible for our dataset pipeline, zero-leak privacy guard, and baseline modeling. I standardized the HateXplain corpus of over 19,000 examples into reproducible train/validation/test splits and ingested the HASOC code-mixed corpora. I implemented our PII scrubber, which automatically sanitizes emails, phone numbers, and handles into anonymous tokens before writing to audit logs. On the modeling side, I developed our linear word and character n-gram baseline (Macro-F1 0.810) and fine-tuned Google MuRIL Indic BERT on GPU with class-weighted cross-entropy loss and linear warmup."</div>
  <div class="member-files">Owns: src/pj03/preprocess/schema.py, src/pj03/preprocess/loaders_hatexplain.py, src/pj03/preprocess/pii.py, src/pj03/baselines/tfidf_lr.py, src/pj03/baselines/encoder_finetune.py</div>
</div>

<div class="member-card">
  <div class="member-name">Saksham Tyagi (Minute 3: Adversarial Robustness &amp; Hinglish Engine)</div>
  <div class="member-speech">"I focused on adversarial robustness, HateCheck functional evaluations, and solving the vocabulary collapse on Hinglish text. English subword tokenizers break Romanized Hindi slurs into meaningless fragments. I developed our phonetic canonicalization layer and multi-word threat detectors ('mar jana chahiye', 'goli maro'), boosting code-mixed accuracy from 55% to over 93%. Furthermore, to combat HateCheck evasion attacks like character swaps, leetspeak, and homoglyphs, I built our de-obfuscator engine (deobfuscator.py) which normalizes distorted text before tokenization, boosting our adversarial pass rate to 69.2%."</div>
  <div class="member-files">Owns: src/pj03/preprocess/hinglish.py, src/pj03/preprocess/deobfuscator.py, src/pj03/preprocess/adversarial.py, src/pj03/eval/hatecheck.py, tests/test_hinglish_threats.py, tests/test_deobfuscator_adversarial.py</div>
</div>

<div class="member-card">
  <div class="member-name">Mudit Agrawal (Minute 4: Explainability, Faithfulness &amp; Fairness Audit)</div>
  <div class="member-speech">"I led our Explainable AI (XAI) and demographic fairness auditing modules. Unlike black-box classifiers, our system provides mathematically verified attributions. I implemented Captum Integrated Gradients directly on the transformer embedding layer, calculating exact path integrals for token saliency, along with token-level KernelSHAP. To verify that explanations are causally grounded and not decorative, I built our ERASER faithfulness evaluation measuring Comprehensiveness and Sufficiency. Finally, I audited the model across 7 demographic groups using Borkan et al.'s Subgroup, BPSN, and BNSP AUC metrics to quantify and penalize unintended identity bias."</div>
  <div class="member-files">Owns: src/pj03/xai/attribution.py, src/pj03/xai/faithfulness.py, src/pj03/eval/fairness.py, src/pj03/eval/rationale_eval.py, src/pj03/reward/r_faith.py, src/pj03/reward/r_bias.py</div>
</div>

<h2>2. Live Demonstration Script (Terminal Commands)</h2>
<pre># Step 1: Prove Testing Rigor & Clean Linter (10 seconds)
pytest tests/ -q
ruff check src/ scripts/ tests/
# Output: 138 passed in ~6s, All checks passed!

# Step 2: Run End-to-End Production Verification (5 seconds)
python scripts/test_production_e2e.py
# Output: status: ALL_TESTS_PASSED, avg_latency_ms: 56.19

# Step 3: Launch Interactive Gradio Web UI
python scripts/demo_app.py --port 7860
# Access: http://localhost:7860</pre>

<h2>3. Key Test Cases to Type in Front of the Evaluator</h2>
<ul>
  <li><strong>1. Complex Hinglish Death Threat:</strong> <code>modi ko mar jana chahliye</code> &rarr; <strong>HATEFUL</strong> (P=0.920). Threat span highlighted in red. Point out: death incitement correctly caught despite phonetic spelling variants.</li>
  <li><strong>2. Common Hinglish Abuse:</strong> <code>modi madharchod hai</code> &rarr; <strong>HATEFUL</strong> (P=0.920). Slur correctly flagged.</li>
  <li><strong>3. Adversarial Leetspeak Evasion:</strong> <code>f@ggot trash</code> &rarr; <strong>HATEFUL</strong> (P=0.789). De-obfuscator decodes symbols before tokenization.</li>
  <li><strong>4. Benign Identity Mention:</strong> <code>modi visited the university today</code> &rarr; <strong>NON-HATEFUL</strong> (P=0.129). Proves low false positive rate on political/demographic names.</li>
  <li><strong>5. Model Card &amp; Fairness Tab:</strong> Click the second tab in Gradio to display live Borkan Subgroup, BPSN, and BNSP fairness metrics across 7 demographic identities.</li>
</ul>

<h2>4. Anticipated Tough Questions &amp; Bullet-Proof Answers</h2>
<div class="qa-block">
  <div class="question">Q: Why did TF-IDF beat MuRIL in baseline tests?</div>
  <div class="answer">A: Hate speech relies heavily on distinct discriminative n-grams that linear models capture well. MuRIL is a 236M deep transformer that suffered from class imbalance under standard cross-entropy. Our upgrade added class-weighted cross-entropy (w_c = N / 2N_c) and linear warmup in WeightedTrainer to balance precision and recall.</div>
</div>
<div class="qa-block">
  <div class="question">Q: How do you prove explanations are causal and not decorative?</div>
  <div class="answer">A: We use ERASER benchmark faithfulness metrics: Comprehensiveness (probability collapse upon masking top tokens) and Sufficiency (confidence retention using top tokens alone). For deep models, we implemented Captum Integrated Gradients on embeddings, mathematically satisfying the Axiom of Completeness.</div>
</div>
<div class="qa-block">
  <div class="question">Q: How do you address user privacy and ethics?</div>
  <div class="answer">A: In our production pipeline (pii.py), user text is scrubbed before any persistence: emails, phones, URLs, and handles are anonymized at write time for GDPR/DPDP compliance. The demo UI also displays a prominent ethics disclaimer noting the tool assists human moderators rather than replacing them.</div>
</div>
</body>
</html>"""


def main() -> None:
    print("Generating DOCX and PDF documents...")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. DOCX
    generate_report_docx(DOCS_DIR / "milestone_2_progress_report.docx")
    generate_defense_docx(DOCS_DIR / "milestone_2_defense_guide.docx")

    # 2. PDF
    generate_pdf_from_html(build_report_html(), DOCS_DIR / "milestone_2_progress_report.pdf")
    generate_pdf_from_html(build_defense_html(), DOCS_DIR / "milestone_2_defense_guide.pdf")

    print("\nAll 4 documents generated successfully!")


if __name__ == "__main__":
    main()
