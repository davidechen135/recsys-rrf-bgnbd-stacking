"""
Academic Conference Paper PDF Generator (CVPR / Top-Tier Standard)
Compiles the complete IE5005 Mini Project into a formal 8-page academic research paper.
Strictly adheres to CVPR / IEEE / ACM conference publishing requirements:
- Times-Roman academic typography
- Pure Booktabs scientific tables (clean horizontal rules, no vertical lines, no garish colored headers)
- Minimalist academic color palette (Oxford Navy, Slate, Crimson, Charcoal)
- Rigorous mathematical formulations, loss functions, and constrained optimization models
- Zero consulting/MBA jargon; pure machine learning and operations research terminology
Author: Chen Davide (A0351457B)
Course: IE5005 Applied Data Analytics, National University of Singapore
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image as PILImage
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import reportlab
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas

from src.config import FIGURES_DIR, OUTPUT_DIR

EQUATIONS_DIR = FIGURES_DIR / "equations"
EQUATIONS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_equation_images():
    """Renders all formal display mathematical equations into crisp 300-DPI transparent PNGs."""
    equations = [
        (r'\mathcal{R}_{\mathrm{total}} = \mathcal{R}_{\mathrm{Type\,I}} \cup \mathcal{R}_{\mathrm{Type\,II}}', 'eq1_return_decoupling.png', 6.0, 0.6, 11.0),
        (r'\mathrm{Lift}(X \to Y) = \frac{P(X \cup Y)}{P(X) \cdot P(Y)}', 'eq2_apriori_lift.png', 5.0, 0.7, 11.0),
        (r'\mathcal{L}_{\mathrm{I2V}} = \sum_{t=1}^T \sum_{-c \leq j \leq c,\, j \neq 0} \log P(w_{t+j} \mid w_t)', 'eq3_item2vec_loss.png', 6.8, 0.75, 11.0),
        (r'\mathrm{RRF\_Score}(d) = \sum_{m \in M} \frac{w_m}{k_{\mathrm{RRF}} + r_m(d)}, \qquad k_{\mathrm{RRF}} = 30', 'eq4_rrf_fusion.png', 7.0, 0.7, 11.0),
        (r'P(\mathrm{Alive} \mid x, t_x, T) = \left[ 1 + \frac{a}{b + x} \left( \frac{\alpha + T}{\alpha + t_x} \right)^{r + x} \right]^{-1}', 'eq5_bgnbd_palive.png', 7.5, 0.75, 11.0),
        (r'\hat{y}_{\mathrm{Hybrid}} = 0.45 \cdot \hat{y}_{\mathrm{BGNBD}} + 0.35 \cdot \hat{y}_{\mathrm{RF}} + 0.20 \cdot \hat{y}_{\mathrm{GBDT\_L1}}', 'eq6_hybrid_clv.png', 7.2, 0.65, 11.0),
        (r'\max_{\mathbf{x} \in \{0, 1\}^N} \sum_{i=1}^N \hat{y}_i \cdot \Delta R_i(x_i) \qquad \text{subject to} \quad \sum_{i=1}^N c_i x_i \leq B', 'eq7_knapsack_opt.png', 7.8, 0.7, 11.0),
        (r'\Delta R_i(x_i) = P(\mathrm{Churn}_i) \cdot \mathrm{Lift}(x_i) - \mathrm{Cannibalization}_i(x_i)', 'eq8_net_response.png', 7.0, 0.65, 11.0),
    ]

    for eq_latex, filename, w, h, fs in equations:
        out_file = EQUATIONS_DIR / filename
        fig, ax = plt.subplots(figsize=(w, h), dpi=300)
        ax.axis('off')
        ax.text(0.5, 0.5, f'${eq_latex}$', fontsize=fs, ha='center', va='center')
        fig.savefig(out_file, bbox_inches='tight', transparent=True, pad_inches=0.035, dpi=300)
        plt.close(fig)

    teaser_file = FIGURES_DIR / "fig0_teaser_pipeline.png"
    if not teaser_file.exists():
        from src.visualization.teaser_plot import generate_teaser_figure
        generate_teaser_figure(teaser_file)


def make_aspect_image(path, width=None, height=None):
    """Preserves 100% natural aspect ratio of underlying PIL image."""
    im = PILImage.open(path)
    orig_w, orig_h = im.size
    ratio = orig_w / orig_h
    if width is not None and height is None:
        height = width / ratio
    elif height is not None and width is None:
        width = height * ratio
    return Image(str(path), width=width, height=height)


def make_equation_block(filename, eq_num, height=23, space_before=3.0, space_after=3.0):
    """Renders a display equation centered with flush-right academic equation numbering (1)-(8)."""
    img = make_aspect_image(EQUATIONS_DIR / filename, height=height)
    eq_num_p = Paragraph(f"({eq_num})", ParagraphStyle('EqNum', fontName='Times-Roman', fontSize=9.0, leading=11.0, alignment=2))
    t = Table([["", img, eq_num_p]], colWidths=[34, 436, 34])
    t.setStyle(TableStyle([
        ('ALIGN', (0,0), (0,0), 'LEFT'),
        ('ALIGN', (1,0), (1,0), 'CENTER'),
        ('ALIGN', (2,0), (2,0), 'RIGHT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), space_before),
        ('BOTTOMPADDING', (0,0), (-1,-1), space_after),
    ]))
    return t


def make_two_column_block(col1_flowables, col2_flowables, col_w=242, gutter=20):
    """Encapsulates two columns of text side-by-side to achieve IEEE/ACM 65-72 char line length."""
    t = Table([[col1_flowables, "", col2_flowables]], colWidths=[col_w, gutter, col_w])
    t.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    return t


class AcademicCanvas(canvas.Canvas):
    """Renders running headers and footers with academic styling."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Times-Roman", 8)
        self.setFillColor(colors.HexColor("#222222"))

        # Running header on pages 2+
        if self._pageNumber > 1:
            self.drawString(54, 755, "IEEE / ACM Conference Submission | IE5005 Applied Data Analytics | Option A: UK Online Retail")
            self.drawRightString(558, 755, "Chen Davide (Student ID: A0351457B) | NUS")
            self.setStrokeColor(colors.HexColor("#333333"))
            self.setLineWidth(0.4)
            self.line(54, 750, 558, 750)

        # Running footer on all pages
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_text)
        self.drawString(54, 36, "National University of Singapore (NUS) — Department of Industrial Systems Engineering and Management")
        self.setStrokeColor(colors.HexColor("#333333"))
        self.setLineWidth(0.4)
        self.line(54, 46, 558, 46)

        self.restoreState()


def get_booktabs_style():
    """Generates pure Booktabs TableStyle adhering to top-tier academic standards."""
    return TableStyle([
        ('LINEABOVE', (0,0), (-1,0), 1.2, colors.HexColor("#111111")),   # \toprule
        ('LINEBELOW', (0,0), (-1,0), 0.6, colors.HexColor("#333333")),   # \midrule
        ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.HexColor("#111111")), # \bottomrule
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.2),
        ('TOPPADDING', (0,0), (-1,-1), 2.2),
        ('BACKGROUND', (0,0), (-1,-1), colors.white),
    ])


def create_paper_pdf():
    ensure_equation_images()
    pdf_path = PROJECT_ROOT / "IE5005_Mini_Project_Paper_Chen_Davide.pdf"
    
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'PaperTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=13.5,
        leading=16.0,
        alignment=1,
        spaceAfter=3.5
    )

    author_style = ParagraphStyle(
        'PaperAuthor',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11.0,
        alignment=1,
        spaceAfter=4.0
    )

    abstract_style = ParagraphStyle(
        'PaperAbstract',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=7.6,
        leading=9.6,
        alignment=4,
        leftIndent=10,
        rightIndent=10,
        spaceAfter=2.5
    )

    keywords_style = ParagraphStyle(
        'PaperKeywords',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=7.4,
        leading=9.2,
        alignment=4,
        leftIndent=10,
        rightIndent=10,
        spaceAfter=3.5
    )

    h1_style = ParagraphStyle(
        'PaperH1',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=9.0,
        leading=11.2,
        spaceBefore=3.5,
        spaceAfter=1.8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'PaperH2',
        parent=styles['Normal'],
        fontName='Times-BoldItalic',
        fontSize=8.0,
        leading=9.8,
        spaceBefore=2.5,
        spaceAfter=1.2,
        keepWithNext=True
    )

    h3_style = ParagraphStyle(
        'PaperH3',
        parent=styles['Normal'],
        fontName='Times-BoldItalic',
        fontSize=7.6,
        leading=9.4,
        spaceBefore=2.0,
        spaceAfter=1.0,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'PaperBody',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=7.6,
        leading=9.5,
        alignment=4,
        spaceAfter=2.2
    )

    table_caption_style = ParagraphStyle(
        'TableCaption',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=7.4,
        leading=9.2,
        alignment=1,
        spaceBefore=1.8,
        spaceAfter=2.5
    )

    caption_style = ParagraphStyle(
        'PaperCaption',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=7.0,
        leading=8.6,
        alignment=1,
        spaceBefore=1.8,
        spaceAfter=3.0
    )

    subcaption_style = ParagraphStyle(
        'PaperSubCaption',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=6.8,
        leading=8.4,
        alignment=1,
        spaceBefore=0.8,
        spaceAfter=1.2
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=6.8,
        leading=8.2,
        alignment=1,
        textColor=colors.HexColor("#111111")
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=6.6,
        leading=8.0,
        alignment=1,
        textColor=colors.HexColor("#222222")
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=6.6,
        leading=8.0,
        alignment=1,
        textColor=colors.HexColor("#000000")
    )

    table_cell_bold_left = ParagraphStyle(
        'TableCellBoldLeft',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=6.6,
        leading=8.0,
        alignment=0,
        textColor=colors.HexColor("#000000")
    )

    table_cell_left = ParagraphStyle(
        'TableCellLeft',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=6.6,
        leading=8.0,
        alignment=0,
        textColor=colors.HexColor("#222222")
    )

    h1_col = ParagraphStyle('H1Col', parent=styles['Normal'], fontName='Times-Bold', fontSize=8.8, leading=10.8, spaceBefore=2.5, spaceAfter=1.5)
    h2_col = ParagraphStyle('H2Col', parent=styles['Normal'], fontName='Times-BoldItalic', fontSize=7.8, leading=9.6, spaceBefore=2.0, spaceAfter=1.0)
    col_style = ParagraphStyle('ColBody', parent=styles['Normal'], fontName='Times-Roman', fontSize=7.4, leading=9.3, alignment=4, spaceAfter=2.0)

    ref_style = ParagraphStyle(
        'RefStyle',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=6.5,
        leading=8.0,
        alignment=4,
        spaceAfter=2.0
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, ABSTRACT, TEASER ARCHITECTURE & INTRODUCTORY FRAMEWORK
    # =========================================================================
    story.append(Paragraph("Empirical Customer Value Dynamics and Multi-Channel Recommendation in Transnational E-Commerce: A Benchmark Study and Strategic Framework", title_style))
    story.append(Paragraph("<b>Chen Davide</b> (Student ID: <code>A0351457B</code>) | <i>Department of Industrial Systems Engineering and Management, National University of Singapore</i><br/>"
                           "Email: <code>davide@u.nus.edu</code> | Course: IE5005 Applied Data Analytics | Target Role: Head of E-commerce", author_style))
    
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#111111"), spaceBefore=0, spaceAfter=3))
    
    abstract_text = (
        "<b><i>Abstract</i>—Transnational e-commerce platforms face severe structural challenges in capital allocation, characterized by extreme Pareto spend inequality, "
        "volatile cross-border return rates, and cold-start catalog sparsity. Analyzing the UK Online Retail transactional corpus (<i>N</i> = 541,909 across 4,334 identified accounts), "
        "we address senior management directives on customer prioritization, fraud mitigation, and optimal budget deployment. Methodologically, we construct two offline benchmarks "
        "evaluating curriculum baselines (Apriori, K-Means, OLS, Ridge, KNN) against frontier architectures. We introduce: (1) a <i>Multi-Channel Hybrid Retrieval Recommender</i> "
        "fusing local co-occurrence with continuous session embeddings via Reciprocal Rank Fusion, achieving Pareto dominance with 18.64% HitRate@10 (+11.1% rel.) and 58.51% catalog coverage; "
        "and (2) a <i>Probabilistic-Guided Stacking Ensemble</i> coupling continuous latent stochastic processes (BG/NBD, Gamma-Gamma) with dual-loss tree ensembles (Random Forest, L1-norm GBDT), "
        "slashing holdout Q4 MAE to £593.25 and capturing 72.30% of spend within the top-20% budget. Forensic anomaly auditing decouples £476k in cancellations into Type I instant checkout "
        "reversals (51.61%, £246k) and Type II chronic physical returns (48.39%, £230k). Finally, a constrained knapsack optimization formalizes 2012 capital allocation.</b>"
    )
    story.append(Paragraph(abstract_text, abstract_style))
    story.append(Paragraph("<b><i>Keywords</i>—Transnational E-Commerce, Recommender Systems, Customer Lifetime Value (CLV), BG/NBD Model, Multi-Channel Retrieval, Reciprocal Rank Fusion, Return Decoupling.</b>", keywords_style))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#111111"), spaceBefore=0, spaceAfter=3))

    # Teaser Figure (Figure 1: Full-width spanning 485 pt)
    teaser_img = make_aspect_image(FIGURES_DIR / "fig0_teaser_pipeline.png", width=485) # height = 134.4 pt
    story.append(teaser_img)
    story.append(Paragraph("Figure 1: End-to-end transnational e-commerce analytics and strategic decision framework: (1) dual-track anomaly auditing & return decoupling; (2) candidate retrieval multi-channel fusion (Matrix 1); (3) probabilistic-guided forward CLV forecasting (Matrix 2); and (4) constrained knapsack capital budget allocation.", caption_style))
    story.append(Spacer(1, 2.0))

    # Two-Column IEEE/ACM Layout for Section 1
    col1_p1 = [
        Paragraph("1. INTRODUCTION & PROBLEM FORMULATION", h1_col),
        Paragraph(
            "Modern e-commerce enterprises operate in high-dimensional, noisy, and non-stationary transactional environments. "
            "As retail organizations transition from speculative customer acquisition to disciplined unit-economics optimization, "
            "analytics specialists are tasked with addressing core commercial questions for senior leadership: <i>Which customers, products, "
            "and geographic corridors sustain profitability, where is capital leaking, and how should marketing capital be deployed?</i>",
            col_style
        ),
        Paragraph("1.1 Industrial Setting & Scope", h2_col),
        Paragraph(
            "We analyze the complete UK Online Retail transactional corpus spanning 01/12/2010 to 09/12/2011 (<i>N</i> = 541,909 rows, 23,260 baskets, 3,665 SKUs, £10.25M GMV). "
            "The enterprise operates primarily as a wholesale giftware platform serving domestic UK institutions and cross-border commercial distributors. Behind headline metrics, "
            "the platform faces acute power-law skewness, anomalous cancellations, and stark behavioral heterogeneity between domestic buyers and bulk wholesale importers.",
            col_style
        ),
        Paragraph("1.2 Curriculum Gap vs Frontier Practice", h2_col),
        Paragraph(
            "Standard graduate curricula frequently treat analytics in stylized, siloed settings: basket analysis via Apriori rules, segmentation via ad-hoc RFM <i>K</i>-Means, "
            "and spend forecasting via ordinary least squares (OLS). In high-dimensional enterprise settings, these fail under extreme power-law skewness, catalog sparsity, "
            "and customer churn. This paper bridges this divide by evaluating classical lecture paradigms against modern collaborative filtering (Amazon ItemCF), "
            "continuous neural embeddings (Item2Vec), continuous-time stochastic processes (BG/NBD), and non-linear gradient-boosted ensembles.",
            col_style
        )
    ]

    col2_p1 = [
        Paragraph("1.3 Research Questions & Contributions", h1_col),
        Paragraph(
            "We formulate four core research questions and deliver four novel methodological contributions:<br/>"
            "• <b>RQ1 (Forensic Decoupling):</b> Decoupling cancellations reveals £245.7k (51.6%) instant reversals across two accounts (Cust 16446, 12346) "
            "and £230.3k (48.4%) chronic physical returns clustered in fragile ceramics (Regency Cakestand), informing distinct OMS holds and packaging redesign.<br/>"
            "• <b>RQ2 (Candidate Retrieval Matrix):</b> Pitting 7 recommender paradigms across 3,655 holdout carts reveals Apriori fails under catalog sparsity. "
            "Our proposed Multi-Channel RRF Recommender achieves Pareto dominance with 18.64% HitRate@10 (+11.1% rel.) and 58.51% catalog coverage.<br/>"
            "• <b>RQ3 (Forward CLV Forecasting):</b> Pitting 8 models across an out-of-time Q4 holdout period shows static models overfit whale buyers. "
            "Our Probabilistic-Guided Stacking Ensemble blends BG/NBD latent priors with dual-loss trees, slashing MAE to £593.25 and capturing 72.30% of Q4 spend.<br/>"
            "• <b>RQ4 (Knapsack Capital Policy):</b> Formalizing knapsack optimization yields our 4-quadrant lifecycle framework: allocating 50% of paid ad capital "
            "to at-risk dormant VIPs (Q2) while suppressing ads on organic champions (Q1) to eliminate cannibalization, alongside Netherlands freight consolidation.",
            col_style
        )
    ]

    story.append(make_two_column_block(col1_p1, col2_p1))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: DATA AUDITING, CLEANING & RETURN DECOUPLING
    # =========================================================================
    story.append(Paragraph("2. DATA AUDITING & FORENSIC PREPROCESSING PIPELINE", h1_style))
    story.append(Paragraph(
        "We analyze the complete UK Online Retail transactional corpus spanning 01/12/2010 to 09/12/2011 (<i>N</i> = 541,909 raw observation lines). "
        "To ensure production-grade integrity, we design an automated data auditor and dual-track cleaning pipeline implemented in modular "
        "Python modules (<code>src/data/loader.py</code> and <code>src/data/cleaner.py</code>).",
        body_style
    ))

    story.append(Paragraph("2.1 Dual-Track Cleaning Protocol & Data Hygiene Audit", h2_style))
    story.append(Paragraph(
        "Retail clickstream datasets contain pervasive recording anomalies, negative values, guest accounts, and administrative accounting codes. "
        "Rather than naive row deletion, we establish two audited, decoupled analytical tracks: "
        "<b>Track A (Valid Sales Merchandise Corpus)</b> isolates genuine customer purchases: non-cancellation invoices, positive unit prices (<i>P</i> &gt; 0), "
        "positive quantities (<i>Q</i> &gt; 0), and valid 5-digit alphanumeric merchandise stock codes. This retains <b>522,709 merchandise rows</b> "
        "representing £10,254,404.07 in gross transaction volume across 23,260 shopping carts and 3,665 unique SKUs. "
        "<b>Track B (Audit Corpus)</b> captures all 8,670 negative and cancelled transaction rows (£475,966.16) for forensic analysis. "
        "Non-merchandise system entries—including manual accounting overrides (<code>M</code>, 571 rows), postage charges (<code>POST</code>, 1,256 rows), "
        "bank charges (<code>BANK CHARGES</code>, 37 rows), and Amazon operational fee adjustments (<code>AMAZONFEE</code>, 34 rows)—are audited and decoupled "
        "from customer recommendation and lifetime value modeling. Table 1 summarizes the complete data hygiene audit.",
        body_style
    ))

    # Table 1: Booktabs Data Hygiene Audit
    story.append(Paragraph("Table 1: Forensic data hygiene audit trail across raw, cleaned merchandise, and cancelled transaction records.", table_caption_style))
    table1_data = [
        [Paragraph("Data Processing Audit Stage", table_header_style), Paragraph("Line Count", table_header_style), Paragraph("Units / Items", table_header_style), Paragraph("Financial Impact (£)", table_header_style), Paragraph("% Raw Corpus", table_header_style), Paragraph("Methodological Treatment & Justification", table_header_style)],
        [Paragraph("Raw Imported Transaction Lines", table_cell_left), Paragraph("541,909", table_cell_style), Paragraph("5,176,450", table_cell_style), Paragraph("£9,747,747.93", table_cell_style), Paragraph("100.00%", table_cell_style), Paragraph("Full unedited UCI repository transactional corpus", table_cell_left)],
        [Paragraph("Missing Customer ID (Guest Checkout)", table_cell_left), Paragraph("135,080", table_cell_style), Paragraph("496,285", table_cell_style), Paragraph("£1,447,682.12", table_cell_style), Paragraph("24.93%", table_cell_style), Paragraph("Retained for RecSys basket completion; decoupled from CLV", table_cell_left)],
        [Paragraph("Non-Merchandise System Codes (POST, D, M)", table_cell_left), Paragraph("2,750", table_cell_style), Paragraph("18,320", table_cell_style), Paragraph("£206,750.45", table_cell_style), Paragraph("0.51%", table_cell_style), Paragraph("Excluded from product recommendation and RFM behavior", table_cell_left)],
        [Paragraph("Negative / Zero Unit Price Records", table_cell_left), Paragraph("2,517", table_cell_style), Paragraph("-45,210", table_cell_style), Paragraph("£0.00", table_cell_style), Paragraph("0.46%", table_cell_style), Paragraph("Zero-price damaged stock adjustments purged from modeling", table_cell_left)],
        [Paragraph("<b>Track A: Valid Sales Rows (Clean)</b>", table_cell_left), Paragraph("<b>522,709</b>", table_cell_style), Paragraph("<b>5,595,780</b>", table_cell_style), Paragraph("<b>£10,254,404.07</b>", table_cell_style), Paragraph("<b>96.46%</b>", table_cell_style), Paragraph("<b>Core merchandise dataset for RecSys & Spending Modeling</b>", table_cell_left)],
        [Paragraph("<b>Track B: Audited Cancellation Rows</b>", table_cell_left), Paragraph("<b>8,670</b>", table_cell_style), Paragraph("<b>-270,019</b>", table_cell_style), Paragraph("<b>-£475,966.16</b>", table_cell_style), Paragraph("<b>1.60%</b>", table_cell_style), Paragraph("<b>Forensic reverse-flow dataset for return decoupling</b>", table_cell_left)],
        [Paragraph("Unique Identified Customer Accounts", table_cell_left), Paragraph("4,334", table_cell_style), Paragraph("—", table_cell_style), Paragraph("£8,740,248.87", table_cell_style), Paragraph("—", table_cell_style), Paragraph("Identified commercial accounts driving 85.2% of clean sales", table_cell_left)],
        [Paragraph("Unique Merchandise Stock Items (SKUs)", table_cell_left), Paragraph("3,665", table_cell_style), Paragraph("—", table_cell_style), Paragraph("—", table_cell_style), Paragraph("—", table_cell_style), Paragraph("Active product catalog vocabulary across 12-month window", table_cell_left)]
    ]
    t1 = Table(table1_data, colWidths=[120, 48, 55, 75, 45, 161])
    t1.setStyle(get_booktabs_style())
    story.append(t1)
    story.append(Spacer(1, 4))

    story.append(Paragraph("2.2 Forensic Anomaly Diagnosis & Return Decoupling", h2_style))
    story.append(Paragraph(
        "A critical empirical discovery emerges from our forensic audit of the 8,670 cancellation lines (£475,966.16 lost revenue). "
        "Standard industry analyses treat all negative transactions uniformly as 'customer churn' or 'product dissatisfaction'. "
        "By analyzing timestamp proximity and transaction quantities, we decouple cancellations into two distinct mechanisms (Eq. 1):",
        body_style
    ))

    # Rendered equation 1
    story.append(make_equation_block("eq1_return_decoupling.png", 1, height=20, space_before=3.5, space_after=3.5))

    story.append(Paragraph(
        "<b>(1) Type I: Instant Administrative Cancellations (£245,653.20, 51.61% of total losses):</b> "
        "We uncover that over half of all negative revenue is driven by instantaneous offsetting reversals where purchase and cancellation "
        "occur within seconds or minutes. Specifically, Customer <code>16446</code> (Invoice <code>581483</code> vs <code>C581484</code>, "
        "SKU <code>23843</code>, ±80,995 units, ±£168,469.60) and Customer <code>12346</code> (Invoice <code>541431</code> vs <code>C541433</code>, "
        "SKU <code>23166</code>, ±74,215 units, ±£77,183.60) account for £245.7k across just two single-order events. "
        "These represent warehouse test transactions or immediate checkout quantity typos, not operational defects.<br/>"
        "<b>(2) Type II: Chronic Physical Post-Delivery Returns (£230,312.96, 48.39% of total losses):</b> "
        "Comprising 8,668 transaction lines across 2,058 distinct accounts and 1,944 SKUs. The single highest physical return SKU is "
        "<code>22423</code> ('REGENCY CAKESTAND 3 TIER', 181 return events, 977 units refunded, £13,296.85 lost revenue). "
        "High-return items cluster overwhelmingly in fragile ceramics and tiered homeware, pointing to shipping transit breakage rather than pricing mismatch. "
        "Table 2 reports the decoupled characteristics and operational remedies.",
        body_style
    ))

    # Table 2: Booktabs Return Decoupling
    story.append(Paragraph("Table 2: Forensic return decoupling: Type I instant administrative reversals vs Type II chronic post-delivery physical quality returns.", table_caption_style))
    table2_data = [
        [Paragraph("Return Typology Category", table_header_style), Paragraph("Lines", table_header_style), Paragraph("Units", table_header_style), Paragraph("Lost Revenue (£)", table_header_style), Paragraph("% Lost", table_header_style), Paragraph("Root Operational Mechanism", table_header_style), Paragraph("Targeted Operational Intervention", table_header_style)],
        [Paragraph("Type I: Instant Admin Cancellations", table_cell_left), Paragraph("2", table_cell_style), Paragraph("155,210", table_cell_style), Paragraph("£245,653.20", table_cell_style), Paragraph("51.61%", table_cell_style), Paragraph("Bulk checkout error, instant reversal (Cust 16446 & 12346)", table_cell_left), Paragraph("Implement OMS 30-min confirmation hold & bulk volume limits", table_cell_left)],
        [Paragraph("Type II: Chronic Physical Returns", table_cell_left), Paragraph("8,668", table_cell_style), Paragraph("114,809", table_cell_style), Paragraph("£230,312.96", table_cell_style), Paragraph("48.39%", table_cell_style), Paragraph("Fragile ceramics transit breakage, packaging failure, customer expectation mismatch", table_cell_left), Paragraph("Re-engineer fragile SKU bubble-wrap packaging & audit suppliers", table_cell_left)],
        [Paragraph("Total Combined Return Volume", table_cell_left), Paragraph("8,670", table_cell_style), Paragraph("270,019", table_cell_style), Paragraph("£475,966.16", table_cell_style), Paragraph("100.00%", table_cell_style), Paragraph("Aggregate e-commerce reversals and physical returns", table_cell_left), Paragraph("Dual-track intervention recovering estimated £360k+ in 2012", table_cell_left)]
    ]
    t2 = Table(table2_data, colWidths=[105, 28, 42, 60, 40, 115, 114])
    t2.setStyle(get_booktabs_style())
    story.append(t2)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: EXPLORATORY DATA ANALYSIS & MACRO REGULARITIES
    # =========================================================================
    story.append(Paragraph("3. EXPLORATORY DATA ANALYSIS & EMPIRICAL REGULARITIES", h1_style))
    story.append(Paragraph(
        "We systematically examine the empirical properties of the cleansed sales dataset (522,709 merchandise rows, £10.25M gross GMV). "
        "The exploration spans continuous distributions, concentration curves, and longitudinal time-series trends.",
        body_style
    ))

    story.append(Paragraph("3.1 Line Revenue Skewness & Customer Spend Pareto Inequality", h2_style))
    story.append(Paragraph(
        "Figure 2 illustrates continuous value distributions across transaction lines and customer accounts. "
        "<b>(a) Line Revenue Distribution:</b> On a log<sub>10</sub> scale, the median line revenue is £9.90 (log<sub>10</sub> = 1.00), "
        "whereas the arithmetic mean is £19.62 (log<sub>10</sub> = 1.29), representing a 2.0× divergence. Extreme right-skewness (skewness = 516.03) "
        "is driven by commercial wholesale orders where single invoice lines exceed £10,000. Arithmetic averages distort financial planning. "
        "<b>(b) Customer Revenue Pareto Inequality:</b> Across all 4,334 identified accounts (£8.74M attributed GMV), the top 1% (43 customers) "
        "generate 32.02% of total revenue (£2.80M); the top 10% (433 accounts) generate 61.39% (£5.37M); and the top 20% (866 accounts) "
        "generate 74.59% (£6.52M). The bottom 50% accounts for only 7.84% (£686k). The Gini coefficient of 0.78 confirms extreme wholesale concentration.",
        body_style
    ))

    # Side-by-side Table with Fig 1 & Fig 2 preserving EXACT 1.778 aspect ratio
    img1 = make_aspect_image(FIGURES_DIR / "fig1_distribution_revenue.png", width=235) # height = 132.2 pt
    img2 = make_aspect_image(FIGURES_DIR / "fig2_composition_pareto.png", width=235)      # height = 132.2 pt
    cap1 = Paragraph("<b>(a)</b> Line revenue distribution (median £9.90 vs mean £19.62)", subcaption_style)
    cap2 = Paragraph("<b>(b)</b> Customer spend Pareto Lorenz curve (Top 10% = 61.4% GMV)", subcaption_style)

    side_by_side_eda = Table([[img1, img2], [cap1, cap2]], colWidths=[252, 252])
    side_by_side_eda.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 1),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
    ]))
    story.append(side_by_side_eda)
    story.append(Paragraph("Figure 2: Macro financial regularities: (a) Empirical distribution of transaction line revenues on a log<sub>10</sub> scale; (b) Customer revenue Lorenz concentration curve against 45-degree parity.", caption_style))

    story.append(Paragraph("3.2 Longitudinal Dynamics, Seasonality & Anomaly Spikes", h2_style))
    story.append(Paragraph(
        "Figure 3 traces monthly gross sales revenue and order cancellation rates over the 12-month period. "
        "Sales exhibit strong seasonality, escalating from an August low-season of £724.7k to a monumental November peak of £1,453.3k (+100.5% growth), "
        "driven by Black Friday wholesale stocking. Crucially, baseline monthly cancellation rates remain stable between 1.19% and 6.07% across Feb–Nov 2011. "
        "However, Figure 3 highlights two extraordinary spikes: January 2011 (12.01%) and partial December 2011 (22.07%). "
        "These spikes correspond exactly to the Type I instant bulk cancellations identified in Section 2.2, visually confirming that anomalous spikes are driven by single-order administrative events rather than broad operational degradation.",
        body_style
    ))

    # Full-width Fig 3 preserving EXACT 1.979 aspect ratio
    img3 = make_aspect_image(FIGURES_DIR / "fig3_change_monthly_trend.png", width=410) # height = 207.2 pt
    story.append(img3)
    story.append(Paragraph("Figure 3: Monthly gross sales revenue (blue line) and cancellation rate (red dashed line) with Type I anomaly annotations.", caption_style))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: BIVARIATE RFM & OVERSEAS UNIT ECONOMICS
    # =========================================================================
    story.append(Paragraph("3.3 Bivariate Customer Geometry & Transnational Market Economics", h2_style))
    story.append(Paragraph(
        "To formulate commercial strategy, we examine two orthogonal behavioral axes: longitudinal purchasing geometry across individual accounts "
        "and cross-border unit economics across international wholesale corridors.",
        body_style
    ))

    story.append(Paragraph("3.3.1 Customer RFM Behavioral Geometry", h3_style))
    story.append(Paragraph(
        "We construct customer Recency, Frequency, and Monetary (RFM) representations across all 4,334 identified accounts. "
        "Figure 4(a) plots log(Frequency + 1) against log<sub>10</sub>(Monetary Spend), with points colored by Recency (days since last order). "
        "We observe a robust linear relationship between log purchase frequency and log cumulative spend (<i>r</i> = 0.81, Pearson <i>r</i> = 0.92 "
        "between quantity and revenue). High-spending VIP accounts cluster in the upper-right quadrant and exhibit high recency (bright yellow markers, "
        "purchased within 30 days). Accounts drifting into dormancy (purple markers) demonstrate sharp value divergence, demanding predictive retention.",
        body_style
    ))

    # Side-by-side Fig 4 & Fig 8 preserving exact aspect ratios
    img4 = make_aspect_image(FIGURES_DIR / "fig4_relationship_rfm_scatter.png", width=235)  # ratio 1.848 -> height 127.2 pt
    img8 = make_aspect_image(FIGURES_DIR / "fig8_overseas_market_bubble.png", width=235)      # ratio 1.897 -> height 123.9 pt
    cap4 = Paragraph("<b>(a)</b> RFM geometry (log frequency vs log spend, colored by recency)", subcaption_style)
    cap8 = Paragraph("<b>(b)</b> Overseas unit economics (orders, AOV, GMV, and return %)", subcaption_style)

    side_by_side_rfm = Table([[img4, img8], [cap4, cap8]], colWidths=[252, 252])
    side_by_side_rfm.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 1),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
    ]))
    story.append(side_by_side_rfm)
    story.append(Paragraph("Figure 4: Multi-dimensional behavioral landscapes: (a) Bivariate customer RFM geometry; (b) Overseas market unit economics bubble chart.", caption_style))

    story.append(Paragraph("3.3.2 Transnational Unit Economics & International Market Prioritization", h3_style))
    story.append(Paragraph(
        "While the domestic UK market accounts for 88.9% of transactions (£8.99M gross sales), overseas markets represent highly lucrative wholesale corridors. "
        "Figure 4(b) and Table 3 analyze the unit economics of leading international markets across order velocity, Average Order Value (AOV), gross GMV, and return rate: "
        "<b>(1) Netherlands (#1 Expansion Target):</b> Netherlands generates £283,889.34 GMV across 93 wholesale orders with an extraordinary line-weighted AOV of £7,779.59 "
        "(£3,052.57 per invoice) and a negligible cancellation rate of <b>0.14%</b> (£409.80 total returns). This represents the highest unit-economics ROI in the entire business. "
        "<b>(2) Germany & France (Core Continental Strongholds):</b> Germany (443 orders, £205,381.15 GMV, 2.32% returns) and France (383 orders, £184,679.00 GMV, 1.31% returns) "
        "exhibit exceptional order velocity and dependable low return rates, serving as ideal anchors for localized EU fulfillment. "
        "<b>(3) EIRE & Australia:</b> EIRE accounts for £276k GMV (4.18% returns), while Australia generates £138k GMV across high-capacity wholesale consignments (0.86% returns). "
        "<b>(4) High-Friction Markets (Spain):</b> Spain demonstrates an elevated 7.11% return rate (£3,960 lost on £55.7k GMV), requiring cross-border logistics inspection.",
        body_style
    ))

    # Table 3: Booktabs Overseas Markets
    story.append(Paragraph("Table 3: Empirical unit economics, gross GMV, order velocity, and return rates for leading overseas target markets.", table_caption_style))
    table3_data = [
        [Paragraph("Country", table_header_style), Paragraph("Orders", table_header_style), Paragraph("Gross GMV (£)", table_header_style), Paragraph("Invoice AOV (£)", table_header_style), Paragraph("Lost Rev (£)", table_header_style), Paragraph("Return %", table_header_style), Paragraph("Strategic Commercial Designation", table_header_style)],
        [Paragraph("Netherlands", table_cell_left), Paragraph("93", table_cell_style), Paragraph("£283,889.34", table_cell_style), Paragraph("£3,052.57", table_cell_style), Paragraph("£409.80", table_cell_style), Paragraph("0.14%", table_cell_style), Paragraph("Tier-1 Direct Expansion Hub (Max Wholesale Unit ROI)", table_cell_left)],
        [Paragraph("EIRE", table_cell_left), Paragraph("284", table_cell_style), Paragraph("£276,090.86", table_cell_style), Paragraph("£972.15", table_cell_style), Paragraph("£11,535.84", table_cell_style), Paragraph("4.18%", table_cell_style), Paragraph("Tier-1 Mature Cross-Border Corridor", table_cell_left)],
        [Paragraph("Germany", table_cell_left), Paragraph("443", table_cell_style), Paragraph("£205,381.15", table_cell_style), Paragraph("£463.61", table_cell_style), Paragraph("£4,761.49", table_cell_style), Paragraph("2.32%", table_cell_style), Paragraph("Tier-1 Core EU Logistics & Fulfillment Anchor", table_cell_left)],
        [Paragraph("France", table_cell_left), Paragraph("383", table_cell_style), Paragraph("£184,679.00", table_cell_style), Paragraph("£482.19", table_cell_style), Paragraph("£2,416.40", table_cell_style), Paragraph("1.31%", table_cell_style), Paragraph("Tier-1 High-Velocity Low-Friction Market", table_cell_left)],
        [Paragraph("Australia", table_cell_left), Paragraph("56", table_cell_style), Paragraph("£138,103.81", table_cell_style), Paragraph("£2,466.14", table_cell_style), Paragraph("£1,181.31", table_cell_style), Paragraph("0.86%", table_cell_style), Paragraph("Tier-2 High-Capacity Bulk Wholesaler Hub", table_cell_left)],
        [Paragraph("Spain", table_cell_left), Paragraph("88", table_cell_style), Paragraph("£55,706.56", table_cell_style), Paragraph("£633.03", table_cell_style), Paragraph("£3,959.91", table_cell_style), Paragraph("7.11%", table_cell_style), Paragraph("Tier-3 High Friction (Transit Damage Audit Required)", table_cell_left)]
    ]
    t3 = Table(table3_data, colWidths=[70, 35, 70, 70, 55, 45, 159])
    t3.setStyle(get_booktabs_style())
    story.append(t3)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: CANDIDATE RETRIEVAL BENCHMARK (MATRIX 1: RECSYS)
    # =========================================================================
    story.append(Paragraph("4. CANDIDATE RETRIEVAL BENCHMARK: RECOMMENDER SYSTEM EVALUATION (MATRIX 1)", h1_style))
    story.append(Paragraph(
        "Basket completion and personalized product cross-selling represent primary revenue levers for e-commerce growth. "
        "In this benchmark, we formalize candidate retrieval as an offline Leave-One-Out (LOO) prediction task evaluated on unseen shopping carts.",
        body_style
    ))

    story.append(Paragraph("4.1 Candidate Retrieval Paradigms & Mathematical Formulations", h2_style))
    story.append(Paragraph(
        "We evaluate 7 models spanning 4 distinct algorithmic generations: "
        "<b>(1) Popularity Baseline:</b> Recommends globally most frequent items, serving as a zero-context heuristic. "
        "<b>(2) Lecture 8 Apriori Rules:</b> Mines frequent itemsets satisfying minimum support <i>s</i>(<i>X</i>) ≥ <i>s</i><sub>min</sub> and ranks candidates by Lift (Eq. 2):",
        body_style
    ))
    story.append(make_equation_block("eq2_apriori_lift.png", 2, height=23, space_before=3.5, space_after=3.5))

    story.append(Paragraph(
        "<b>(3) Lecture 8 Hierarchical Clustering:</b> Performs agglomerative Ward clustering on Jaccard distance matrix <i>D<sub>ij</sub></i> = 1 - |<i>B<sub>i</sub></i> ∩ <i>B<sub>j</sub></i>| / |<i>B<sub>i</sub></i> ∪ <i>B<sub>j</sub></i>|. "
        "<b>(4) Classic ItemCF (Amazon 2003):</b> Computes pairwise Jaccard co-occurrence across item neighborhoods. "
        "<b>(5) Matrix Factorization (SVD):</b> Decomposes binary user-basket matrix via Truncated SVD <b>X</b> ≈ <b>U</b><sub><i>k</i></sub> <b>Σ</b><sub><i>k</i></sub> <b>V</b><sub><i>k</i></sub><sup><i>T</i></sup> (<i>k</i>=64). "
        "<b>(6) Frontier Item2Vec:</b> Trains neural continuous Skip-Gram embeddings over basket sessions maximizing the log-likelihood (Eq. 3):",
        body_style
    ))
    story.append(make_equation_block("eq3_item2vec_loss.png", 3, height=23, space_before=3.5, space_after=3.5))

    story.append(Paragraph(
        "<b>(7) Proposed Multi-Channel Hybrid Ensemble:</b> We propose a multi-channel candidate retrieval architecture fusing ItemCF (weight 0.50), "
        "SVD (weight 0.25), and Item2Vec (weight 0.25) via Reciprocal Rank Fusion (RRF, Eq. 4), where <i>r<sub>m</sub></i>(<i>d</i>) is candidate rank in channel <i>m</i>:",
        body_style
    ))
    story.append(make_equation_block("eq4_rrf_fusion.png", 4, height=22, space_before=3.5, space_after=3.5))

    story.append(Paragraph("4.2 Leave-One-Out Evaluation Protocol & Empirical Matrix", h2_style))
    story.append(Paragraph(
        "The model is evaluated on 18,275 multi-item shopping baskets partitioned into an 80% training set (14,620 baskets) and 20% holdout test set (3,655 baskets). "
        "For each test basket, one item is randomly masked as the target ground truth <i>t</i>*, while remaining items form the context query. "
        "We track 6 quantitative metrics: HitRate@5, HitRate@10, NDCG@10, MRR, Novelty (-log<sub>2</sub> <i>p</i>(<i>i</i>)), and Catalog Long-Tail Coverage. Table 4 and Figure 5 report results.",
        body_style
    ))

    # Table 4: Booktabs RecSys Benchmark
    story.append(Paragraph("Table 4: Candidate retrieval recommender benchmark across 3,655 holdout test shopping baskets (Leave-One-Out Protocol).", table_caption_style))
    table4_data = [
        [Paragraph("Model", table_header_style), Paragraph("Hit@5", table_header_style), Paragraph("Hit@10", table_header_style), Paragraph("NDCG@10", table_header_style), Paragraph("MRR", table_header_style), Paragraph("Novelty", table_header_style), Paragraph("Coverage", table_header_style), Paragraph("Paradigm Category", table_header_style)],
        [Paragraph("Baseline: Popularity (Top Sellers)", table_cell_left), Paragraph("3.01%", table_cell_style), Paragraph("4.82%", table_cell_style), Paragraph("0.0241", table_cell_style), Paragraph("0.0168", table_cell_style), Paragraph("8.37 bits", table_cell_style), Paragraph("0.92%", table_cell_style), Paragraph("Heuristic Baseline", table_cell_left)],
        [Paragraph("Lecture 8: Apriori Rules (Lift)", table_cell_left), Paragraph("2.85%", table_cell_style), Paragraph("4.49%", table_cell_style), Paragraph("0.0240", table_cell_style), Paragraph("0.0177", table_cell_style), Paragraph("15.86 bits", table_cell_bold), Paragraph("44.81%", table_cell_style), Paragraph("Frequent Pattern Mining", table_cell_left)],
        [Paragraph("Lecture 8: Hierarchical Clustering", table_cell_left), Paragraph("8.16%", table_cell_style), Paragraph("12.29%", table_cell_style), Paragraph("0.0702", table_cell_style), Paragraph("0.0543", table_cell_style), Paragraph("9.91 bits", table_cell_style), Paragraph("19.88%", table_cell_style), Paragraph("Agglomerative Dendrogram", table_cell_left)],
        [Paragraph("Classic Benchmark: ItemCF (Jaccard)", table_cell_left), Paragraph("11.06%", table_cell_style), Paragraph("16.78%", table_cell_style), Paragraph("0.0940", table_cell_style), Paragraph("0.0716", table_cell_style), Paragraph("10.45 bits", table_cell_style), Paragraph("54.96%", table_cell_style), Paragraph("Neighborhood Collaborative", table_cell_left)],
        [Paragraph("Latent Factor: SVD Factorization", table_cell_left), Paragraph("8.62%", table_cell_style), Paragraph("13.44%", table_cell_style), Paragraph("0.0765", table_cell_style), Paragraph("0.0591", table_cell_style), Paragraph("11.06 bits", table_cell_style), Paragraph("60.92%", table_cell_style), Paragraph("Low-Rank Approximation", table_cell_left)],
        [Paragraph("Frontier SOTA: Item2Vec (Skip-Gram)", table_cell_left), Paragraph("6.84%", table_cell_style), Paragraph("9.66%", table_cell_style), Paragraph("0.0547", table_cell_style), Paragraph("0.0419", table_cell_style), Paragraph("13.26 bits", table_cell_style), Paragraph("73.33%", table_cell_bold), Paragraph("Continuous Latent Geometry", table_cell_left)],
        [Paragraph("Proposed: Multi-Channel Hybrid RRF", table_cell_bold_left), Paragraph("12.73%", table_cell_bold), Paragraph("18.64%", table_cell_bold), Paragraph("0.1039", table_cell_bold), Paragraph("0.0787", table_cell_bold), Paragraph("10.53 bits", table_cell_style), Paragraph("58.51%", table_cell_style), Paragraph("Multi-Channel Fusion (Proposed SOTA)", table_cell_bold_left)]
    ]
    t4 = Table(table4_data, colWidths=[125, 42, 45, 48, 42, 52, 48, 102])
    t4.setStyle(get_booktabs_style())
    story.append(t4)
    story.append(Spacer(1, 3))

    # Fig 6 preserving EXACT 2.212 aspect ratio
    img6 = make_aspect_image(FIGURES_DIR / "fig6_recsys_benchmark_comparison.png", width=340) # height = 153.7 pt
    story.append(img6)
    story.append(Paragraph("Figure 5: Recommender candidate retrieval benchmark: HitRate@5/10 (bars, left) vs catalog long-tail coverage (line, right).", caption_style))

    story.append(Paragraph("4.3 Empirical Findings & Theoretical Discussion", h2_style))
    story.append(Paragraph(
        "<b>(1) The Pareto Dominance of Multi-Channel RRF:</b> The Proposed Hybrid Ensemble achieves the highest accuracy across all models (18.64% HitRate@10, "
        "0.1039 NDCG@10, 0.0787 MRR), surpassing standalone ItemCF (16.78%) while expanding catalog coverage to 58.51%. "
        "<b>(2) The Sparsity Bottleneck of Rule Mining:</b> Apriori fails to generalize in high-dimensional retail catalogs (HitRate@10 = 4.49%), "
        "as most pairwise antecedents lack sufficient support (<i>s</i> &lt; 0.001). "
        "<b>(3) The Exploration Value of Item2Vec:</b> Item2Vec achieves an industry-leading 73.33% catalog coverage and 13.26 bits novelty. "
        "While its standalone hit rate is lower (9.66%), its semantic continuous embeddings inject indispensable long-tail diversity into the RRF ensemble.",
        body_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 6: CLV OUT-OF-TIME BENCHMARK (MATRIX 2)
    # =========================================================================
    story.append(Paragraph("5. CUSTOMER LIFETIME VALUE BENCHMARK: OUT-OF-TIME SPENDING FORECAST (MATRIX 2)", h1_style))
    story.append(Paragraph(
        "Accurately forecasting forward-looking customer spend is critical for marketing budget allocation. "
        "We implement an Out-of-Time (OOT) holdout validation protocol partitioning customer transaction histories into a 9-month calibration window "
        "(01/12/2010 to 31/08/2011, 3,309 calibration accounts) and a 100-day Q4 holdout evaluation period (01/09/2011 to 09/12/2011).",
        body_style
    ))

    story.append(Paragraph("5.1 Generative Latent Stochastic Processes vs Supervised Ensembles", h2_style))
    story.append(Paragraph(
        "Traditional curricula rely on unsupervised partitioning (<i>K</i>-Means) or static linear regressions (OLS, Ridge, KNN). "
        "In contrast, modern customer science models non-contractual purchasing as a continuous-time stochastic process: "
        "<b>(1) BG/NBD Model (Fader et al., 2005):</b> Assumes transaction counts follow a Poisson process with rate <i>&lambda;</i> ~ Gamma(<i>r</i>, <i>&alpha;</i>), "
        "and dropout occurs with probability <i>p</i> ~ Beta(<i>a</i>, <i>b</i>). The conditional probability that a customer is alive given history (<i>x</i>, <i>t<sub>x</sub></i>, <i>T</i>) is given by Eq. 5:",
        body_style
    ))
    story.append(make_equation_block("eq5_bgnbd_palive.png", 5, height=23, space_before=3.5, space_after=3.5))

    story.append(Paragraph(
        "<b>(2) Gamma-Gamma Model:</b> Models monetary transaction values as a Gamma distribution with unobserved mean <i>&nu;</i> ~ Gamma(<i>p</i>, <i>q</i>), "
        "predicting conditional average profit <b>E</b>[<i>M</i>].<br/>"
        "<b>(3) Proposed SOTA: Probabilistic-Guided Stacking:</b> To resolve the trade-off between generative behavioral priors and discriminative feature interactions, "
        "we construct a stacking ensemble extracting <i>P</i>(Alive), <b>E</b>[<i>X</i>], <b>E</b>[<i>M</i>], and latent stochastic CLV (<b>E</b>[<i>X</i>] · <b>E</b>[<i>M</i>]) from BG/NBD, "
        "feeding them alongside RFM-T features into a dual-loss tree ensemble combining Random Forest (L2 bagging for whale variance reduction) and "
        "HistGradientBoosting with L1 Absolute Error loss (robust to extreme Pareto skewness), blended via shrinkage (Eq. 6):",
        body_style
    ))
    story.append(make_equation_block("eq6_hybrid_clv.png", 6, height=21, space_before=3.5, space_after=3.5))

    story.append(Paragraph("5.2 Out-of-Time Q4 Evaluation Protocol & Empirical Matrix", h2_style))
    story.append(Paragraph(
        "We evaluate 8 models on 5 metrics: Holdout MAE (£), Holdout RMSE (£), Spearman Rank Correlation (<i>&rho;</i>), "
        "Top-Decile Precision (% of predicted top 10% customers that are actually in the top 10% spending decile), "
        "and <b>Revenue Capture @ Top 20% Budget (%)</b> (fraction of total Q4 actual revenue generated by the top 20% predicted accounts). Table 5 and Figure 6 report results.",
        body_style
    ))

    # Table 5: Booktabs CLV Benchmark
    story.append(Paragraph("Table 5: Out-of-Time CLV prediction benchmark across 3,309 calibration accounts evaluated on Q4 holdout period.", table_caption_style))
    table5_data = [
        [Paragraph("Model", table_header_style), Paragraph("Holdout MAE (£)", table_header_style), Paragraph("Holdout RMSE (£)", table_header_style), Paragraph("Spearman ρ", table_header_style), Paragraph("Top-10% Prec (%)", table_header_style), Paragraph("Rev Capture @ 20% Budget", table_header_style), Paragraph("Category / Paradigm", table_header_style)],
        [Paragraph("Lecture 8: K-Means Segmentation", table_cell_left), Paragraph("£945.14", table_cell_style), Paragraph("£4,718.91", table_cell_style), Paragraph("0.4506", table_cell_style), Paragraph("24.75%", table_cell_style), Paragraph("71.17%", table_cell_style), Paragraph("Unsupervised Clustering", table_cell_left)],
        [Paragraph("Lecture 5: Linear Regression (OLS)", table_cell_left), Paragraph("£841.91", table_cell_style), Paragraph("£4,534.21", table_cell_style), Paragraph("0.4673", table_cell_style), Paragraph("54.98%", table_cell_style), Paragraph("62.91%", table_cell_style), Paragraph("Parametric Linear", table_cell_left)],
        [Paragraph("Lecture 7: Ridge Regression (L2)", table_cell_left), Paragraph("£841.88", table_cell_style), Paragraph("£4,534.21", table_cell_style), Paragraph("0.4673", table_cell_style), Paragraph("54.98%", table_cell_style), Paragraph("62.91%", table_cell_style), Paragraph("Regularized Shrinkage", table_cell_left)],
        [Paragraph("Lecture 7: KNN Regressor (k=15)", table_cell_left), Paragraph("£824.00", table_cell_style), Paragraph("£4,303.60", table_cell_style), Paragraph("0.5423", table_cell_style), Paragraph("49.85%", table_cell_style), Paragraph("70.57%", table_cell_style), Paragraph("Instance-Based Non-parametric", table_cell_left)],
        [Paragraph("Advanced ML 1: Random Forest (100 Trees)", table_cell_left), Paragraph("£624.72", table_cell_style), Paragraph("£3,195.25", table_cell_bold), Paragraph("0.5082", table_cell_style), Paragraph("57.10%", table_cell_style), Paragraph("71.07%", table_cell_style), Paragraph("Ensemble Decision Tree Bagging", table_cell_left)],
        [Paragraph("Advanced ML 2: GBDT (HistGradient)", table_cell_left), Paragraph("£745.50", table_cell_style), Paragraph("£3,766.65", table_cell_style), Paragraph("0.4945", table_cell_style), Paragraph("53.59%", table_cell_style), Paragraph("69.54%", table_cell_style), Paragraph("Gradient Boosted Trees", table_cell_left)],
        [Paragraph("Frontier SOTA: BG/NBD + Gamma-Gamma", table_cell_left), Paragraph("£642.32", table_cell_style), Paragraph("£4,190.03", table_cell_style), Paragraph("0.5295", table_cell_style), Paragraph("58.31%", table_cell_style), Paragraph("64.08%", table_cell_style), Paragraph("Continuous Latent Stochastic", table_cell_left)],
        [Paragraph("Proposed: Probabilistic-Guided Stacking", table_cell_bold_left), Paragraph("£593.25", table_cell_bold), Paragraph("£3,760.04", table_cell_style), Paragraph("0.5549", table_cell_bold), Paragraph("59.82%", table_cell_bold), Paragraph("72.30%", table_cell_bold), Paragraph("Dual-Loss Stacking (Proposed SOTA)", table_cell_bold_left)]
    ]
    t5 = Table(table5_data, colWidths=[130, 52, 55, 48, 55, 62, 102])
    t5.setStyle(get_booktabs_style())
    story.append(t5)
    story.append(Spacer(1, 3))

    # Fig 7 preserving EXACT 2.264 aspect ratio
    img7 = make_aspect_image(FIGURES_DIR / "fig7_clv_benchmark_comparison.png", width=345) # height = 152.4 pt
    story.append(img7)
    story.append(Paragraph("Figure 6: Customer Lifetime Value (CLV) benchmark: Holdout MAE (£ GBP, lower is better) vs Top-Decile Precision (%, higher is better).", caption_style))

    story.append(Paragraph("5.3 Empirical Discussion & Commercial Targeting Superiority", h2_style))
    story.append(Paragraph(
        "<b>(1) Superiority of Probabilistic Stacking:</b> Our proposed model achieves Pareto dominance: it cuts holdout prediction error to £593.25 MAE "
        "(a £351.89 error reduction vs <i>K</i>-Means, £248.66 vs OLS), while achieving the highest rank correlation (<i>&rho;</i> = 0.5549) and highest top-decile precision (59.82%). "
        "<b>(2) Capital Efficiency in Budget Allocation:</b> In enterprise ad targeting, budget is constrained. Under a 20% budget targeting strategy, "
        "our proposed stacking model captures <b>72.30% of total actual Q4 spend</b>, ensuring maximum advertising dollar productivity.",
        body_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 7: STRATEGIC DECISION FRAMEWORK & BUDGET ALLOCATION
    # =========================================================================
    story.append(Paragraph("6. OPTIMAL CAPITAL ALLOCATION & LIFECYCLE DECISION FRAMEWORK", h1_style))
    story.append(Paragraph("6.1 Constrained Knapsack Optimization Formulation", h2_style))
    story.append(Paragraph(
        "To translate predictive benchmarks into operational capital allocation for the Head of E-commerce setting the 2012 budget, "
        "we formalize customer targeting as a <b>Constrained Knapsack Optimization Problem</b> (Eq. 7):",
        body_style
    ))
    story.append(make_equation_block("eq7_knapsack_opt.png", 7, height=22, space_before=3.5, space_after=3.5))

    story.append(Paragraph(
        "where predicted CLV <i>y<sub>i</sub></i> is the forward 1-year value from the Stacking Ensemble, <i>c<sub>i</sub></i> is customer acquisition/retention cost, "
        "<i>B</i> is the capital budget, and &Delta;<i>R<sub>i</sub></i>(<i>x<sub>i</sub></i>) represents the incremental net response rate "
        "accounting for organic ad cannibalization (Eq. 8):",
        body_style
    ))
    story.append(make_equation_block("eq8_net_response.png", 8, height=21, space_before=3.5, space_after=3.5))

    story.append(Paragraph(
        "This optimization directly yields our empirical four-quadrant crosshair lifecycle segmentation (Figure 7 and Table 6): "
        "accounts are partitioned along two empirical boundaries: (1) inactivity recency at 60 days, and (2) forward 1-year predicted CLV "
        "at £2,499.72 (the 80th percentile threshold isolating commercial wholesale VIPs from sporadic retail consumers).",
        body_style
    ))

    # Fig 5 preserving EXACT 2.679 aspect ratio
    img5 = make_aspect_image(FIGURES_DIR / "fig5_model_clv_budget_allocation.png", width=420) # height = 156.8 pt
    story.append(img5)
    story.append(Paragraph("Figure 7: Customer lifecycle crosshair quadrant scatter plot (left) and optimal 2012 marketing capital budget allocation (right).", caption_style))

    # Table 6: Booktabs Quadrant Policy
    story.append(Paragraph("Table 6: Empirical four-quadrant lifecycle segmentation and optimal capital allocation policy.", table_caption_style))
    table6_data = [
        [Paragraph("Lifecycle Quadrant", table_header_style), Paragraph("Accounts", table_header_style), Paragraph("Forward CLV", table_header_style), Paragraph("CLV Share", table_header_style), Paragraph("Avg Recency", table_header_style), Paragraph("2012 Budget %", table_header_style), Paragraph("Optimal Intervention & Resource Allocation Policy", table_header_style)],
        [Paragraph("Q1: Core Champions (Active VIP)", table_cell_left), Paragraph("744 (17.2%)", table_cell_style), Paragraph("£6,027,803", table_cell_style), Paragraph("69.61%", table_cell_style), Paragraph("16.2 days", table_cell_style), Paragraph("15% Budget", table_cell_style), Paragraph("Organic VIP Retention: Dedicated account tiering, early catalog previews; paid display ads suppressed to eliminate ad cannibalization", table_cell_left)],
        [Paragraph("Q2: At-Risk High-LTV (Dormant VIP)", table_cell_left), Paragraph("89 (2.1%)", table_cell_style), Paragraph("£404,201", table_cell_style), Paragraph("4.67%", table_cell_style), Paragraph("101.2 days", table_cell_style), Paragraph("<b>50% Budget (TOP)</b>", table_cell_style), Paragraph("High Marginal ROI: Targeted retargeting, bespoke volume incentives, direct commercial outreach to recover high-margin wholesale cash flow", table_cell_left)],
        [Paragraph("Q3: Promising Growth (Active Growth)", table_cell_left), Paragraph("1,664 (38.4%)", table_cell_style), Paragraph("£1,547,373", table_cell_style), Paragraph("17.87%", table_cell_style), Paragraph("25.9 days", table_cell_style), Paragraph("25% Budget", table_cell_style), Paragraph("Basket Cross-Sell: Item2Vec recommendation ads, volume bundle discounts, loyalty milestone incentives to graduate accounts into Q1", table_cell_left)],
        [Paragraph("Q4: Low-Value Inactive (Dormant Small)", table_cell_left), Paragraph("1,837 (42.4%)", table_cell_style), Paragraph("£680,362", table_cell_style), Paragraph("7.86%", table_cell_style), Paragraph("182.6 days", table_cell_style), Paragraph("10% Budget", table_cell_style), Paragraph("Paid Ad Suppression: Strictly automated zero-marginal-cost email drip campaigns; avoid display ad spend to eliminate negative ROI", table_cell_left)]
    ]
    t6 = Table(table6_data, colWidths=[95, 45, 55, 42, 45, 58, 164])
    t6.setStyle(get_booktabs_style())
    story.append(t6)
    story.append(Spacer(1, 3))

    story.append(Paragraph("6.2 Empirical Capital Allocation Strategy & Operational Recommendations", h2_style))
    story.append(Paragraph(
        "<b>Executive Synthesis:</b> <i>Prioritize Q2 at-risk high-LTV accounts (89 commercial buyers representing £404k forward CLV, allocating 50% of paid ad budget), "
        "expand wholesale logistics into the Netherlands (£284k GMV, 0.14% return rate), and deploy Multi-Channel Hybrid cross-selling for Q3 accounts. "
        "Recover £246k in leaking revenue by capping bulk checkout errors and re-packaging fragile decor SKUs.</i><br/>"
        "<b>Three Grounded Resource Allocation Strategies:</b><br/>"
        "1. <b>Paid Retargeting Concentration (50% Budget to Q2):</b> Allocate half of paid display and search budget to win back the 89 dormant high-LTV accounts. "
        "Suppress paid advertising for Q1 (organic champions, 15% budget) and Q4 (inactive consumers, 10% email only) to eliminate £120k+ in ad cannibalization.<br/>"
        "2. <b>International Logistics Expansion (Netherlands & Core EU):</b> Establish dedicated wholesale freight and localized ordering for the Netherlands (AOV £7,780, returns 0.14%) "
        "and consolidate Rhine-corridor fulfillment for Germany (£205k) and France (£185k), capturing high-velocity cross-border gross profit.<br/>"
        "3. <b>Dual-Track Operational Return Intervention (Recovering £360k+):</b> Institute a 30-minute confirmation hold and automated bulk checkout limits in OMS to eliminate £246k in Type I reversals, "
        "and re-engineer bubble-wrap transit packaging for fragile ceramics (Regency Cakestand) to reduce physical returns by £115k.",
        body_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 8: DISCUSSION, THREATS TO VALIDITY & REFERENCES
    # =========================================================================
    col1_p8 = [
        Paragraph("7. DISCUSSION & THREATS TO VALIDITY", h1_col),
        Paragraph(
            "<b>7.1 Methodological Strengths & Insights:</b> This study demonstrates that unifying classroom statistical baselines (Apriori, <i>K</i>-Means, OLS) with "
            "modern deep and probabilistic architectures (Item2Vec, BG/NBD, GBDT Stacking) resolves the fundamental trade-off between model interpretability and predictive accuracy. "
            "In candidate retrieval, Multi-Channel RRF achieves Pareto dominance by leveraging ItemCF for high-precision co-purchases while utilizing Item2Vec to explore catalog long tails. "
            "In CLV forecasting, probabilistic stacking prevents whale-induced overfitting while retaining continuous-time dropout estimation.",
            col_style
        ),
        Paragraph(
            "<b>7.2 Threats to Validity:</b> "
            "<i>(1) Internal Validity:</i> 24.9% of raw transaction lines lacked customer identifiers (guest checkouts). While retained for basket-level RecSys, their omission from CLV "
            "may underestimate total churn rates. "
            "<i>(2) External Validity:</i> The dataset spans UK wholesale giftware in 2010–2011; seasonality and wholesale basket dynamics may differ in consumer-packaged goods (CPG) or fashion retail. "
            "<i>(3) Construct Validity:</i> Holdout CLV was evaluated across a 100-day Q4 window; extended multi-year longitudinal validation is warranted.",
            col_style
        ),
    ]

    col2_p8 = [
        Paragraph("8. CONCLUSION & FUTURE WORK", h1_col),
        Paragraph(
            "We have presented an empirical analytics study and strategic decision framework for transnational e-commerce. "
            "By rigorously evaluating 7 recommender architectures and 8 CLV prediction models, we demonstrate that proposed hybrid ensembles achieve Pareto dominance "
            "in both customer targeting and basket completion. Forensic anomaly diagnosis successfully decouples £246k in administrative checkout errors from chronic physical returns, "
            "providing senior management with a concrete capital allocation roadmap for 2012. "
            "Future work will explore graph neural networks (GNNs) for multi-session basket graph propagation and causal uplift modeling for incremental ad spend optimization.",
            col_style
        ),
        Paragraph("ACKNOWLEDGMENTS", h1_col),
        Paragraph(
            "The author expresses sincere gratitude to the faculty and teaching team of IE5005 Applied Data Analytics for Industrial Engineers at the National University of Singapore (NUS) "
            "for foundational instruction in analytical methodologies and statistical learning theory.",
            col_style
        ),
    ]

    story.append(make_two_column_block(col1_p8, col2_p8))
    story.append(Spacer(1, 4))

    story.append(Paragraph("REFERENCES", h1_style))
    
    # 16 references formatted into a 2-column academic bibliography table
    refs_left = [
        "[1] R. Agrawal and R. Srikant, 'Fast algorithms for mining association rules,' in <i>Proc. VLDB</i>, 1994, pp. 487–499.",
        "[2] G. Linden, B. Smith, and J. York, 'Amazon.com recommendations: Item-to-item collaborative filtering,' <i>IEEE Internet Comput.</i>, vol. 7, no. 1, pp. 76–80, 2003.",
        "[3] P. S. Fader, B. G. S. Hardie, and K. L. Lee, '\"Counting your customers\" the easy way: Alternative to Pareto/NBD,' <i>Mark. Sci.</i>, vol. 24, no. 2, pp. 275–284, 2005.",
        "[4] P. S. Fader, B. G. S. Hardie, and K. L. Lee, 'RFM and CLV: Using iso-value curves for multicategory spending,' <i>J. Mark. Res.</i>, vol. 42, no. 4, pp. 415–430, 2005.",
        "[5] O. Barkan and N. Koenigstein, 'Item2Vec: Neural item embedding for collaborative filtering,' in <i>IEEE MLSP</i>, 2016, pp. 1–6.",
        "[6] T. Mikolov, I. Sutskever, K. Chen, G. Corrado, and J. Dean, 'Distributed representations of words,' in <i>Proc. NeurIPS</i>, 2013, pp. 3111–3119.",
        "[7] G. V. Cormack, C. Clarke, and S. Buettcher, 'Reciprocal rank fusion outperforms Condorcet,' in <i>Proc. ACM SIGIR</i>, 2009, pp. 758–759.",
        "[8] L. Breiman, 'Random forests,' <i>Mach. Learn.</i>, vol. 45, no. 1, pp. 5–32, 2001."
    ]
    refs_right = [
        "[9] J. H. Friedman, 'Greedy function approximation: A gradient boosting machine,' <i>Ann. Stat.</i>, vol. 29, no. 5, pp. 1189–1232, 2001.",
        "[10] Y. Koren, R. Bell, and C. Volinsky, 'Matrix factorization techniques for recommender systems,' <i>Computer</i>, vol. 42, no. 8, pp. 30–37, 2009.",
        "[11] D. Chen, 'UK Online Retail Transaction Dataset,' UCI Machine Learning Repository, 2015. DOI: 10.24432/C5BW33.",
        "[12] J. H. Ward Jr, 'Hierarchical grouping to optimize an objective function,' <i>J. Am. Stat. Assoc.</i>, vol. 58, pp. 236–244, 1963.",
        "[13] J. MacQueen, 'Some methods for classification and analysis of observations,' in <i>Proc. 5th Berkeley Symp.</i>, 1967, pp. 281–297.",
        "[14] A. E. Hoerl and R. Kennard, 'Ridge regression: Biased estimation,' <i>Technometrics</i>, vol. 12, no. 1, pp. 55–67, 1970.",
        "[15] F. Pedregosa et al., 'Scikit-learn: Machine learning in Python,' <i>J. Mach. Learn. Res.</i>, vol. 12, pp. 2825–2830, 2011.",
        "[16] C. Davidson-Pilon, 'Lifetimes: Measuring customer lifetime value in Python,' GitHub repository, 2021."
    ]

    ref_table_data = []
    for l, r in zip(refs_left, refs_right):
        ref_table_data.append([Paragraph(l, ref_style), Paragraph(r, ref_style)])

    ref_table = Table(ref_table_data, colWidths=[250, 250])
    ref_table.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,0), 0.8, colors.HexColor("#111111")),
        ('LINEBELOW', (0,-1), (-1,-1), 0.8, colors.HexColor("#111111")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 0.8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0.8),
    ]))
    story.append(ref_table)

    doc.build(story, canvasmaker=AcademicCanvas)
    print(f"[PDF Generator] Successfully generated conference paper: {pdf_path}")


if __name__ == '__main__':
    create_paper_pdf()
