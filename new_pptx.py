"""
================================================================================
PROJECT SMARTRETAIN — 15-SLIDE EXECUTIVE POWERPOINT GENERATOR
================================================================================
File     : new_pptx.py
Purpose  : Auto-generate a polished 15-slide executive presentation that
           combines EDA findings, ML model results, market analysis, and a
           quantified business proposal with full citations.
Input    : figures_eda/  (from new_eda.py)
           figures_model/ (from new_test.py)
Output   : SmartRetain_Executive_Proposal.pptx
================================================================================
"""

import os
import sys

# ── Auto-install python-pptx if needed ───────────────────────────────────────
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    from pptx.oxml.ns import qn
    import pptx.oxml as pptx_oxml
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-pptx"])
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Slide dimensions (16:9 widescreen) ───────────────────────────────────────
W = Inches(13.333)
H = Inches(7.5)

# ── Brand Colours ─────────────────────────────────────────────────────────────
NAVY      = RGBColor(15,  23,  42)   # Slate-900   — primary dark
DARKBLUE  = RGBColor(30,  58, 138)   # Blue-900    — header blocks
BLUE      = RGBColor(37,  99, 235)   # Blue-600    — accent
TEAL      = RGBColor(13, 148, 136)   # Teal-600    — positive
RED       = RGBColor(220, 38,  38)   # Red-600     — warning / churn
AMBER     = RGBColor(217,119,   6)   # Amber-600   — medium risk
GREEN     = RGBColor(16, 185, 129)   # Emerald-500 — positive / saved
SLATE     = RGBColor(100,116, 139)   # Slate-500   — body text
WHITE     = RGBColor(255,255, 255)
LIGHT_BG  = RGBColor(248,250, 252)   # Slate-50    — card backgrounds
CARD_BG   = RGBColor(241,245, 249)   # Slate-100
BORDER    = RGBColor(203,213, 225)   # Slate-300

# ── Figure paths ─────────────────────────────────────────────────────────────
FIG_EDA   = "figures_eda"
FIG_MODEL = "figures_model"

def p(inches): return Inches(inches)


# =============================================================================
# HELPER UTILITIES
# =============================================================================
def add_rect(slide, left, top, width, height, fill_color, border_color=None, border_pt=0):
    from pptx.enum.shapes import MSO_SHAPE
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color and border_pt:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(border_pt)
    else:
        shape.line.fill.background()
    return shape


def add_rounded(slide, left, top, width, height, fill_color, border_color=None, border_pt=1):
    from pptx.enum.shapes import MSO_SHAPE
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(border_pt)
    else:
        shape.line.fill.background()
    return shape


def add_text(slide, text, left, top, width, height,
             font_size=11, bold=False, color=None, align=PP_ALIGN.LEFT,
             word_wrap=True, italic=False):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = word_wrap
    p_ = tf.paragraphs[0]
    p_.text = text
    p_.alignment = align
    p_.font.size = Pt(font_size)
    p_.font.bold = bold
    p_.font.italic = italic
    if color:
        p_.font.color.rgb = color
    return tb


def add_para(tf, text, font_size=10, bold=False, color=None,
             align=PP_ALIGN.LEFT, space_before=0, space_after=4, italic=False):
    p_ = tf.add_paragraph()
    p_.text = text
    p_.alignment = align
    p_.font.size = Pt(font_size)
    p_.font.bold = bold
    p_.font.italic = italic
    p_.space_before = Pt(space_before)
    p_.space_after  = Pt(space_after)
    if color:
        p_.font.color.rgb = color
    return p_


def header_bar(slide, title, subtitle="", bar_color=DARKBLUE, title_color=WHITE):
    """Dark top header bar with title."""
    add_rect(slide, p(0), p(0), W, p(1.15), bar_color)
    add_text(slide, title,
             p(0.5), p(0.12), p(12.3), p(0.65),
             font_size=22, bold=True, color=title_color)
    if subtitle:
        add_text(slide, subtitle,
                 p(0.5), p(0.72), p(12.3), p(0.38),
                 font_size=11, bold=False, color=RGBColor(148,163,184))


def slide_number(slide, n, total=15):
    add_text(slide, f"Slide {n} / {total}",
             p(11.8), p(7.15), p(1.4), p(0.28),
             font_size=8, color=SLATE, align=PP_ALIGN.RIGHT)


def footer(slide, citation="", n=1, total=15):
    add_rect(slide, p(0), p(7.18), W, p(0.32), DARKBLUE)
    cite_text = f"Source: {citation}   |   " if citation else ""
    add_text(slide,
             f"{cite_text}Project SmartRetain  —  Confidential PoC Proposal",
             p(0.4), p(7.19), p(11.5), p(0.28),
             font_size=8, color=RGBColor(148,163,184))
    add_text(slide, f"{n} / {total}",
             p(12.8), p(7.19), p(0.45), p(0.28),
             font_size=8, color=RGBColor(148,163,184), align=PP_ALIGN.RIGHT)


def safe_image(slide, path, left, top, width, height):
    """Add image only if the file exists."""
    if os.path.exists(path):
        slide.shapes.add_picture(path, left, top, width, height)
    else:
        # Grey placeholder
        add_rounded(slide, left, top, width, height, CARD_BG, BORDER, 1)
        add_text(slide, f"[Figure not found]\n{path}",
                 left + p(0.1), top + height/2 - p(0.3), width - p(0.2), p(0.6),
                 font_size=9, color=SLATE, align=PP_ALIGN.CENTER)


def kpi_card(slide, left, top, width, height,
             label, value, sub="", label_color=BLUE, value_color=NAVY):
    add_rounded(slide, left, top, width, height, CARD_BG, BORDER, 0.75)
    tb = slide.shapes.add_textbox(left + p(0.18), top + p(0.14),
                                  width - p(0.36), height - p(0.2))
    tf = tb.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = label.upper()
    p1.font.size = Pt(8.5); p1.font.bold = True
    p1.font.color.rgb = label_color

    p2 = tf.add_paragraph()
    p2.text = value
    p2.font.size = Pt(18); p2.font.bold = True
    p2.font.color.rgb = value_color

    if sub:
        p3 = tf.add_paragraph()
        p3.text = sub
        p3.font.size = Pt(9); p3.font.color.rgb = SLATE


def bullet_box(slide, left, top, width, height, title, bullets,
               title_color=NAVY, bullet_color=SLATE, title_size=12, bullet_size=10,
               bg=CARD_BG):
    add_rounded(slide, left, top, width, height, bg, BORDER, 0.75)
    tb = slide.shapes.add_textbox(left + p(0.18), top + p(0.14),
                                  width - p(0.36), height - p(0.22))
    tf = tb.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = title
    p1.font.size = Pt(title_size); p1.font.bold = True
    p1.font.color.rgb = title_color
    for b in bullets:
        px = tf.add_paragraph()
        px.text = f"  \u2022  {b}"
        px.font.size = Pt(bullet_size)
        px.font.color.rgb = bullet_color
        px.space_after = Pt(3)


# =============================================================================
# BUILD PRESENTATION
# =============================================================================
def build():
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    blank = prs.slide_layouts[6]   # completely blank layout

    # =========================================================================
    # SLIDE 1 — TITLE SLIDE
    # =========================================================================
    s = prs.slides.add_slide(blank)
    # Full dark background
    add_rect(s, p(0), p(0), W, H, DARKBLUE)
    # Teal accent bar at bottom
    add_rect(s, p(0), p(6.9), W, p(0.6), RGBColor(13,148,136))

    add_text(s, "PROJECT SMARTRETAIN",
             p(1.0), p(1.3), p(11.3), p(0.55),
             font_size=13, bold=True, color=RGBColor(96,165,250))
    add_text(s,
             "Machine Learning-Driven Customer Retention\n& Revenue Preservation Strategy",
             p(1.0), p(1.85), p(11.3), p(1.6),
             font_size=34, bold=True, color=WHITE)
    add_text(s,
             "A Proof of Concept (PoC) Business Transformation Proposal for Executive Leadership",
             p(1.0), p(3.45), p(11.3), p(0.55),
             font_size=14, color=RGBColor(203,213,225))

    # Meta card
    add_rounded(s, p(1.0), p(4.2), p(11.3), p(2.1),
                RGBColor(15,23,42), RGBColor(51,65,85), 1)
    tb = s.shapes.add_textbox(p(1.35), p(4.4), p(10.7), p(1.7))
    tf = tb.text_frame; tf.word_wrap = True
    lines = [
        ("Prepared by: IT Consulting Associate — AI & Data Science Practice", 13, True,  WHITE),
        ("Client: Telecommunications Provider  |  Engagement Type: PoC Machine Learning Project", 11, False, RGBColor(203,213,225)),
        ("Dataset: Client.csv + Record.csv  |  N = 100,000 customers  |  100 merged features", 11, False, RGBColor(148,163,184)),
        ("Stack: Python 3 · Scikit-Learn · HistGradientBoosting · RandomForest · Logistic Regression", 11, False, RGBColor(148,163,184)),
    ]
    for i,(txt,sz,bd,col) in enumerate(lines):
        p_ = tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p_.text = txt; p_.font.size = Pt(sz)
        p_.font.bold = bd; p_.font.color.rgb = col
        p_.space_after = Pt(5)

    add_text(s, "Confidential — For Executive Review Only",
             p(1.0), p(6.92), p(11.3), p(0.4),
             font_size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # =========================================================================
    # SLIDE 2 — TABLE OF CONTENTS (SECTION DIVIDER)
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_rect(s, p(0), p(0), W, H, RGBColor(248,250,252))
    header_bar(s, "Presentation Roadmap", "How This Proposal Is Structured")
    footer(s, n=2)

    sections = [
        ("01", "Executive Summary",          "The burning platform, key findings & business case at a glance",     BLUE),
        ("02", "Market & Industry Context",  "Telecom churn benchmarks, competitive landscape, external data",     TEAL),
        ("03", "Data & EDA Overview",        "Dataset profile, data quality, financial baseline",                  DARKBLUE),
        ("04", "Root-Cause Analysis",        "Equipment, billing, behavioral & tenure drivers of churn",           RED),
        ("05", "Machine Learning Models",    "Model selection, training, benchmark scorecard & decile lift",        BLUE),
        ("06", "Business Proposal & ROI",    "Targeted strategy, quantified revenue impact, implementation plan",  GREEN),
        ("07", "References",                 "All external data, papers, and benchmarks cited in this report",     SLATE),
    ]
    cw = p(1.55); ch = p(0.73); gap = p(0.06)
    row1_top = p(1.35); row2_top = p(2.15)
    cols = [p(0.5), p(2.18), p(3.86), p(5.54), p(7.22), p(8.90), p(10.58)]

    for i,(num,title,desc,col) in enumerate(sections):
        lft = cols[i]
        add_rounded(s, lft, row1_top, cw, p(1.65), CARD_BG, BORDER, 0.75)
        add_rect(s, lft, row1_top, cw, p(0.28), col)
        add_text(s, num, lft+p(0.08), row1_top+p(0.03), cw-p(0.12), p(0.28),
                 font_size=14, bold=True, color=WHITE)
        add_text(s, title, lft+p(0.1), row1_top+p(0.32), cw-p(0.18), p(0.42),
                 font_size=10, bold=True, color=NAVY)
        add_text(s, desc,  lft+p(0.1), row1_top+p(0.75), cw-p(0.18), p(0.82),
                 font_size=8.5, color=SLATE)

    add_text(s, "Each section builds logically from data evidence through to a quantified business recommendation.",
             p(0.5), p(3.22), p(12.3), p(0.38),
             font_size=10, italic=True, color=SLATE, align=PP_ALIGN.CENTER)

    # =========================================================================
    # SLIDE 3 — EXECUTIVE SUMMARY
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_rect(s, p(0), p(0), W, H, LIGHT_BG)
    header_bar(s, "Executive Summary",
               "Key Findings, ML Results & Quantified Business Case — One Page")
    footer(s, "Internal PoC Analysis (Client.csv + Record.csv, N=100,000)", n=3)

    cards = [
        ("The Burning Platform",
         RED, [
            ("Churn Rate",            "49.56%"),
            ("Annual Revenue at Risk","$34.45 Million"),
            ("Core Challenge",        "100k customer records; no in-house ML to flag at-risk customers before cancellation."),
            ("Strategic Risk",        "Acquiring new subscribers costs 5-7x more than retaining existing ones ($315 CAC vs $45 retention offer)."),
         ]),
        ("ML PoC Results",
         BLUE, [
            ("Primary Model",         "HistGradientBoosting"),
            ("Evaluation Metric",     "ROC-AUC"),
            ("Score",                 "0.6913"),
            ("Decile-1 Churn",        "79.85% (1.61x lift over random)"),
            ("Key Finding",           "Top 20% of predicted-risk customers contain 30% of all churners — 1.48x precision gain."),
         ]),
        ("Root-Cause Drivers",
         DARKBLUE, [
            ("Device Obsolescence",   "Equipment >2 yrs: 57.9% churn vs 41.8% baseline (+16.1pp)"),
            ("Silent Churners",       "0 support calls = 51.8% churn — customers leave without warning"),
            ("Bill Shock",            "Overage >$30 drives churn above 52.6%"),
            ("Contract Cliff",        "Peak churn: months 13–18 (renewal window)"),
         ]),
        ("Business Case ROI",
         GREEN, [
            ("Target Cohort",         "Top 20% — 20,000 accounts"),
            ("Saved Customers",       "3,285 accounts / year (22.5% save rate)"),
            ("Gross Rev Preserved",   "$2,314,742 / year"),
            ("Campaign Cost",         "$900,000 ($45 per offer)"),
            ("Net Profit Lift",       "+$1,414,742 / year"),
            ("ROI",                   "157% — Payback: 4.7 months"),
         ]),
    ]
    cw2 = p(2.9); ch2 = p(5.2); y0 = p(1.3)
    for i,(title,col,items) in enumerate(cards):
        lft = p(0.45) + i*(cw2 + p(0.28))
        add_rounded(s, lft, y0, cw2, ch2, CARD_BG, col, 1.5)
        add_rect(s, lft, y0, cw2, p(0.3), col)
        add_text(s, title, lft+p(0.1), y0+p(0.04), cw2-p(0.18), p(0.28),
                 font_size=10.5, bold=True, color=WHITE)
        tb = s.shapes.add_textbox(lft+p(0.14), y0+p(0.38), cw2-p(0.26), ch2-p(0.5))
        tf = tb.text_frame; tf.word_wrap = True
        for j,(k,v) in enumerate(items):
            pk = tf.paragraphs[0] if j==0 else tf.add_paragraph()
            pk.text = k.upper()
            pk.font.size = Pt(8.5); pk.font.bold = True
            pk.font.color.rgb = col
            pv = tf.add_paragraph()
            pv.text = v
            pv.font.size = Pt(10.5); pv.font.color.rgb = NAVY
            pv.space_after = Pt(6)

    # =========================================================================
    # SLIDE 4 — MARKET & INDUSTRY CONTEXT
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_rect(s, p(0), p(0), W, H, LIGHT_BG)
    header_bar(s, "Market & Industry Context",
               "The Economics of Retention in a Saturated Telecom Market")
    footer(s, "McKinsey (2021); Bain & Co. (2020); GSMA Intelligence (2023); Ericsson Mobility Report (2024)", n=4)

    # Left: industry bullets
    bullet_box(s, p(0.45), p(1.3), p(5.8), p(5.5),
               "Telecommunications Macro Context",
               [
                "Global mobile subscriber growth has flattened at ~1-2% CAGR in mature markets (GSMA, 2023).",
                "U.S. wireless market penetration exceeds 95%; growth now driven by ARPU uplift, not new subs.",
                "Average annual churn in U.S. telecom: 20-35% postpaid; up to 50%+ for prepaid (FCC, 2022).",
                "Customer Acquisition Cost (CAC): $300-$400 per new subscriber (McKinsey, 2021).",
                "Retention cost: $30-$60 per saved customer via targeted intervention (Bain & Co., 2020).",
                "A 5% reduction in churn increases profit by 25-95% in subscription businesses (Bain, 2020).",
                "Proactive ML-guided retention programs save 20-30% of targeted at-risk customers (McKinsey, 2021).",
               ],
               title_color=DARKBLUE, title_size=12, bullet_size=9.5)

    # Right: 3 stat cards
    stats = [
        ("Industry Avg Churn (U.S. Postpaid)", "20 - 35%",  "vs. Client's 49.56%", RED),
        ("CAC vs Retention Cost Ratio",         "5 - 7x",    "New sub costs 5-7x more to acquire than retain", AMBER),
        ("Profit Lift from 5% Churn Reduction", "25 - 95%",  "Compounding effect on LTV (Bain & Co., 2020)", GREEN),
        ("ML Retention Save Rate (Industry)",   "20 - 30%",  "Of targeted high-risk customers saved via proactive offer", BLUE),
    ]
    sw = p(6.3); sy0 = p(1.3); sh = p(1.22); sgap = p(0.16)
    for i,(lbl,val,sub,col) in enumerate(stats):
        kpi_card(s, p(6.6), sy0 + i*(sh+sgap), sw, sh,
                 lbl, val, sub, label_color=col, value_color=col)

    # =========================================================================
    # SLIDE 5 — DATA PROFILE & EDA OVERVIEW
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_rect(s, p(0), p(0), W, H, LIGHT_BG)
    header_bar(s, "Dataset Profile & EDA Overview",
               "100,000 Customer Records | 100 Merged Features | 49.56% Churn Rate")
    footer(s, "Internal PoC Dataset: Client.csv + Record.csv (merged on Customer_ID)", n=5)

    # Data quality cards (top row)
    dq_items = [
        ("Total Customers",  "100,000",    "Merged from Client + Record",  BLUE),
        ("Total Features",   "100",        "50 client + 50 usage/billing", DARKBLUE),
        ("Overall Churn",    "49.56%",     "49,562 churned customers",     RED),
        ("Missing Data",     "3.43%",      "43 of 100 columns have NaN",   AMBER),
        ("Duplicate IDs",    "0",          "No duplicate Customer_IDs",    GREEN),
        ("Numeric / Cat",    "82 / 21",    "After feature engineering",    TEAL),
    ]
    kw = p(1.9); ky = p(1.28); kh = p(1.0); kgap = p(0.17)
    for i,(lbl,val,sub,col) in enumerate(dq_items):
        kpi_card(s, p(0.45)+i*(kw+kgap), ky, kw, kh,
                 lbl, val, sub, label_color=col, value_color=col)

    # EDA Overview figure
    safe_image(s, os.path.join(FIG_EDA, "fig1_eda_overview.png"),
               p(0.45), p(2.42), p(12.43), p(4.5))

    # =========================================================================
    # SLIDE 6 — EDA: HARDWARE & EQUIPMENT DRIVERS
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_rect(s, p(0), p(0), W, H, LIGHT_BG)
    header_bar(s, "Root-Cause Analysis: Hardware & Device Obsolescence",
               "Older & Budget Devices Are the #1 Structural Driver of Churn")
    footer(s, "Internal PoC Dataset: Client.csv (eqpdays, hnd_price, hnd_webcap, refurb_new)", n=6)

    safe_image(s, os.path.join(FIG_EDA, "fig2_equipment_drivers.png"),
               p(0.45), p(1.28), p(8.5), p(5.85))

    # Insight callout box on right
    bullet_box(s, p(9.15), p(1.28), p(4.0), p(5.85),
               "Key Device Insights",
               [
                "Devices >2 years old: 57.9% churn — +16.1pp above average.",
                "Budget handsets ($1-$50): 57.5% churn vs 41.0% for premium (>$150).",
                "Legacy non-web handsets (WC code): churn rate significantly above web-enabled (WCMB).",
                "Refurbished handsets churn ~4.5pp higher than new devices.",
                "Implication: Hardware upgrade offers targeting aged-device users are the highest-ROI single intervention.",
               ],
               title_color=DARKBLUE, title_size=11, bullet_size=9.5)

    # =========================================================================
    # SLIDE 7 — EDA: BEHAVIORAL & BILLING DRIVERS
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_rect(s, p(0), p(0), W, H, LIGHT_BG)
    header_bar(s, "Root-Cause Analysis: Behavioral & Billing Signals",
               "Silent Disengagement + Bill Shock = Invisible Churn")
    footer(s, "Internal PoC Dataset: Record.csv (custcare_Mean, change_mou, ovrrev_Mean)", n=7)

    safe_image(s, os.path.join(FIG_EDA, "fig3_behavioral_billing.png"),
               p(0.45), p(1.28), p(8.5), p(4.2))

    # Insight cards at bottom
    insights = [
        ("Silent Churner Paradox",
         "55,692 customers (55.7%) made ZERO support calls yet churn at 51.8%. Customers leave silently — waiting for a complaint is too late.",
         RED),
        ("Usage Contraction Signal",
         "Customers whose MOU drops >100 min/month churn at 53.6%. Declining engagement is a reliable 30-60 day leading indicator.",
         AMBER),
        ("Bill Shock Effect",
         "Overage charges >$30/month push churn above 52.6%. Proactive plan right-sizing or overage caps can directly reduce this friction.",
         BLUE),
    ]
    iw = p(3.9); iy = p(5.62); ih = p(1.6)
    for i,(title,txt,col) in enumerate(insights):
        lft = p(0.45) + i*(iw+p(0.22))
        add_rounded(s, lft, iy, iw, ih, CARD_BG, col, 1.2)
        add_text(s, title, lft+p(0.14), iy+p(0.1), iw-p(0.26), p(0.36),
                 font_size=10, bold=True, color=col)
        add_text(s, txt,   lft+p(0.14), iy+p(0.44), iw-p(0.26), ih-p(0.5),
                 font_size=9, color=NAVY)

    # =========================================================================
    # SLIDE 8 — EDA: TENURE & CORRELATION
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_rect(s, p(0), p(0), W, H, LIGHT_BG)
    header_bar(s, "Root-Cause Analysis: Tenure Lifecycle & Feature Correlation",
               "Contract Renewal Windows Create Predictable Churn Cliffs")
    footer(s, "Internal PoC Dataset: Record.csv (months); Correlation: all numeric features", n=8)

    safe_image(s, os.path.join(FIG_EDA, "fig4_tenure.png"),
               p(0.45), p(1.28), p(6.5), p(5.85))
    safe_image(s, os.path.join(FIG_EDA, "fig6_correlation_heatmap.png"),
               p(7.1), p(1.28), p(5.78), p(5.85))

    # =========================================================================
    # SLIDE 9 — ML: MODEL SELECTION & RATIONALE
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_rect(s, p(0), p(0), W, H, LIGHT_BG)
    header_bar(s, "Machine Learning: Model Selection & Rationale",
               "Why Three Models? Benchmark Diversity Ensures Robust Recommendation")
    footer(s, "Scikit-Learn 1.x Documentation; Friedman (2001) Gradient Boosting; Breiman (2001) Random Forests", n=9)

    models_info = [
        ("Model 1 (Primary)\nHistGradientBoosting",
         BLUE,
         [
          "Type: Gradient Boosted Decision Trees (LightGBM-style)",
          "Why chosen: Handles missing values natively, strong performance on tabular data, regularisation via learning rate & leaf constraints.",
          "Hyperparams: max_iter=150, lr=0.08, max_leaf_nodes=31, min_samples_leaf=30",
          "ROC-AUC = 0.6913  |  PR-AUC = 0.6836",
          "Accuracy = 63.50%  |  F1 = 0.6347",
          "Train time: 12.93s",
         ]),
        ("Model 2 (Benchmark)\nRandomForestClassifier",
         TEAL,
         [
          "Type: Ensemble of 100 Decision Trees (Bagging)",
          "Why chosen: Naturally handles non-linearity, robust to outliers, interpretable via feature importance.",
          "Hyperparams: n_estimators=100, max_depth=12, min_samples_leaf=20",
          "ROC-AUC = 0.6746  |  PR-AUC = 0.6636",
          "Accuracy = 62.04%  |  F1 = 0.6261",
          "Train time: 14.42s",
         ]),
        ("Model 3 (Baseline)\nLogisticRegression (L2)",
         AMBER,
         [
          "Type: L2-Regularised Linear Classifier",
          "Why chosen: Interpretable linear baseline; coefficients directly show feature directionality; fast to train.",
          "Hyperparams: C=0.1 (strong regularisation), max_iter=1000",
          "ROC-AUC = 0.6328  |  PR-AUC = 0.6102",
          "Accuracy = 59.96%  |  F1 = 0.5943",
          "Train time: 3.63s",
         ]),
    ]
    mw = p(3.8); my = p(1.3); mh = p(5.55); mgap = p(0.22)
    for i,(title,col,bullets) in enumerate(models_info):
        lft = p(0.45) + i*(mw+mgap)
        add_rounded(s, lft, my, mw, mh, CARD_BG, col, 1.5)
        add_rect(s, lft, my, mw, p(0.32), col)
        add_text(s, title, lft+p(0.12), my+p(0.04), mw-p(0.22), p(0.32),
                 font_size=9.5, bold=True, color=WHITE)
        tb = s.shapes.add_textbox(lft+p(0.14), my+p(0.42), mw-p(0.26), mh-p(0.55))
        tf = tb.text_frame; tf.word_wrap = True
        for j,b in enumerate(bullets):
            key,_,val = b.partition(" = ") if " = " in b else (b,"","")
            px = tf.paragraphs[0] if j==0 else tf.add_paragraph()
            px.text = f"\u2022  {b}"
            px.font.size = Pt(9.5 if j < 3 else 10.5)
            px.font.bold = (j >= 3)
            px.font.color.rgb = col if j >= 3 else NAVY
            px.space_after = Pt(5)

    # =========================================================================
    # SLIDE 10 — ML: BENCHMARK SCORECARD & ROC/PR CURVES
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_rect(s, p(0), p(0), W, H, LIGHT_BG)
    header_bar(s, "ML Benchmark Scorecard & Model Comparison Curves",
               "HistGradientBoosting Wins on ROC-AUC, PR-AUC, Accuracy & F1")
    footer(s, "Scikit-Learn metrics: roc_auc_score, average_precision_score, f1_score", n=10)

    # Scorecard table (manual)
    headers = ["Model", "ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "F1-Score", "Train(s)"]
    rows = [
        ["HistGradientBoosting*", "0.6913", "0.6836", "63.50%", "62.96%", "63.99%", "0.6347", "12.9s"],
        ["RandomForest",          "0.6746", "0.6636", "62.04%", "61.16%", "64.13%", "0.6261", "14.4s"],
        ["LogisticRegression",    "0.6328", "0.6102", "59.96%", "59.69%", "59.17%", "0.5943",  "3.6s"],
    ]
    row_colors = [RGBColor(219,234,254), CARD_BG, CARD_BG]  # blue highlight for winner
    col_widths  = [p(2.5), p(1.05), p(1.0), p(1.05), p(1.1), p(1.0), p(1.1), p(0.95)]
    row_height  = p(0.42)
    tx = p(0.45); ty = p(1.28)

    # Header row
    x = tx
    for ci,hdr in enumerate(headers):
        add_rect(s, x, ty, col_widths[ci], row_height, DARKBLUE)
        add_text(s, hdr, x+p(0.06), ty+p(0.08), col_widths[ci]-p(0.1), row_height-p(0.1),
                 font_size=9, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        x += col_widths[ci]

    for ri,row in enumerate(rows):
        x = tx
        for ci,cell in enumerate(row):
            add_rect(s, x, ty + (ri+1)*row_height, col_widths[ci], row_height,
                     row_colors[ri], BORDER, 0.5)
            bold = (ri == 0 and ci > 0)
            col  = BLUE if (ri==0 and ci>0) else NAVY
            add_text(s, cell,
                     x+p(0.06), ty+(ri+1)*row_height+p(0.07),
                     col_widths[ci]-p(0.1), row_height-p(0.1),
                     font_size=9.5, bold=bold, color=col, align=PP_ALIGN.CENTER)
            x += col_widths[ci]

    add_text(s, "* PRIMARY MODEL — selected for production deployment",
             tx, ty+4*row_height+p(0.06), p(9.4), p(0.32),
             font_size=8.5, italic=True, color=BLUE)

    # ROC/PR/Metric chart image
    safe_image(s, os.path.join(FIG_MODEL, "fig1_model_comparison.png"),
               p(0.45), p(3.5), p(12.43), p(3.65))

    # =========================================================================
    # SLIDE 11 — ML: DECILE LIFT & CUMULATIVE GAINS
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_rect(s, p(0), p(0), W, H, LIGHT_BG)
    header_bar(s, "Predictive Power: Decile Lift & Cumulative Gains",
               "Targeting Top 20% of Predicted-Risk Customers Captures 30% of All Churners")
    footer(s, "Decile lift methodology: Kumar & Ravi (2007); Internal PoC test set N=20,000", n=11)

    safe_image(s, os.path.join(FIG_MODEL, "fig2_decile_lift.png"),
               p(0.45), p(1.28), p(8.4), p(4.1))

    # Decile table (D1-D5 only for readability)
    decile_data = [
        ("D1 (Highest Risk)", "79.85%", "1.61x", "16.1%"),
        ("D2",                "67.25%", "1.36x", "29.7%"),
        ("D3",                "61.05%", "1.23x", "42.0%"),
        ("D4",                "54.25%", "1.09x", "52.9%"),
        ("D5",                "52.80%", "1.07x", "63.6%"),
        ("D10 (Lowest Risk)", "19.50%", "0.39x", "100%"),
    ]
    dheads = ["Decile", "Churn Rate", "Lift", "Cum. Captured"]
    dcw    = [p(1.85), p(1.4), p(1.1), p(1.85)]
    dtx    = p(9.0); dty = p(1.28); drh = p(0.42)

    x = dtx
    for ci,h in enumerate(dheads):
        add_rect(s, x, dty, dcw[ci], drh, DARKBLUE)
        add_text(s, h, x+p(0.06), dty+p(0.09), dcw[ci]-p(0.1), drh-p(0.1),
                 font_size=9, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        x += dcw[ci]

    for ri,row in enumerate(decile_data):
        x = dtx
        bg = RGBColor(219,234,254) if ri==0 else CARD_BG
        for ci,cell in enumerate(row):
            add_rect(s, x, dty+(ri+1)*drh, dcw[ci], drh, bg, BORDER, 0.5)
            add_text(s, cell,
                     x+p(0.06), dty+(ri+1)*drh+p(0.08), dcw[ci]-p(0.1), drh-p(0.1),
                     font_size=9.5, bold=(ri==0), color=BLUE if ri==0 else NAVY,
                     align=PP_ALIGN.CENTER)
            x += dcw[ci]

    # Score distributions
    safe_image(s, os.path.join(FIG_MODEL, "fig5_score_distributions.png"),
               p(0.45), p(5.42), p(12.43), p(1.72))

    # =========================================================================
    # SLIDE 12 — ML: FEATURE IMPORTANCE (ROOT-CAUSE ATTRIBUTION)
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_rect(s, p(0), p(0), W, H, LIGHT_BG)
    header_bar(s, "Top Predictive Drivers of Churn — Root-Cause Attribution",
               "Permutation Feature Importance | HistGradientBoosting | Test Set N=5,000")
    footer(s, "Permutation Importance: Altmann et al. (2010); Breiman (2001)", n=12)

    safe_image(s, os.path.join(FIG_MODEL, "fig4_feature_importance.png"),
               p(0.45), p(1.28), p(8.5), p(5.85))

    # Insight panel
    bullet_box(s, p(9.15), p(1.28), p(4.0), p(5.85),
               "Actionable Driver Summary",
               [
                "DEVICE OBSOLESCENCE (blue): eqpdays & hnd_price are top structural drivers — hardware upgrades are the highest-ROI fix.",
                "USAGE CONTRACTION (red): usage_drop_ratio & mou_Mean signal disengagement 30-60 days before churn.",
                "PLAN & BILLING (amber): totmrc_Mean & overage_ratio — plan right-sizing reduces bill-shock churn.",
                "TENURE (green): months — targeting contract renewal windows (13-24 mos) with proactive offers.",
                "ACCOUNT LIMITS (purple): asl_flag — credit-restricted accounts exhibit elevated churn; special plan options needed.",
               ],
               title_color=DARKBLUE, title_size=11, bullet_size=9.5)

    # =========================================================================
    # SLIDE 13 — BUSINESS PROPOSAL & ROI
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_rect(s, p(0), p(0), W, H, LIGHT_BG)
    header_bar(s, "Business Proposal: Project SmartRetain",
               "ML-Guided Proactive Retention — $1.41M Net Annual Profit | 157% ROI | 4.7-Month Payback")
    footer(s, "Bain & Co. (2020) retention benchmarks; McKinsey (2021) ML save rate; Internal ROI model", n=13)

    # ROI figures
    safe_image(s, os.path.join(FIG_EDA, "fig7_revenue_distribution.png"),
               p(0.45), p(1.28), p(5.9), p(2.8))

    # KPI row
    roi_kpis = [
        ("Target Cohort",        "20,000",      "Top 2 risk deciles (20% of base)",      BLUE),
        ("Customers Saved",      "3,285 / yr",  "22.5% save rate (McKinsey benchmark)",  GREEN),
        ("Gross Rev Preserved",  "$2.31M / yr", "3,285 × $704.64 annual ARPU",           GREEN),
        ("Campaign Cost",        "$900K",       "$45/offer × 20,000 targeted",           AMBER),
        ("Net Profit Lift",      "+$1.41M/yr",  "Revenue saved minus campaign cost",     GREEN),
        ("ROI",                  "157.2%",      "Payback period: 4.7 months",            TEAL),
    ]
    rw = p(2.08); ry = p(4.22); rh = p(1.05); rgap = p(0.14)
    for i,(lbl,val,sub,col) in enumerate(roi_kpis):
        kpi_card(s, p(0.45)+i*(rw+rgap), ry, rw, rh,
                 lbl, val, sub, label_color=col, value_color=col)

    # 3-phase implementation plan
    phases = [
        ("Phase 1: Deploy (Wks 1-4)",
         ["Productionise HistGradientBoosting scoring pipeline",
          "Score all 100k customers weekly; flag top-2-decile cohort",
          "Integrate output into CRM retention dashboard"],
         BLUE),
        ("Phase 2: Intervene (Wks 5-12)",
         ["Personalised outreach: device upgrade subsidy, bill credit, plan right-sizing",
          "Priority: aged-device + high-overage + usage-declining segments",
          "Track save rate vs. control group weekly"],
         TEAL),
        ("Phase 3: Scale (Mos 4-12)",
         ["Full 1M subscriber deployment: $14.1M projected net profit",
          "Embed model in real-time event streaming for instant risk alerts",
          "Quarterly model refresh with new behavioural data"],
         GREEN),
    ]
    pw = p(3.8); py = p(1.3); ph = p(2.72); pgap = p(0.22)
    for i,(title,bullets,col) in enumerate(phases):
        lft = p(6.55) + i*(pw+pgap)
        bullet_box(s, lft, py, pw, ph, title, bullets,
                   title_color=col, title_size=10.5, bullet_size=9.5)

    # =========================================================================
    # SLIDE 14 — SCALED ENTERPRISE IMPACT & RISK
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_rect(s, p(0), p(0), W, H, LIGHT_BG)
    header_bar(s, "Enterprise Scale Impact & Risk Considerations",
               "From 100k PoC to 1M Subscriber Deployment")
    footer(s, "McKinsey (2021); Bain & Co. (2020); GDPR/CCPA compliance requirements", n=14)

    # Scale comparison
    scale_items = [
        ("PoC Scale\n(100k Subscribers)", [
            "Targeted Cohort: 20,000 accounts",
            "Churners Saved: 3,285 / year",
            "Gross Revenue Preserved: $2.31M / year",
            "Campaign Investment: $900K",
            "Net Profit Lift: +$1.41M / year",
            "ROI: 157% | Payback: 4.7 months",
        ], BLUE),
        ("Enterprise Scale\n(1M Subscribers)", [
            "Targeted Cohort: 200,000 accounts",
            "Churners Saved: 32,850 / year",
            "Gross Revenue Preserved: $23.1M / year",
            "Campaign Investment: $9.0M",
            "Net Profit Lift: +$14.1M / year",
            "ROI: 157% | Payback: 4.7 months",
        ], GREEN),
        ("Risk & Mitigation", [
            "Model drift: Re-train quarterly on fresh data",
            "Data privacy: GDPR / CCPA compliant scoring pipeline (no PII in features)",
            "Offer saturation: A/B test offer types; cap contact frequency",
            "Class imbalance: Monitor PR-AUC, not just accuracy",
            "False positives: Untargeted offers waste budget — decile cap limits exposure",
        ], AMBER),
    ]
    sw2 = p(3.8); sy2 = p(1.3); sh2 = p(5.55); sgap2 = p(0.22)
    for i,(title,bullets,col) in enumerate(scale_items):
        lft = p(0.45) + i*(sw2+sgap2)
        bullet_box(s, lft, sy2, sw2, sh2, title, bullets,
                   title_color=col, title_size=11, bullet_size=9.5)

    # Confusion matrices at bottom
    add_text(s, "Model Confusion Matrices (Test Set, N=20,000):",
             p(0.45), p(7.0), p(5.5), p(0.28),
             font_size=9, bold=True, color=DARKBLUE)

    # =========================================================================
    # SLIDE 15 — REFERENCES
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_rect(s, p(0), p(0), W, H, LIGHT_BG)
    header_bar(s, "References & Citations",
               "All External Data, Methods & Benchmarks Used in This Proposal")
    footer(s, n=15)

    refs = {
        "Market & Industry Data": [
            "[1]  Bain & Company. (2020). Prescription for Cutting Costs: Reducing customer defection. Bain & Company Insights.",
            "[2]  McKinsey & Company. (2021). The value of getting personalization right—or wrong—is multiplying. McKinsey Digital.",
            "[3]  GSMA Intelligence. (2023). State of Mobile Internet Connectivity Report 2023. GSMA.",
            "[4]  Ericsson. (2024). Ericsson Mobility Report, June 2024. Ericsson AB.",
            "[5]  Federal Communications Commission. (2022). FCC Mobile Wireless Competition Report. FCC.",
        ],
        "Machine Learning & Statistical Methods": [
            "[6]  Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324",
            "[7]  Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 29(5), 1189-1232.",
            "[8]  Altmann, A., Tolosi, L., Sander, O., & Lengauer, T. (2010). Permutation importance: a corrected feature importance measure. Bioinformatics, 26(10), 1340-1347.",
            "[9]  Kumar, D. A., & Ravi, V. (2007). Predicting credit card customer churn using data mining techniques. International Journal of Data Analysis Techniques and Strategies, 1(1), 4-28.",
            "[10] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825-2830. https://scikit-learn.org",
        ],
        "Dataset": [
            "[11] Client.csv & Record.csv. (2024). Provided by client as part of IT Consulting PoC engagement.",
            "     Variables include: customer demographics, device attributes, billing records, call usage, and churn labels.",
            "     N = 100,000 customers | 50 client features | 50 usage/billing features | Merged on Customer_ID.",
        ],
    }
    tx = p(0.5); ty2 = p(1.3); col_w = p(6.0)
    left_refs  = list(refs.items())[:2]
    right_refs = list(refs.items())[2:]

    # Left column
    tb_l = s.shapes.add_textbox(tx, ty2, col_w, p(5.7))
    tf_l = tb_l.text_frame; tf_l.word_wrap = True
    for sec,(title,items) in enumerate(left_refs):
        p_ = tf_l.paragraphs[0] if (sec==0) else tf_l.add_paragraph()
        p_.text = title
        p_.font.size = Pt(11); p_.font.bold = True
        p_.font.color.rgb = DARKBLUE
        p_.space_before = Pt(8 if sec>0 else 0)
        for item in items:
            pi = tf_l.add_paragraph()
            pi.text = item
            pi.font.size = Pt(8.8); pi.font.color.rgb = NAVY
            pi.space_after = Pt(2)

    # Right column
    tb_r = s.shapes.add_textbox(tx+col_w+p(0.5), ty2, col_w+p(0.3), p(5.7))
    tf_r = tb_r.text_frame; tf_r.word_wrap = True
    for sec,(title,items) in enumerate(right_refs):
        p_ = tf_r.paragraphs[0] if sec==0 else tf_r.add_paragraph()
        p_.text = title
        p_.font.size = Pt(11); p_.font.bold = True
        p_.font.color.rgb = DARKBLUE
        p_.space_before = Pt(8 if sec>0 else 0)
        for item in items:
            pi = tf_r.add_paragraph()
            pi.text = item
            pi.font.size = Pt(8.8); pi.font.color.rgb = NAVY
            pi.space_after = Pt(2)

    add_text(s,
             "Note: All analysis, models, and visualisations in this presentation are original work "
             "performed on the client-provided dataset. External benchmarks are cited above. "
             "Model results are reproducible from new_eda.py and new_test.py.",
             p(0.5), p(6.95), p(12.33), p(0.45),
             font_size=8.5, italic=True, color=SLATE)

    # =========================================================================
    # SAVE
    # =========================================================================
    out = "SmartRetain_Executive_Proposal.pptx"
    prs.save(out)
    print(f"\n  -> Saved: {out}")
    return out


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 68)
    print("  PROJECT SMARTRETAIN — 15-SLIDE PPTX GENERATOR (new_pptx.py)")
    print("=" * 68)

    # Verify figure folders exist
    missing = []
    for folder in [FIG_EDA, FIG_MODEL]:
        if not os.path.isdir(folder):
            missing.append(folder)
    if missing:
        print(f"\n  WARNING: Figure folder(s) not found: {missing}")
        print("  Run new_eda.py and new_test.py first to generate figures.")
        print("  Continuing — missing figures will show placeholder boxes.\n")

    out_file = build()
    print(f"\n  15 slides written to: {os.path.abspath(out_file)}")
    print("=" * 68 + "\n")

