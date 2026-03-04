"""
core/report.py
──────────────
Branded one-page PDF fluency report using reportlab.

Design
──────
  - A4 page, dark header strip (navy), teal accents
  - User identity + timestamp at top
  - Score gauge (text-based, no images required)
  - Three core feature bars + three extended feature bars
  - Benchmark table
  - Coaching priority + top exercises
  - Research citations at bottom
  - Fully self-contained — no external fonts or assets required

Usage
──────
    from core.report import build_pdf
    pdf_bytes = build_pdf(result, feedback, username, benchmark)
    st.download_button("Download PDF", pdf_bytes, file_name="report.pdf")
"""

from __future__ import annotations
import io
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ── PALETTE ───────────────────────────────────────────────────────────────────
NAVY    = colors.HexColor("#0D1F35")
PANEL   = colors.HexColor("#0a1726")
CARD    = colors.HexColor("#162030")
ACCENT  = colors.HexColor("#00D4AA")
ACCENT2 = colors.HexColor("#FF6B35")
GOLD    = colors.HexColor("#FFD166")
LIGHT   = colors.HexColor("#F0F4F8")
GRAY    = colors.HexColor("#8892A4")
WHITE   = colors.white

W, H = A4   # 210mm × 297mm


# ── STYLES ────────────────────────────────────────────────────────────────────

def _styles():
    base = ParagraphStyle("base", fontName="Helvetica", fontSize=8,
                           textColor=GRAY, leading=12)
    return {
        "base":    base,
        "title":   ParagraphStyle("title",  parent=base, fontName="Helvetica-Bold",
                                  fontSize=18, textColor=ACCENT, leading=22),
        "sub":     ParagraphStyle("sub",    parent=base, fontSize=8,
                                  textColor=GRAY, leading=11),
        "h2":      ParagraphStyle("h2",     parent=base, fontName="Helvetica-Bold",
                                  fontSize=10, textColor=LIGHT, leading=14,
                                  spaceAfter=4),
        "h3":      ParagraphStyle("h3",     parent=base, fontName="Helvetica-Bold",
                                  fontSize=8,  textColor=ACCENT, leading=12,
                                  spaceBefore=6),
        "body":    ParagraphStyle("body",   parent=base, fontSize=8,
                                  textColor=LIGHT, leading=12),
        "small":   ParagraphStyle("small",  parent=base, fontSize=6.5,
                                  textColor=GRAY, leading=10),
        "score":   ParagraphStyle("score",  parent=base, fontName="Helvetica-Bold",
                                  fontSize=42, textColor=ACCENT,
                                  alignment=TA_CENTER, leading=48),
        "label":   ParagraphStyle("label",  parent=base, fontName="Helvetica-Bold",
                                  fontSize=10, textColor=LIGHT,
                                  alignment=TA_CENTER, leading=14),
        "center":  ParagraphStyle("center", parent=base, alignment=TA_CENTER,
                                  textColor=GRAY, fontSize=7),
        "cite":    ParagraphStyle("cite",   parent=base, fontSize=6,
                                  textColor=GRAY, leading=9),
    }


def _bar(score: float, width_pts: float = 200) -> Table:
    """Render a simple horizontal score bar as a Table."""
    fill_w  = max(2.0, score / 100 * width_pts)
    empty_w = width_pts - fill_w

    if score >= 68:
        bar_color = ACCENT
    elif score >= 50:
        bar_color = GOLD
    else:
        bar_color = ACCENT2

    data = [["", ""]]
    t = Table(data, colWidths=[fill_w, empty_w], rowHeights=[5])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), bar_color),
        ("BACKGROUND", (1, 0), (1, 0), CARD),
        ("LINEBELOW",  (0, 0), (-1, -1), 0, NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
    ]))
    return t


# ── BUILD ─────────────────────────────────────────────────────────────────────

def build_pdf(
    result: dict,
    feedback: dict,
    username: str,
    benchmark: str,
    top_sorted: list[tuple],
) -> bytes:
    """
    Generate a branded A4 PDF fluency report.
    Returns raw PDF bytes suitable for st.download_button.
    """
    from core.score import score_label, BENCHMARKS

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
    )

    S   = _styles()
    sc  = result["fluency_score"]
    bm  = BENCHMARKS[benchmark]
    lbl = score_label(sc)
    ts  = datetime.now().strftime("%B %d, %Y  %H:%M")
    dur_m = int(result["duration_s"] // 60)
    dur_s = int(result["duration_s"] % 60)

    story = []
    usable_w = W - 36 * mm   # ~537 pts

    # ── HEADER STRIP ──────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("SPEECH FLUENCY ANALYZER", ParagraphStyle(
            "hdr", fontName="Helvetica-Bold", fontSize=9,
            textColor=ACCENT, leading=12)),
        Paragraph(f"Project 03 · Diego Palencia Research", ParagraphStyle(
            "hdr2", fontName="Helvetica", fontSize=7,
            textColor=GRAY, leading=12, alignment=TA_RIGHT)),
    ]]
    header_t = Table(header_data, colWidths=[usable_w * 0.6, usable_w * 0.4])
    header_t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (0,  -1), 10),
        ("RIGHTPADDING",  (-1, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(header_t)
    story.append(Spacer(1, 8))

    # ── USER + DATE ───────────────────────────────────────────────────────────
    meta_data = [[
        Paragraph(f"<b>Candidate:</b>  {username}", S["body"]),
        Paragraph(f"<b>Benchmark:</b>  {benchmark}", S["body"]),
        Paragraph(f"<b>Generated:</b>  {ts}", S["body"]),
        Paragraph(f"<b>Duration:</b>  {dur_m}m {dur_s}s", S["body"]),
    ]]
    meta_t = Table(meta_data, colWidths=[usable_w / 4] * 4)
    meta_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), CARD),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("LINEBELOW",     (0, 0), (-1, -1), 0.5, PANEL),
    ]))
    story.append(meta_t)
    story.append(Spacer(1, 10))

    # ── SCORE + FEATURE BREAKDOWN ─────────────────────────────────────────────
    score_col = [
        Paragraph(f"{sc}", S["score"]),
        Paragraph(lbl.upper(), S["label"]),
        Spacer(1, 4),
        Paragraph(f"{'ABOVE' if sc >= bm['min_score'] else 'BELOW'} {benchmark.upper()} TARGET (≥ {bm['min_score']})",
                  ParagraphStyle("delta", parent=S["center"], fontSize=6.5,
                                 textColor=ACCENT if sc >= bm["min_score"] else ACCENT2)),
    ]

    feat_items = [
        ("Speaking Rate",    f"{result['wpm']:.0f} WPM",         result["wpm_score"],    "0.40"),
        ("Pause Control",    f"{result['pause_rate']:.1f}/min",   result["pause_score"],  "0.35"),
        ("Filler Words",     f"{result['filler_rate']:.1f}/min",  result["filler_score"], "0.25"),
    ]

    feat_rows = []
    bar_w = usable_w * 0.58 * 0.72  # proportional bar width

    for name, val, raw, weight in feat_items:
        feat_rows.append([
            Paragraph(f"<b>{name}</b>", S["h3"]),
            Paragraph(val, S["small"]),
            Paragraph(f"<b>{raw:.0f}</b>/100", S["small"]),
        ])
        feat_rows.append([_bar(raw, bar_w), "", ""])

    feat_t = Table(
        feat_rows,
        colWidths=[usable_w * 0.32, usable_w * 0.15, usable_w * 0.11],
    )
    feat_t.setStyle(TableStyle([
        ("SPAN",          (0, i * 2 + 1), (-1, i * 2 + 1)) for i in range(len(feat_items))
    ] + [
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
    ]))

    top_row = Table(
        [[score_col, feat_t]],
        colWidths=[usable_w * 0.32, usable_w * 0.68],
    )
    top_row.setStyle(TableStyle([
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND",  (0, 0), (0, 0),  CARD),
        ("TOPPADDING",  (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0,0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",(0, 0), (-1, -1), 6),
    ]))
    story.append(top_row)
    story.append(Spacer(1, 8))

    # ── EXTENDED FEATURES ─────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=PANEL))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Extended Acoustic Analysis", S["h2"]))

    ext_items = [
        ("Articulation Rate",
         f"{result.get('articulation_rate', 0):.0f} WPM (speech-only)",
         result.get("art_score", 50.0),
         "WPM excluding pauses — Kormos & Denes (2004)"),
        ("Pitch Variation",
         f"F0 std dev: {result.get('pitch_std', 0):.0f} Hz",
         result.get("pitch_score", 50.0),
         "F0 standard deviation — Hincks (2005)"),
        ("Transcription Confidence",
         f"Whisper logprob score",
         result.get("confidence", 50.0),
         "Phoneme-level clarity proxy — Whisper avg_logprob"),
    ]

    ext_bar_w = usable_w * 0.4

    ext_rows = []
    for name, val, raw, note in ext_items:
        ext_rows.append([
            Paragraph(f"<b>{name}</b>", S["h3"]),
            _bar(raw, ext_bar_w),
            Paragraph(f"{raw:.0f}/100", S["small"]),
            Paragraph(val, S["small"]),
        ])

    ext_t = Table(
        ext_rows,
        colWidths=[usable_w * 0.22, usable_w * 0.37, usable_w * 0.08, usable_w * 0.33],
    )
    ext_t.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.3, PANEL),
    ]))
    story.append(ext_t)
    story.append(Spacer(1, 8))

    # ── BENCHMARKS ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=PANEL))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Benchmark Comparison", S["h2"]))

    from core.score import BENCHMARKS
    bm_rows = [
        [Paragraph("<b>Standard</b>",       S["h3"]),
         Paragraph("<b>Min Score</b>",       S["h3"]),
         Paragraph("<b>Your Score</b>",      S["h3"]),
         Paragraph("<b>Result</b>",          S["h3"])],
    ]
    for bm_name, bm_vals in BENCHMARKS.items():
        passes = sc >= bm_vals["min_score"]
        c = ACCENT if passes else ACCENT2
        marker = "PASS" if passes else "BELOW"
        bm_rows.append([
            Paragraph(f"{'> ' if bm_name == benchmark else ''}{bm_name}", S["body"]),
            Paragraph(f">= {bm_vals['min_score']}", S["body"]),
            Paragraph(f"{sc}", S["body"]),
            Paragraph(f"<b>{marker}</b>",
                      ParagraphStyle("res", parent=S["body"], textColor=c)),
        ])

    bm_t = Table(bm_rows, colWidths=[usable_w * 0.4, usable_w * 0.18,
                                      usable_w * 0.18, usable_w * 0.24])
    bm_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), PANEL),
        ("BACKGROUND",    (0, 1), (-1, -1), CARD),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [CARD, colors.HexColor("#111d2c")]),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("LINEBELOW",     (0, 0), (-1, -1), 0.3, PANEL),
    ]))
    story.append(bm_t)
    story.append(Spacer(1, 8))

    # ── COACHING ──────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=PANEL))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Coaching Summary", S["h2"]))
    story.append(Paragraph(feedback.get("overall", ""), S["body"]))
    story.append(Spacer(1, 4))

    coaching_cols = []
    for key, title in [("wpm", "Speaking Rate"), ("pauses", "Pause Control"), ("fillers", "Filler Words")]:
        d = feedback.get(key, {})
        coaching_cols.append([
            Paragraph(f"<b>{title}</b>", S["h3"]),
            Paragraph(d.get("advice", ""), S["small"]),
        ])

    coach_t = Table(
        [coaching_cols],
        colWidths=[usable_w / 3] * 3,
    )
    coach_t.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND",    (0, 0), (-1, -1), CARD),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("LINEAFTER",     (0, 0), (-2, -1), 0.5, PANEL),
    ]))
    story.append(coach_t)
    story.append(Spacer(1, 4))

    # Priority + exercises
    priority_text = feedback.get("priority", "")
    story.append(Paragraph(f"<b>Priority focus:</b>  {priority_text}", S["body"]))
    story.append(Spacer(1, 4))

    exercises = feedback.get("exercises", [])[:3]
    ex_rows = [[Paragraph(ex, S["small"])] for ex in exercises]
    if ex_rows:
        ex_t = Table(ex_rows, colWidths=[usable_w])
        ex_t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), CARD),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("LINEBELOW",     (0, 0), (-1, -2), 0.2, PANEL),
        ]))
        story.append(ex_t)

    story.append(Spacer(1, 8))

    # ── TRANSCRIPT SNIPPET ────────────────────────────────────────────────────
    transcript = result.get("transcript", "")
    if transcript:
        story.append(HRFlowable(width="100%", thickness=0.5, color=PANEL))
        story.append(Spacer(1, 4))
        story.append(Paragraph("Transcript (excerpt)", S["h2"]))
        snippet = transcript[:400] + ("…" if len(transcript) > 400 else "")
        story.append(Paragraph(snippet, S["small"]))
        story.append(Spacer(1, 6))

    # ── FORMULA + CITATIONS ───────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=PANEL))
    story.append(Spacer(1, 4))

    formula = "Fluency Score = (0.40 x WPM score) + (0.35 x Pause score) + (0.25 x Filler score)"
    story.append(Paragraph(formula, S["cite"]))
    story.append(Spacer(1, 3))

    refs = [
        "Lennon (1990). Investigating fluency in EFL. Language Learning, 40(3).",
        "Skehan (1996). Task-based instruction. Applied Linguistics, 17(1).",
        "Tavakoli & Skehan (2005). Strategic planning and task performance.",
        "Kormos & Denes (2004). Exploring measures of fluency. System, 32(1).",
        "Hincks (2005). Measures and perceptions of liveliness. Interspeech.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, S["cite"]))

    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Speech Fluency Analyzer · Project 03 · Diego Jose Palencia Robles · "
        "github.com/diegopalencia-research · Guatemala City · 2025",
        ParagraphStyle("footer", parent=S["cite"], alignment=TA_CENTER, textColor=GRAY),
    ))

    doc.build(story)
    return buf.getvalue()
