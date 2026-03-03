"""
core/score.py
─────────────
Composite fluency scoring formula grounded in L2 acquisition research.

Formula
───────
wpm_score    = clip(normalize(WPM, 60, 180) × 100, 0, 100)
pause_score  = clip((1 − pause_rate / 10) × 100, 0, 100)
filler_score = clip((1 − filler_rate / 10) × 100, 0, 100)

Fluency_Score = 0.40 × wpm_score + 0.35 × pause_score + 0.25 × filler_score

Weights reflect relative effect sizes in:
  Lennon (1990)         — WPM as primary fluency proxy
  Tavakoli & Skehan (2005) — pause rate as strong secondary predictor
  Skehan (1996)         — filler rate as planning difficulty marker

Thresholds
──────────
  0–49   Developing
 50–67   Emerging
 68–79   Proficient  (call center entry)
 80–100  Professional
"""

from __future__ import annotations
import re
import numpy as np


# ── FILLER DETECTION ─────────────────────────────────────────────────────────

FILLER_RE = re.compile(
    r"\b(uh+|um+|like|you know|basically|so|right|hmm+|err+|ah+|well)\b",
    re.IGNORECASE,
)


def detect_fillers(transcript: str) -> list:
    """Return list of re.Match objects for every filler in the transcript."""
    return list(FILLER_RE.finditer(transcript))


def annotate_transcript(transcript: str, filler_matches: list) -> str:
    """
    Return HTML string with filler words wrapped in a highlight span.
    Safe for st.markdown with unsafe_allow_html=True.
    """
    if not filler_matches:
        return transcript
    html, last = "", 0
    for m in filler_matches:
        html += transcript[last:m.start()]
        html += f'<span class="filler">{m.group()}</span>'
        last = m.end()
    html += transcript[last:]
    return html


# ── SCORING ───────────────────────────────────────────────────────────────────

def _norm(value: float, lo: float, hi: float) -> float:
    return float(np.clip((value - lo) / (hi - lo) * 100, 0, 100))


def compute_scores(
    wpm: float,
    pause_rate: float,
    filler_rate: float,
) -> dict[str, float]:
    """
    Compute all component scores and the composite Fluency Score.

    Returns dict with keys:
        fluency, wpm_score, pause_score, filler_score
    """
    wpm_s    = _norm(wpm, 60, 180)
    pause_s  = float(np.clip((1 - pause_rate / 10) * 100, 0, 100))
    filler_s = float(np.clip((1 - filler_rate / 10) * 100, 0, 100))
    composite = 0.40 * wpm_s + 0.35 * pause_s + 0.25 * filler_s
    return {
        "fluency":      round(composite, 1),
        "wpm_score":    round(wpm_s, 1),
        "pause_score":  round(pause_s, 1),
        "filler_score": round(filler_s, 1),
    }


# ── LABELS & THRESHOLDS ───────────────────────────────────────────────────────

BENCHMARKS = {
    "Call Center Entry":    {"min_score": 68, "wpm": (120, 160), "pause_pm": 4.0, "filler_pm": 3.0},
    "Professional":         {"min_score": 80, "wpm": (140, 165), "pause_pm": 2.5, "filler_pm": 1.5},
    "Native Casual":        {"min_score": 75, "wpm": (130, 180), "pause_pm": 4.5, "filler_pm": 3.5},
}


def score_label(score: float) -> str:
    if score < 50:  return "Developing"
    if score < 68:  return "Emerging"
    if score < 80:  return "Proficient"
    return "Professional"


def score_css(score: float) -> str:
    if score < 50:  return "score-low"
    if score < 68:  return "score-mid"
    if score < 80:  return "score-high"
    return "score-pro"


def score_color(score: float) -> str:
    if score < 50:  return "#FF6B35"
    if score < 68:  return "#FFD166"
    if score < 80:  return "#00D4AA"
    return "#00f0c0"
