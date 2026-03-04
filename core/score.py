"""
core/score.py
─────────────
Fluency scoring formula and extended acoustic feature scoring.

Core composite formula (unchanged — weighted per literature):
  Fluency = 0.40 × wpm_score + 0.35 × pause_score + 0.25 × filler_score

Extended features (displayed separately, not in composite):
  articulation_rate  — WPM over speech-only time (Kormos & Denes, 2004)
  pitch_variation    — F0 std-dev score (Hincks, 2005)
  confidence         — Whisper avg_logprob normalised (proxy for articulation clarity)

References
──────────
  Lennon (1990)            — WPM as primary fluency proxy
  Skehan (1996)            — filler rate as planning difficulty marker
  Tavakoli & Skehan (2005) — pause rate as secondary predictor
  Kormos & Denes (2004)    — articulation rate vs speech rate distinction
  Hincks (2005)            — pitch variation as fluency and engagement signal
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
    return list(FILLER_RE.finditer(transcript))


def annotate_transcript(transcript: str, matches: list) -> str:
    if not matches:
        return transcript
    html, last = "", 0
    for m in matches:
        html += transcript[last:m.start()]
        html += f'<span class="filler">{m.group()}</span>'
        last = m.end()
    html += transcript[last:]
    return html


# ── CORE SCORING ─────────────────────────────────────────────────────────────

def _norm(v: float, lo: float, hi: float) -> float:
    return float(np.clip((v - lo) / (hi - lo) * 100, 0, 100))


def compute_scores(
    wpm: float,
    pause_rate: float,
    filler_rate: float,
) -> dict[str, float]:
    """
    Composite Fluency Score (0–100) from three core features.
    Weights are fixed to match literature-cited values.
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


def compute_articulation_score(articulation_rate: float) -> float:
    """
    Score articulation rate (WPM over speech-only time).
    Target range: 150–200 WPM (faster than speech rate because pauses excluded).
    Kormos & Denes (2004).
    """
    return round(_norm(articulation_rate, 80, 220), 1)


# ── BENCHMARKS ───────────────────────────────────────────────────────────────

BENCHMARKS = {
    "Call Center Entry":  {"min_score": 68, "wpm": (120, 160), "pause_pm": 4.0, "filler_pm": 3.0},
    "Professional":       {"min_score": 80, "wpm": (140, 165), "pause_pm": 2.5, "filler_pm": 1.5},
    "Native Casual":      {"min_score": 75, "wpm": (130, 180), "pause_pm": 4.5, "filler_pm": 3.5},
}


def score_label(s: float) -> str:
    if s < 50:  return "Developing"
    if s < 68:  return "Emerging"
    if s < 80:  return "Proficient"
    return "Professional"


def score_css(s: float) -> str:
    if s < 50:  return "score-low"
    if s < 68:  return "score-mid"
    if s < 80:  return "score-high"
    return "score-pro"


def score_color(s: float) -> str:
    if s < 50:  return "#FF6B35"
    if s < 68:  return "#FFD166"
    if s < 80:  return "#00D4AA"
    return "#00f0c0"
