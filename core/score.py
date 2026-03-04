"""
core/score.py
─────────────
Composite fluency scoring + linguistic quality analysis.

Core formula (unchanged):
  Fluency = 0.40 × wpm_score + 0.35 × pause_score + 0.25 × filler_score

New additions (FS document integration):
  detect_discourse_connectors()  — sequencing / cohesion markers (Schmidt 1990)
  score_discourse_coherence()    — penalise absence of connectors in >30s speech
  annotate_transcript_full()     — highlights both fillers AND connectors

Research basis:
  Lennon (1990)            — WPM as primary fluency proxy
  Skehan (1996)            — filler rate as planning difficulty marker
  Tavakoli & Skehan (2005) — pause rate as secondary predictor
  Schmidt (1990)           — Noticing Hypothesis: connectors signal discourse planning
  Celce-Murcia et al.      — Discourse competence as CEFR C1/C2 marker
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
    """Return list of re.Match objects for every filler word in the transcript."""
    return list(FILLER_RE.finditer(transcript))


# ── DISCOURSE CONNECTOR DETECTION ────────────────────────────────────────────

# Grouped by function — mirrors the FS lesson structure
CONNECTORS = {
    "sequencing":    ["first", "second", "third", "then", "next", "after that",
                      "finally", "lastly", "to begin with", "to start with"],
    "contrast":      ["but", "however", "although", "even though", "on the other hand",
                      "despite", "in contrast", "nevertheless", "yet", "still"],
    "cause_effect":  ["because", "so", "therefore", "as a result", "that is why",
                      "due to", "since", "consequently", "thus"],
    "addition":      ["also", "and", "moreover", "furthermore", "in addition",
                      "besides", "not only", "as well as"],
    "example":       ["for example", "for instance", "such as", "like", "including"],
    "summary":       ["in conclusion", "to sum up", "overall", "in short",
                      "to summarize", "all in all"],
}

# Build flat regex from all groups
_all_connectors = [c for group in CONNECTORS.values() for c in group]
# Sort longest first to avoid partial matches (e.g. "in addition" before "in")
_all_connectors.sort(key=len, reverse=True)
CONNECTOR_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in _all_connectors) + r")\b",
    re.IGNORECASE,
)


def detect_discourse_connectors(transcript: str) -> dict:
    """
    Detect discourse connectors and classify by function.

    Returns dict:
        matches    — list of re.Match objects (for annotation)
        by_type    — {type: [word, ...]}
        count      — total unique connector instances
        types_used — number of different connector types present
    """
    matches = list(CONNECTOR_RE.finditer(transcript))
    by_type: dict[str, list[str]] = {k: [] for k in CONNECTORS}

    for m in matches:
        word = m.group().lower()
        for ctype, words in CONNECTORS.items():
            if word in words:
                by_type[ctype].append(word)
                break

    return {
        "matches":    matches,
        "by_type":    by_type,
        "count":      len(matches),
        "types_used": sum(1 for v in by_type.values() if v),
    }


def score_discourse_coherence(
    connector_data: dict,
    duration_s: float,
    word_count: int,
) -> float:
    """
    Score discourse coherence 0–100.

    Rationale (Celce-Murcia et al.):
      - Speech under 30 s: connectors optional, score neutral (60)
      - 30–60 s with 0 connectors: poor planning signal (< 40)
      - 30–60 s with 1–2 types: emerging (50–70)
      - 30–60 s with 3+ types: proficient (75–100)
      - > 60 s needs proportionally more
    """
    if duration_s < 25 or word_count < 30:
        return 60.0   # too short to evaluate meaningfully

    types_used = connector_data["types_used"]
    count      = connector_data["count"]

    # Scale target by speech duration
    target_count = max(2, duration_s / 20)   # ~1 connector per 20 s
    count_score  = float(np.clip(count / target_count * 80, 0, 85))
    type_bonus   = min(15.0, types_used * 4.0)   # up to +15 for variety

    return round(min(100.0, count_score + type_bonus), 1)


# ── TRANSCRIPT ANNOTATION ─────────────────────────────────────────────────────

def annotate_transcript(transcript: str, filler_matches: list) -> str:
    """Annotate fillers only (backward-compatible)."""
    if not filler_matches:
        return transcript
    html, last = "", 0
    for m in filler_matches:
        html += transcript[last:m.start()]
        html += f'<span class="filler">{m.group()}</span>'
        last = m.end()
    html += transcript[last:]
    return html


def annotate_transcript_full(
    transcript: str,
    filler_matches: list,
    connector_matches: list,
) -> str:
    """
    Annotate transcript with both fillers (orange) and connectors (teal).
    Handles overlapping spans gracefully (fillers take priority).
    """
    # Build a combined, sorted list of all spans
    events: list[tuple] = []
    for m in filler_matches:
        events.append((m.start(), m.end(), "filler", m.group()))
    for m in connector_matches:
        # Skip if overlaps with a filler
        overlap = any(f.start() <= m.start() < f.end() or
                      m.start() <= f.start() < m.end()
                      for f in filler_matches)
        if not overlap:
            events.append((m.start(), m.end(), "connector", m.group()))

    events.sort(key=lambda x: x[0])

    html, last = "", 0
    for start, end, tag, text in events:
        if start < last:
            continue
        html += transcript[last:start]
        if tag == "filler":
            html += f'<span class="filler">{text}</span>'
        else:
            html += f'<span class="connector">{text}</span>'
        last = end
    html += transcript[last:]
    return html


# ── GRAMMAR ERROR DETECTION (lightweight, no external API) ───────────────────

GRAMMAR_PATTERNS = [
    # Subject-verb agreement
    (re.compile(r"\bhe (go|come|eat|run|play|work|do|have)\b", re.I),
     "he go → he goes  (third-person -s)"),
    (re.compile(r"\bshe (go|come|eat|run|play|work|do|have)\b", re.I),
     "she go → she goes  (third-person -s)"),
    # Missing article
    (re.compile(r"\b(i went to|i go to|i am at) (school|hospital|gym|church|work|home)\b(?! of)", re.I),
     "to school is correct — no article needed with these nouns"),
    # Wrong tense marker
    (re.compile(r"\byesterday i (go|come|eat|play|work|run)\b", re.I),
     "Yesterday + simple past: 'I went', 'I came', etc."),
    # Double negative
    (re.compile(r"\bdon't (never|nothing|nobody|no one)\b", re.I),
     "Double negative — use 'don't ever' or 'never' alone"),
]


def detect_grammar_issues(transcript: str) -> list[dict]:
    """
    Lightweight grammar checker for the most common L2 errors.
    Returns list of {match_text, rule, suggestion}.
    Does NOT replace Groq — used for instant rule-based feedback.
    """
    issues = []
    for pattern, rule in GRAMMAR_PATTERNS:
        for m in pattern.finditer(transcript):
            issues.append({
                "text":       m.group(),
                "rule":       rule,
                "start":      m.start(),
                "end":        m.end(),
            })
    return issues


# ── CORE SCORING ──────────────────────────────────────────────────────────────

def _norm(value: float, lo: float, hi: float) -> float:
    return float(np.clip((value - lo) / (hi - lo) * 100, 0, 100))


def compute_scores(
    wpm: float,
    pause_rate: float,
    filler_rate: float,
) -> dict[str, float]:
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
