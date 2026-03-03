"""
core/feedback.py
────────────────
Two-tier feedback system:

TIER 1 — Rule-based engine (always free, always available)
    Analyses all features and produces specific, actionable coaching advice.
    No API key required.

TIER 2 — Groq AI coaching (optional, free API tier available)
    Sends transcript + metrics to Groq (llama-3.3-70b) for personalised,
    paragraph-level narrative coaching.
    Requires GROQ_API_KEY (free at console.groq.com — no credit card needed).

The caller receives both outputs when Groq is available; only Tier 1 otherwise.
"""

from __future__ import annotations
from core.score import BENCHMARKS


# ── TIER 1: RULE-BASED ENGINE ────────────────────────────────────────────────

def rule_based_feedback(
    wpm: float,
    pause_rate: float,
    filler_rate: float,
    filler_count: int,
    filler_matches: list,
    wpm_score: float,
    pause_score: float,
    filler_score: float,
    fluency_score: float,
    transcript: str,
    benchmark: str = "Call Center Entry",
) -> dict:
    """
    Generate structured, specific feedback for every feature.

    Returns dict:
        overall   — one-sentence summary
        wpm       — dict(status, icon, advice)
        pauses    — dict(status, icon, advice)
        fillers   — dict(status, icon, advice, top_words)
        priority  — str, the #1 thing to work on
        exercises — list of concrete practice drills
    """
    bm = BENCHMARKS[benchmark]

    # ── WPM ──
    wpm_lo, wpm_hi = bm["wpm"]
    if wpm < 90:
        wpm_status = "critical"
        wpm_icon   = "🔴"
        wpm_advice = (
            f"Your pace ({wpm:.0f} WPM) is well below fluent speech. "
            "Slow rate is often caused by word retrieval difficulty or anxiety. "
            "Practice speaking from a script at target pace first, then improvise."
        )
    elif wpm < wpm_lo:
        wpm_status = "below"
        wpm_icon   = "🟡"
        wpm_advice = (
            f"Your pace ({wpm:.0f} WPM) is below the {benchmark} target "
            f"({wpm_lo}–{wpm_hi} WPM). Try shadowing native speakers or reading "
            "a paragraph aloud while timing yourself."
        )
    elif wpm > 185:
        wpm_status = "too_fast"
        wpm_icon   = "🟡"
        wpm_advice = (
            f"You're speaking very fast ({wpm:.0f} WPM). Above 180 WPM, "
            "clarity suffers. Deliberately slow down on key phrases — "
            "a 1-second pause before important words improves comprehension."
        )
    elif wpm > wpm_hi:
        wpm_status = "slightly_fast"
        wpm_icon   = "🟢"
        wpm_advice = (
            f"Pace ({wpm:.0f} WPM) is above the ideal range. "
            "Slightly fast is better than slow, but practice intentional slowing "
            "to land key ideas."
        )
    else:
        wpm_status = "good"
        wpm_icon   = "✅"
        wpm_advice = (
            f"Excellent pace ({wpm:.0f} WPM) — right in the professional range "
            f"({wpm_lo}–{wpm_hi} WPM). Maintain this in stressful conversations."
        )

    # ── PAUSES ──
    bm_pause = bm["pause_pm"]
    if pause_rate > bm_pause * 2:
        pause_status = "critical"
        pause_icon   = "🔴"
        pause_advice = (
            f"High pause frequency ({pause_rate:.1f}/min vs. target ≤{bm_pause:.0f}/min). "
            "Long pauses signal word search difficulty. "
            "Practice 'bridging phrases' — 'What I mean is…', 'Let me put it this way…' — "
            "to keep speech flowing while you think."
        )
    elif pause_rate > bm_pause:
        pause_status = "below"
        pause_icon   = "🟡"
        pause_advice = (
            f"Pause rate ({pause_rate:.1f}/min) exceeds target. "
            "Focus on reducing pauses inside sentences — it's acceptable to pause "
            "between sentences, but mid-clause gaps disrupt comprehension most."
        )
    else:
        pause_status = "good"
        pause_icon   = "✅"
        pause_advice = (
            f"Pause frequency ({pause_rate:.1f}/min) is within range. "
            "Note that strategic pauses before key information are a pro technique — "
            "don't eliminate all pauses, only the involuntary ones."
        )

    # ── FILLERS ──
    bm_filler = bm["filler_pm"]

    # Count top filler words
    top_words: dict[str, int] = {}
    for m in filler_matches:
        w = m.group().lower()
        top_words[w] = top_words.get(w, 0) + 1
    top_sorted = sorted(top_words.items(), key=lambda x: -x[1])[:3]

    if filler_rate > bm_filler * 2:
        filler_status = "critical"
        filler_icon   = "🔴"
        filler_advice = (
            f"Filler rate is high ({filler_rate:.1f}/min vs. target ≤{bm_filler:.0f}/min). "
            "Fillers are often habitual. Record yourself daily and count them — "
            "awareness alone reduces rate by 30–50% within two weeks."
        )
    elif filler_rate > bm_filler:
        filler_status = "below"
        filler_icon   = "🟡"
        filler_advice = (
            f"Filler rate ({filler_rate:.1f}/min) is above target. "
            "Replace fillers with silence — a brief pause sounds more confident than 'um'."
        )
    else:
        filler_status = "good"
        filler_icon   = "✅"
        filler_advice = (
            f"Low filler rate ({filler_rate:.1f}/min) — under control. "
            "Most professionals average 2–4/min, so you're performing well."
        )

    # ── PRIORITY ──
    weakest = min(
        [("Speaking rate", wpm_score), ("Pause control", pause_score), ("Filler words", filler_score)],
        key=lambda x: x[1]
    )
    priority = f"**{weakest[0]}** (score: {weakest[1]:.0f}/100) — your highest-impact area to improve."

    # ── EXERCISES ──
    exercises = []
    if wpm < wpm_lo:
        exercises.append("📖 Read a newspaper paragraph aloud, aiming for 150 WPM. Record and compare.")
    if pause_rate > bm_pause:
        exercises.append("🎙️ Practise the 'topic → example → summary' structure for 30-second responses with no mid-sentence pauses.")
    if filler_rate > bm_filler:
        exercises.append(f"🚫 Record a 60-second response and delete every '{top_sorted[0][0] if top_sorted else 'filler'}' with silence. Observe the difference.")
    exercises.append("🔁 Re-record this exact prompt in 24 hours and compare scores.")
    exercises.append("🎯 Target one benchmark: score ≥ 68 for call center, ≥ 80 for professional standard.")

    # ── OVERALL ──
    if fluency_score >= 80:
        overall = f"Professional-level fluency ({fluency_score:.0f}/100) — you meet the standard."
    elif fluency_score >= 68:
        overall = f"Call-center ready ({fluency_score:.0f}/100) — proficient, with targeted room to grow."
    elif fluency_score >= 50:
        overall = f"Emerging fluency ({fluency_score:.0f}/100) — clear progress path with focused drills."
    else:
        overall = f"Developing stage ({fluency_score:.0f}/100) — consistent daily practice will create rapid gains."

    return {
        "overall": overall,
        "wpm":     {"status": wpm_status,    "icon": wpm_icon,    "advice": wpm_advice},
        "pauses":  {"status": pause_status,  "icon": pause_icon,  "advice": pause_advice},
        "fillers": {"status": filler_status, "icon": filler_icon, "advice": filler_advice,
                    "top_words": top_sorted},
        "priority":  priority,
        "exercises": exercises,
    }


# ── TIER 2: GROQ AI COACHING ─────────────────────────────────────────────────

def groq_coaching(
    transcript: str,
    wpm: float,
    pause_rate: float,
    filler_rate: float,
    fluency_score: float,
    filler_top: list[tuple],
    api_key: str,
) -> str:
    """
    Call Groq API (llama-3.3-70b-versatile) for personalized coaching.
    Free tier at console.groq.com — no credit card required.

    Returns a coaching paragraph as a string, or an error message.
    """
    from groq import Groq

    top_fillers_str = ", ".join(f"'{w}' ×{n}" for w, n in filler_top) if filler_top else "none detected"

    prompt = f"""You are an expert English fluency coach specialising in L2 speakers and professional communication.

Analyse this spoken English sample and provide specific, actionable coaching.

TRANSCRIPT:
\"\"\"{transcript}\"\"\"

ACOUSTIC METRICS:
- Fluency Score: {fluency_score}/100
- Speaking Rate: {wpm:.0f} WPM (professional target: 140–160)
- Pause Rate: {pause_rate:.1f} per minute (target ≤ 3.0)
- Filler Rate: {filler_rate:.1f} per minute (target ≤ 2.0)
- Top fillers: {top_fillers_str}

Write 3–4 sentences of personalised coaching. Be specific — reference actual words or phrases from the transcript. 
Identify the single most impactful change the speaker should make. End with one concrete 5-minute practice drill.
Do NOT repeat the metrics back. Write as a coach speaking directly to the student."""

    client = Groq(api_key=api_key)
    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.6,
    )
    return chat.choices[0].message.content.strip()
