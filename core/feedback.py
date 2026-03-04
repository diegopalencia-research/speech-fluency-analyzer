"""
core/feedback.py
────────────────
Two-tier feedback system.

TIER 1 — Rule-based engine (always free)
    Analyses WPM, pause rate, filler rate, discourse connectors, and
    lightweight grammar patterns. Returns structured, specific coaching.

TIER 2 — Groq AI coaching (optional, free API)
    Two modes:
      "narrative"    — paragraph-level coaching (original behaviour)
      "corrections"  — sentence-by-sentence error list, format:
                       "Almost. Say it like this: ___. Please repeat after me."
                       Mirrors the Finishing School correction protocol exactly.

Research basis:
  Skehan (1996)       — filler rate as planning difficulty
  Schmidt (1990)      — Noticing Hypothesis: explicit correction triggers noticing
  Long (1996)         — Interaction Hypothesis: corrective feedback in conversation
  Celce-Murcia (2007) — Discourse coherence as advanced fluency marker
"""

from __future__ import annotations
from core.score import BENCHMARKS


# ── TIER 1: RULE-BASED ENGINE ─────────────────────────────────────────────────

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
    discourse_data: dict | None = None,
    grammar_issues: list | None = None,
    task_prompt: str | None = None,
) -> dict:
    """
    Generate structured, actionable feedback for every measured feature.

    Returns dict:
        overall, wpm, pauses, fillers, discourse, grammar,
        priority, exercises
    """
    bm = BENCHMARKS[benchmark]

    # ── WPM feedback ──────────────────────────────────────────────────────
    wpm_lo, wpm_hi = bm["wpm"]
    if wpm < 90:
        wpm_icon, wpm_status = "○", "critical"
        wpm_advice = (
            f"Your pace ({wpm:.0f} WPM) is significantly below fluent speech. "
            "Slow rate often signals word retrieval difficulty. "
            "Practice reading a short paragraph aloud while timing yourself — "
            "aim for 130 WPM before recording."
        )
    elif wpm < wpm_lo:
        wpm_icon, wpm_status = "◑", "below"
        wpm_advice = (
            f"Pace ({wpm:.0f} WPM) is below the {benchmark} target ({wpm_lo}–{wpm_hi} WPM). "
            "Try shadowing native speakers or using a metronome during practice."
        )
    elif wpm > 185:
        wpm_icon, wpm_status = "◑", "too_fast"
        wpm_advice = (
            f"Speaking very fast ({wpm:.0f} WPM). Above 180 WPM, clarity suffers. "
            "Pause intentionally before key information — one beat of silence reads as confidence."
        )
    elif wpm > wpm_hi:
        wpm_icon, wpm_status = "●", "slightly_fast"
        wpm_advice = (
            f"Pace ({wpm:.0f} WPM) is slightly above the ideal range. "
            "Fast is better than slow, but practice landing key phrases with a pause."
        )
    else:
        wpm_icon, wpm_status = "●", "good"
        wpm_advice = (
            f"Excellent pace ({wpm:.0f} WPM) — right in the professional range. "
            "Maintain this under conversational pressure."
        )

    # ── Pause feedback ────────────────────────────────────────────────────
    bm_pause = bm["pause_pm"]
    if pause_rate > bm_pause * 2:
        pause_icon, pause_status = "○", "critical"
        pause_advice = (
            f"High pause frequency ({pause_rate:.1f}/min vs target ≤{bm_pause:.0f}/min). "
            "Mid-clause gaps are the most disruptive. Use bridging phrases: "
            "'What I mean is...', 'Let me put it this way...' to keep speech moving."
        )
    elif pause_rate > bm_pause:
        pause_icon, pause_status = "◑", "below"
        pause_advice = (
            f"Pause rate ({pause_rate:.1f}/min) exceeds target. "
            "Focus on eliminating pauses inside sentences — "
            "pauses between sentences are acceptable and even professional."
        )
    else:
        pause_icon, pause_status = "●", "good"
        pause_advice = (
            f"Pause frequency ({pause_rate:.1f}/min) is under control. "
            "Strategic pauses before key words are a pro technique — "
            "only eliminate involuntary mid-clause gaps."
        )

    # ── Filler feedback ───────────────────────────────────────────────────
    bm_filler = bm["filler_pm"]
    top_words: dict[str, int] = {}
    for m in filler_matches:
        w = m.group().lower()
        top_words[w] = top_words.get(w, 0) + 1
    top_sorted = sorted(top_words.items(), key=lambda x: -x[1])[:3]

    if filler_rate > bm_filler * 2:
        filler_icon, filler_status = "○", "critical"
        filler_advice = (
            f"Filler rate is high ({filler_rate:.1f}/min vs target ≤{bm_filler:.0f}/min). "
            "Fillers are habitual — awareness is the primary fix. "
            "Record yourself daily and count them. "
            "Awareness alone reduces rate by 30–50% within two weeks."
        )
    elif filler_rate > bm_filler:
        filler_icon, filler_status = "◑", "below"
        filler_advice = (
            f"Filler rate ({filler_rate:.1f}/min) is above target. "
            "Replace every filler with silence — a brief pause sounds far more "
            "confident than 'um'. Silence is not weakness."
        )
    else:
        filler_icon, filler_status = "●", "good"
        filler_advice = (
            f"Filler rate ({filler_rate:.1f}/min) is under control. "
            "Most professionals average 2–4/min. You're performing well."
        )

    # ── Discourse connector feedback ──────────────────────────────────────
    disc = discourse_data or {}
    disc_count      = disc.get("count", 0)
    disc_types      = disc.get("types_used", 0)
    disc_score      = disc.get("discourse_score", 60.0)
    disc_by_type    = disc.get("by_type", {})

    used_seq = bool(disc_by_type.get("sequencing"))
    missing_types = [k for k, v in disc_by_type.items() if not v]

    if disc_score < 40:
        disc_icon = "○"
        disc_advice = (
            f"No sequencing or cohesion markers detected. "
            "Discourse connectors (first, then, however, because, finally) "
            "signal planning ability and are a key marker of B2+ fluency. "
            "Try: 'First... Then... Finally...' in your next response."
        )
    elif disc_score < 65:
        disc_icon = "◑"
        disc_advice = (
            f"{disc_count} connectors used, {disc_types} type(s). "
            "You have some connectors but limited variety. "
            f"Add {'contrast words (however, although)' if 'contrast' in missing_types else 'cause-effect words (because, therefore)'}. "
            "Variety across connector types is a C1 marker."
        )
    else:
        disc_icon = "●"
        disc_advice = (
            f"{disc_count} connectors across {disc_types} types — good discourse structure. "
            "This signals active planning and is a strong fluency indicator. "
            f"{'Sequencing markers present — excellent.' if used_seq else 'Add explicit sequencing (first, then, finally) for maximum clarity.'}"
        )

    # ── Grammar feedback ──────────────────────────────────────────────────
    grammar = grammar_issues or []
    if not grammar:
        grammar_icon = "●"
        grammar_advice = (
            "No common grammar errors detected in this sample. "
            "Note: the rule-based checker targets the most frequent L2 patterns. "
            "For deeper grammar analysis, enable Groq AI in the sidebar."
        )
    else:
        grammar_icon = "◑" if len(grammar) <= 2 else "○"
        examples = "\n".join(
            f"• \"{g['text']}\" — {g['rule']}" for g in grammar[:3]
        )
        grammar_advice = (
            f"{len(grammar)} grammar pattern(s) flagged:\n{examples}\n"
            "Correct each one and repeat the phrase aloud immediately "
            "to activate noticing (Schmidt, 1990)."
        )

    # ── Task relevance ────────────────────────────────────────────────────
    task_feedback = None
    if task_prompt:
        word_count = len(transcript.split())
        if word_count < 30:
            task_feedback = (
                "Response is very short for the given task. "
                "Aim for at least 5–6 sentences per prompt to practice extended speech."
            )
        elif not used_seq and len(transcript) > 100:
            task_feedback = (
                f"Task: '{task_prompt[:60]}...' — response recorded. "
                "Consider using sequencing words to structure your answer: "
                "First... Then... Finally... This directly addresses the task."
            )

    # ── Priority ──────────────────────────────────────────────────────────
    all_scores = [
        ("Speaking Rate",       wpm_score),
        ("Pause Control",       pause_score),
        ("Filler Words",        filler_score),
        ("Discourse Structure", disc_score),
    ]
    weakest = min(all_scores, key=lambda x: x[1])
    priority = f"{weakest[0]} (score {weakest[1]:.0f}/100) is your highest-impact area."

    # ── Exercises ─────────────────────────────────────────────────────────
    exercises = []
    if wpm < wpm_lo:
        exercises.append("Read a paragraph aloud daily targeting 150 WPM. Record and compare.")
    if pause_rate > bm_pause:
        exercises.append(
            "Practice the topic → example → summary structure for 30-second responses "
            "with no mid-sentence pauses."
        )
    if filler_rate > bm_filler:
        tw = top_sorted[0][0] if top_sorted else "filler"
        exercises.append(
            f"Record a 60-second response. Every time you say '{tw}', "
            "replace it with silence. Listen back and notice the difference."
        )
    if disc_score < 65:
        exercises.append(
            "Retell your morning routine using: First... Then... After that... Finally... "
            "Record and check that all four markers appear."
        )
    if grammar:
        rule = grammar[0]["rule"]
        exercises.append(f"Grammar drill: {rule}")
    exercises.append("Re-record this exact prompt in 24 hours and compare scores.")

    # ── Overall summary ───────────────────────────────────────────────────
    if fluency_score >= 80:
        overall = f"Professional-level fluency ({fluency_score:.0f}/100) — you meet the standard."
    elif fluency_score >= 68:
        overall = f"Call-center ready ({fluency_score:.0f}/100) — proficient, with targeted room to grow."
    elif fluency_score >= 50:
        overall = f"Emerging fluency ({fluency_score:.0f}/100) — clear progress path with focused drills."
    else:
        overall = f"Developing stage ({fluency_score:.0f}/100) — consistent daily practice will produce rapid gains."

    return {
        "overall":   overall,
        "wpm":       {"status": wpm_status,    "icon": wpm_icon,    "advice": wpm_advice},
        "pauses":    {"status": pause_status,  "icon": pause_icon,  "advice": pause_advice},
        "fillers":   {"status": filler_status, "icon": filler_icon, "advice": filler_advice,
                      "top_words": top_sorted},
        "discourse": {"icon": disc_icon, "advice": disc_advice, "score": disc_score},
        "grammar":   {"icon": grammar_icon, "advice": grammar_advice, "issues": grammar},
        "task":      task_feedback,
        "priority":  priority,
        "exercises": exercises,
    }


# ── TIER 2a: GROQ NARRATIVE COACHING ─────────────────────────────────────────

def groq_coaching(
    transcript: str,
    wpm: float,
    pause_rate: float,
    filler_rate: float,
    fluency_score: float,
    filler_top: list[tuple],
    api_key: str,
    task_prompt: str | None = None,
    discourse_score: float = 60.0,
) -> str:
    """
    Groq API (llama-3.3-70b) — paragraph-level personalised coaching.
    References actual transcript phrases, task prompt, and discourse structure.
    """
    from groq import Groq

    top_fillers_str = ", ".join(f"'{w}' ×{n}" for w, n in filler_top) if filler_top else "none detected"
    task_context    = f"\nSpeaker was responding to this task: '{task_prompt}'" if task_prompt else ""

    prompt = f"""You are an expert English fluency coach specialising in L2 speakers and professional communication.

Analyse this spoken English sample and provide specific, actionable coaching.{task_context}

TRANSCRIPT:
\"\"\"{transcript}\"\"\"

ACOUSTIC AND LINGUISTIC METRICS:
- Fluency Score: {fluency_score}/100
- Speaking Rate: {wpm:.0f} WPM (professional target: 140–160)
- Pause Rate: {pause_rate:.1f} per minute (target ≤ 3.0)
- Filler Rate: {filler_rate:.1f} per minute (target ≤ 2.0)
- Top fillers: {top_fillers_str}
- Discourse Coherence Score: {discourse_score:.0f}/100

Write 3–4 sentences of personalised coaching. Be specific — reference actual words or phrases from the transcript.
Identify the single most impactful change the speaker should make.
End with one concrete 5-minute practice drill they can do right now.
Do NOT repeat the metrics back. Write directly to the student as their coach."""

    client = Groq(api_key=api_key)
    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=320,
        temperature=0.6,
    )
    return chat.choices[0].message.content.strip()


# ── TIER 2b: GROQ SENTENCE-LEVEL CORRECTIONS ─────────────────────────────────

def groq_sentence_corrections(
    transcript: str,
    api_key: str,
    task_prompt: str | None = None,
) -> list[dict]:
    """
    Groq API — Finishing School correction protocol.
    Returns a list of corrections in the format:
      {original, corrected, rule, repeat_after_me}

    Each correction mirrors the FS protocol:
      "Almost. Say it like this: ___. Please repeat after me."

    Only returns corrections where an actual error exists.
    Max 8 corrections to avoid overwhelming the learner.
    """
    from groq import Groq
    import json

    task_ctx = f"Task: '{task_prompt}'" if task_prompt else "Free speech sample"

    prompt = f"""You are an English fluency coach using the Finishing School correction protocol.

Analyse this transcript for grammar, vocabulary, and pronunciation-related errors.
{task_ctx}

TRANSCRIPT:
\"\"\"{transcript}\"\"\"

Return ONLY a JSON array of corrections. Each item must have:
  - "original": the exact phrase from the transcript with the error
  - "corrected": the corrected version
  - "rule": a short explanation (max 8 words, e.g. "third-person present needs -s")
  - "repeat_after_me": the full corrected sentence the student should repeat

Rules:
- Only include REAL errors, not style preferences
- Max 8 corrections
- If the transcript has NO errors, return []
- Do not include filler words (uh, um) as grammar errors
- Output ONLY the JSON array, no preamble, no markdown fences"""

    client = Groq(api_key=api_key)
    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.2,
    )
    raw = chat.choices[0].message.content.strip()
    # Strip any accidental markdown
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        corrections = json.loads(raw)
        return corrections if isinstance(corrections, list) else []
    except Exception:
        return []
