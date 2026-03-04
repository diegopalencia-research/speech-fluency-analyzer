"""
app.py — Speech Fluency Analyzer
══════════════════════════════════════════════════════════════════════════════
Project 03 · Diego Jose Palencia Robles · github.com/diegopalencia-research

FS Document integrations (Day 4 Grammar Protocol):
  • Optional task prompt field — contextualises analysis
  • Discourse connector detection and scoring
  • Lightweight grammar error detection
  • Sentence-level correction mode (Groq) — mirrors FS correction protocol
  • Dual transcript annotation: fillers (orange) + connectors (teal)

Core features: WPM · Pause rate · Filler rate · Articulation rate ·
               Pitch variation · Whisper confidence · Discourse coherence
Research: Lennon (1990) · Skehan (1996) · Tavakoli & Skehan (2005) ·
          Schmidt (1990) · Kormos & Denes (2004) · Hincks (2005)
══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import numpy as np
import json
import os
import tempfile
from datetime import datetime

st.set_page_config(
    page_title="Speech Fluency Analyzer",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

:root {
  --navy:   #0D1F35; --panel:  #0a1726; --card:   #162030;
  --blue:   #1E3A5F; --accent: #00D4AA; --a2: #FF6B35;
  --gold:   #FFD166; --light:  #F0F4F8; --gray: #8892A4;
  --border: rgba(0,212,170,0.14);
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background: var(--navy) !important;
  color: var(--light) !important;
}

[data-testid="stSidebar"] {
  background: var(--panel) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif !important; }

[data-testid="metric-container"] {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: .85rem 1rem;
}
[data-testid="stMetricValue"] {
  color: var(--accent) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 1.65rem !important;
}
[data-testid="stMetricLabel"] {
  color: var(--gray) !important; font-size: .7rem !important;
  letter-spacing: .05em; text-transform: uppercase;
}

h1 { font-family:'Space Mono',monospace !important; color:var(--accent) !important;
     font-size:1.7rem !important; letter-spacing:-.02em; }
h2 { font-family:'Space Mono',monospace !important; color:var(--light) !important;
     border-bottom:1px solid var(--border); padding-bottom:.35rem;
     font-size:.95rem !important; letter-spacing:.01em; }
h3 { font-family:'Space Mono',monospace !important; color:var(--accent) !important;
     font-size:.85rem !important; }

.score-badge {
  font-family:'Space Mono',monospace; font-size:3rem; font-weight:700;
  text-align:center; padding:1.2rem 1rem .85rem; border-radius:12px;
  border:2px solid; line-height:1.1;
}
.score-low  { color:#FF6B35; border-color:rgba(255,107,53,.28); background:rgba(255,107,53,.05); }
.score-mid  { color:#FFD166; border-color:rgba(255,209,102,.28);background:rgba(255,209,102,.05);}
.score-high { color:#00D4AA; border-color:rgba(0,212,170,.28);  background:rgba(0,212,170,.05); }
.score-pro  { color:#00f0c0; border-color:rgba(0,240,192,.38);  background:rgba(0,240,192,.07); }

.fb-wrap { background:rgba(0,212,170,.07); border-radius:4px; height:6px; width:100%; margin:4px 0 9px; }
.fb-fill      { border-radius:4px; height:6px; background:linear-gradient(90deg,#00D4AA,#00f0c0); }
.fb-fill-warn { border-radius:4px; height:6px; background:linear-gradient(90deg,#FFD166,#ffba00); }
.fb-fill-bad  { border-radius:4px; height:6px; background:linear-gradient(90deg,#FF6B35,#ff4f00); }

/* Filler = orange; Connector = teal */
.filler    { background:rgba(255,107,53,.18); color:#FF6B35; border-radius:3px;
             padding:1px 4px; font-weight:600; }
.connector { background:rgba(0,212,170,.15); color:#00D4AA; border-radius:3px;
             padding:1px 4px; font-weight:500; }

.transcript-box {
  background:var(--card); border:1px solid var(--border);
  border-radius:10px; padding:1.1rem 1.4rem; line-height:2.1; font-size:.9rem;
}

.surface { background:var(--card); border:1px solid var(--border);
           border-radius:10px; padding:1rem 1.25rem; margin-bottom:.55rem; }

.fb-card { background:var(--card); border:1px solid var(--border);
           border-radius:10px; padding:.95rem 1.2rem; margin-bottom:.55rem;
           font-size:.87rem; line-height:1.65; }
.fb-card-ai  { border-color:rgba(0,212,170,.3); background:rgba(0,212,170,.04); }
.fb-card-err { border-color:rgba(255,107,53,.3); background:rgba(255,107,53,.04); }

/* Sentence correction card */
.correction-card {
  background:var(--card); border:1px solid rgba(255,209,102,.25);
  border-radius:10px; padding:1rem 1.25rem; margin-bottom:.55rem;
  font-size:.87rem; line-height:1.7;
}
.correction-original { color:#FF6B35; text-decoration:line-through;
                        font-family:'Space Mono',monospace; font-size:.82rem; }
.correction-fixed    { color:#00D4AA; font-family:'Space Mono',monospace;
                        font-size:.82rem; font-weight:700; }
.repeat-prompt { color:#FFD166; font-size:.79rem; margin-top:.3rem; font-style:italic; }

.tr { display:flex; justify-content:space-between; align-items:center;
      padding:.38rem 0; border-bottom:1px solid rgba(0,212,170,.07); font-size:.83rem; }
.tr-pass { color:#00D4AA; font-family:'Space Mono',monospace; font-size:.77rem; }
.tr-fail { color:#FF6B35; font-family:'Space Mono',monospace; font-size:.77rem; }

.identity-bar { background:var(--card); border:1px solid var(--border);
                border-radius:8px; padding:.55rem .95rem;
                display:flex; align-items:center; gap:.65rem;
                font-family:'Space Mono',monospace; font-size:.77rem; margin-bottom:.4rem; }
.identity-dot { width:7px; height:7px; border-radius:50%;
                background:var(--accent); flex-shrink:0; }

.sb-label { font-size:.64rem; color:var(--accent); text-transform:uppercase;
            letter-spacing:.1em; font-family:'Space Mono',monospace;
            margin:0 0 7px; display:flex; align-items:center; gap:6px; }
.sb-label::after { content:''; flex:1; height:1px; background:var(--border); }

.key-card { background:var(--card); border:1px solid var(--border);
            border-radius:8px; padding:8px 11px; margin-bottom:6px; }
.key-card-head { font-size:.66rem; color:var(--gray); text-transform:uppercase;
                 letter-spacing:.06em; margin-bottom:3px; display:flex;
                 align-items:center; gap:5px; }
.badge      { font-size:.58rem; padding:1px 5px; border-radius:20px;
              text-transform:uppercase; letter-spacing:.04em; }
.badge-free { background:rgba(0,212,170,.14); color:var(--accent); }
.badge-opt  { background:rgba(136,146,164,.1); color:var(--gray); }

/* Task prompt box */
.task-box { background:rgba(0,212,170,.06); border:1px solid rgba(0,212,170,.2);
            border-radius:8px; padding:.7rem 1rem; margin-bottom:.6rem;
            font-size:.83rem; color:#F0F4F8; }
.task-label { font-family:'Space Mono',monospace; font-size:.68rem; color:var(--accent);
              text-transform:uppercase; letter-spacing:.08em; margin-bottom:.3rem; }

/* Connector legend */
.conn-legend { display:flex; gap:1rem; flex-wrap:wrap; margin:.4rem 0; font-size:.77rem; }
.conn-dot-f  { display:inline-block; width:8px; height:8px; border-radius:2px;
               background:rgba(255,107,53,.5); margin-right:4px; }
.conn-dot-c  { display:inline-block; width:8px; height:8px; border-radius:2px;
               background:rgba(0,212,170,.5); margin-right:4px; }

@keyframes pulse-ring  { 0%{transform:scale(.8);opacity:1} 100%{transform:scale(2.4);opacity:0} }
@keyframes pulse-dot   { 0%,100%{opacity:1} 50%{opacity:.35} }
@keyframes bar-bounce  { 0%,100%{height:5px} 50%{height:19px} }

.rec-wrap { display:flex; align-items:center; gap:13px;
            background:rgba(255,107,53,.07); border:1px solid rgba(255,107,53,.28);
            border-radius:10px; padding:12px 15px; margin:10px 0; }
.rec-dot-wrap { position:relative; width:13px; height:13px; flex-shrink:0; }
.rec-dot  { position:absolute; inset:0; background:#FF6B35; border-radius:50%;
            animation:pulse-dot 1.2s ease-in-out infinite; }
.rec-ring { position:absolute; inset:0; border:2px solid #FF6B35; border-radius:50%;
            animation:pulse-ring 1.2s ease-out infinite; }
.rec-bars { display:flex; align-items:center; gap:3px; height:22px; }
.rec-bar  { width:3px; background:#FF6B35; border-radius:2px;
            animation:bar-bounce .8s ease-in-out infinite; }
.rec-bar:nth-child(1){animation-delay:0s}  .rec-bar:nth-child(2){animation-delay:.13s}
.rec-bar:nth-child(3){animation-delay:.26s}.rec-bar:nth-child(4){animation-delay:.39s}
.rec-bar:nth-child(5){animation-delay:.13s}
.rec-title { font-family:'Space Mono',monospace; font-size:.77rem;
             color:#FF6B35; letter-spacing:.07em; }
.rec-sub { font-size:.73rem; color:var(--gray); margin-top:2px; }
.captured { background:rgba(0,212,170,.07); border:1px solid rgba(0,212,170,.25);
            border-radius:10px; padding:10px 15px; margin:10px 0;
            font-family:'Space Mono',monospace; font-size:.77rem; color:var(--accent);
            display:flex; align-items:center; gap:9px; }
.captured::before { content:'✓'; font-size:.95rem; }

.eyebrow { font-family:'Space Mono',monospace; font-size:.66rem; color:var(--accent);
           letter-spacing:.16em; text-transform:uppercase;
           display:flex; align-items:center; gap:.65rem; margin-bottom:.85rem; }
.eyebrow::before { content:''; width:28px; height:1px; background:var(--accent); }

.eval-card { background:var(--card); border:1px solid var(--border);
             border-radius:10px; padding:.9rem 1rem; height:100%; }
.eval-title { font-family:'Space Mono',monospace; font-size:.68rem; color:var(--accent);
              text-transform:uppercase; letter-spacing:.06em; margin-bottom:.4rem; }
.eval-body { font-size:.78rem; color:var(--gray); line-height:1.6; }

.stTabs [data-baseweb="tab"] { font-family:'Space Mono',monospace; font-size:.76rem; }
.stTabs [aria-selected="true"] { color:var(--accent) !important;
                                  border-bottom-color:var(--accent) !important; }
[data-testid="stFileUploader"] { border:2px dashed var(--border) !important;
                                  border-radius:10px !important;
                                  background:var(--card) !important; }
.stButton > button { background:var(--card) !important;
                     border:1px solid var(--border) !important;
                     color:var(--light) !important;
                     font-family:'Space Mono',monospace !important;
                     font-size:.74rem !important; border-radius:6px !important;
                     transition:all .17s !important; }
.stButton > button:hover { border-color:var(--accent) !important;
                           color:var(--accent) !important; }
.stProgress > div > div > div > div { background:var(--accent) !important; }
.stAlert { border-radius:8px !important; font-size:.84rem !important; }
code { font-family:'Space Mono',monospace !important; color:var(--accent) !important;
       background:rgba(0,212,170,.07) !important; padding:.1em .33em; border-radius:3px; }
.footer { text-align:center; padding:1.8rem 0 .8rem; color:var(--gray);
          font-size:.7rem; font-family:'Space Mono',monospace;
          border-top:1px solid var(--border); margin-top:2.5rem;
          letter-spacing:.04em; line-height:1.9; }
@media (max-width:768px) {
  .stTabs [data-baseweb="tab"] { font-size:.67rem !important; padding:5px 6px !important; }
  [data-testid="stMetricValue"]  { font-size:1.25rem !important; }
  .score-badge { font-size:2.5rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ── IMPORTS ───────────────────────────────────────────────────────────────────
# Core functions — always present in every version of score.py
from core.score import (
    detect_fillers, annotate_transcript, compute_scores,
    BENCHMARKS, score_label, score_css, score_color,
)

# Extended functions added in v4.
# Stubs below keep the app alive if score.py on GitHub is still the old version.
try:
    from core.score import (
        annotate_transcript_full,
        detect_discourse_connectors,
        score_discourse_coherence,
        detect_grammar_issues,
        compute_articulation_score,
    )
except ImportError:
    import numpy as _np

    def annotate_transcript_full(transcript, filler_matches, connector_matches):
        return annotate_transcript(transcript, filler_matches)

    def detect_discourse_connectors(transcript):
        return {"matches": [], "by_type": {}, "count": 0, "types_used": 0}

    def score_discourse_coherence(disc, duration_s, word_count):
        return 60.0

    def detect_grammar_issues(transcript):
        return []

    def compute_articulation_score(art_rate):
        return float(_np.clip((art_rate - 80) / 140 * 100, 0, 100))

from core.storage import get_history, add_session, clear_history, export_json, import_json


# ── HELPERS ───────────────────────────────────────────────────────────────────
def bar_html(score: float) -> str:
    cls = "fb-fill" if score >= 68 else ("fb-fill-warn" if score >= 50 else "fb-fill-bad")
    return (f'<div class="fb-wrap">'
            f'<div class="{cls}" style="width:{int(score)}%"></div></div>')


# ── DEMO ──────────────────────────────────────────────────────────────────────
def make_demo() -> dict:
    wpm, dur_s = 118.4, 48.0
    dur_m      = dur_s / 60
    pause_count, filler_count = 6, 7
    pause_rate  = round(pause_count / dur_m, 2)
    filler_rate = round(filler_count / dur_m, 2)
    art_rate    = round(wpm * dur_m / max(dur_s - 14.0, 1) * 60, 1)
    scores      = compute_scores(wpm, pause_rate, filler_rate)
    transcript  = (
        "So, uh, I wanted to talk about the project timeline, you know, "
        "because basically we have a few things to consider. Um, the first "
        "thing is the deadline, which is, like, coming up pretty soon. "
        "However, I think we need to, right, prioritise the core features. "
        "Finally, the main deliverable is the dashboard and the scoring module."
    )
    fm   = detect_fillers(transcript)
    disc = detect_discourse_connectors(transcript)
    disc["discourse_score"] = score_discourse_coherence(disc, dur_s, len(transcript.split()))
    gi   = detect_grammar_issues(transcript)
    return {
        "fluency_score":    scores["fluency"],
        "wpm": wpm,         "wpm_score": scores["wpm_score"],
        "pause_count":      pause_count, "pause_rate": pause_rate,
        "pause_score":      scores["pause_score"],
        "filler_count":     filler_count, "filler_rate": filler_rate,
        "filler_score":     scores["filler_score"],
        "articulation_rate":art_rate,
        "art_score":        compute_articulation_score(art_rate),
        "pitch_std":        38.4, "pitch_score": 74.0, "confidence": 68.5,
        "duration_s":       dur_s, "transcript": transcript,
        "filler_matches":   fm,
        "discourse":        disc,
        "grammar_issues":   gi,
        "demo": True,
        "pauses": [(3.1,3.6,.5),(9.4,10.1,.7),(17.2,17.9,.7),
                   (24.1,24.7,.6),(31.0,31.7,.7),(38.0,38.5,.5)],
        "y": None, "sr": 16000,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='font-family:Space Mono,monospace;font-size:.9rem;"
        "color:#00D4AA;margin-bottom:2px'>Speech Fluency Analyzer</div>"
        "<div style='font-size:.69rem;color:#8892A4;font-family:Space Mono,monospace;"
        "margin-bottom:10px'>Project 03 · Palencia Research</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown('<div class="sb-label">Identity</div>', unsafe_allow_html=True)
    username = st.text_input("uid", value=st.session_state.get("username", ""),
                              placeholder="Name or username",
                              label_visibility="collapsed").strip() or "default"
    st.session_state["username"] = username
    st.markdown(
        f"<div style='font-size:.69rem;color:#8892A4;margin-top:1px;margin-bottom:7px'>"
        f"Saved as <code>{username}</code></div>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown('<div class="sb-label">API Keys</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="key-card">
      <div class="key-card-head">OpenAI Whisper
        <span class="badge badge-opt">Optional</span>
      </div>
      <div style='font-size:.69rem;color:#8892A4;margin-bottom:4px'>
        Faster transcription (~5 s vs ~30 s local)
      </div>
    </div>""", unsafe_allow_html=True)
    openai_key = st.text_input("ok", type="password", placeholder="sk-…",
                                label_visibility="collapsed")

    st.markdown("""
    <div class="key-card">
      <div class="key-card-head">Groq / Llama-3.3
        <span class="badge badge-free">Free</span>
      </div>
      <div style='font-size:.69rem;color:#8892A4;margin-bottom:4px'>
        AI coaching + sentence corrections · console.groq.com
      </div>
    </div>""", unsafe_allow_html=True)
    groq_key = st.text_input("gk", type="password", placeholder="gsk_…",
                              label_visibility="collapsed")
    st.markdown(
        "<div style='font-size:.67rem;color:rgba(136,146,164,.5);margin-top:2px'>"
        "The app is fully free without keys.</div>",
        unsafe_allow_html=True,
    )
    openai_key = openai_key or st.secrets.get("OPENAI_API_KEY", "")
    groq_key   = groq_key   or st.secrets.get("GROQ_API_KEY",   "")

    st.divider()
    st.markdown('<div class="sb-label">Settings</div>', unsafe_allow_html=True)
    benchmark  = st.selectbox("Benchmark", list(BENCHMARKS.keys()), index=0)
    silence_db = st.slider("Silence threshold (dB)", 15, 45, 30)
    min_pause  = st.slider("Min pause (ms)", 200, 800, 400, step=50)
    model_size = st.selectbox("Local Whisper model", ["tiny","base","small"], index=0)

    # Groq feedback mode toggle
    correction_mode = st.toggle(
        "Correction mode (Groq)",
        value=False,
        help="When ON: Groq returns sentence-by-sentence corrections in Finishing School format. "
             "When OFF: paragraph coaching narrative.",
    )

    st.divider()
    st.markdown("""
    <div style='font-size:.7rem;color:#8892A4;line-height:1.85'>
    <span style='color:#00D4AA;font-weight:600'>Formula</span><br>
    40% WPM · 35% Pause · 25% Filler<br><br>
    <span style='color:#00D4AA;font-weight:600'>Targets</span><br>
    Call center: ≥ 68 · Professional: ≥ 80<br><br>
    <span style='color:#00D4AA;font-weight:600'>Research</span><br>
    Lennon (1990) · Skehan (1996)<br>
    Tavakoli & Skehan (2005)<br>
    Schmidt (1990) · Hincks (2005)
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="sb-label">Session History</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Export", data=export_json(username),
                            file_name=f"fluency_{username}.json",
                            mime="application/json", use_container_width=True)
    with c2:
        if st.button("Clear", use_container_width=True):
            clear_history(username); st.rerun()
    with st.expander("Import previous history"):
        st.markdown("<div style='font-size:.7rem;color:#8892A4;margin-bottom:5px'>"
                    "Upload a previously exported history file.</div>",
                    unsafe_allow_html=True)
        hist_file = st.file_uploader("hf", type="json", label_visibility="collapsed")
        if hist_file:
            ok, msg = import_json(hist_file.read().decode(), username)
            (st.success if ok else st.error)(msg)
            if ok: st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="eyebrow">Project 03  ·  Speech Science  ·  L2 Assessment</div>',
            unsafe_allow_html=True)
st.title("Speech Fluency Analyzer")

if username != "default":
    st.markdown(
        f'<div class="identity-bar"><div class="identity-dot"></div>'
        f'<span style="color:#8892A4">Logged in as</span>'
        f'<span style="color:#F0F4F8">{username}</span>'
        f'<span style="color:#8892A4;margin-left:auto;font-size:.7rem">'
        f'{len(get_history(username))} session(s)</span></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    "<p style='color:#8892A4;font-size:.9rem;max-width:720px;"
    "margin-top:-.3rem;line-height:1.75'>"
    "Upload or record a 30–60 second English speech sample. The analyzer extracts "
    "six acoustic and linguistic features "
    "(speaking rate, articulation rate, pause frequency, pitch variation, "
    "filler word rate, and transcription confidence) "
    "plus discourse connector analysis and grammar error detection. "
    "Returns a composite <b style='color:#F0F4F8'>Fluency Score 0–100</b>, "
    "annotated transcript, waveform, structured coaching, "
    "sentence-level corrections, and a branded PDF report. "
    "<b style='color:#00D4AA'>No API key required.</b>"
    "</p>",
    unsafe_allow_html=True,
)

ec1, ec2, ec3 = st.columns(3)
for col, title, body in [
    (ec1, "Acoustic Features",
     "Speaking rate (WPM), articulation rate (speech-only WPM), pause frequency, "
     "and pitch variation (F0 std dev) — extracted with librosa."),
    (ec2, "Linguistic Features",
     "Filler word detection, discourse connector analysis (sequencing, contrast, "
     "cause-effect), lightweight grammar error detection."),
    (ec3, "Task-Aware Feedback",
     "Optional task prompt field contextualises the analysis. "
     "Groq AI coaching in two modes: narrative or sentence-level corrections "
     "mirroring the Finishing School correction protocol."),
]:
    with col:
        col.markdown(
            f'<div class="eval-card"><div class="eval-title">{title}</div>'
            f'<div class="eval-body">{body}</div></div>',
            unsafe_allow_html=True,
        )

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TASK PROMPT (optional)
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("Set a task prompt (optional)", expanded=False):
    st.markdown(
        "<div style='font-size:.8rem;color:#8892A4;margin-bottom:.5rem'>"
        "Describe what you asked the speaker to do. This gives the AI more context "
        "to evaluate relevance, task completion, and discourse structure.<br>"
        "<b style='color:#F0F4F8'>Examples:</b> 'Describe your morning routine' · "
        "'Retell the story using first, then, finally' · "
        "'Answer: Why do people follow routines?'</div>",
        unsafe_allow_html=True,
    )
    task_prompt = st.text_input(
        "task_prompt_input",
        placeholder="e.g. Describe your morning routine using sequencing words",
        label_visibility="collapsed",
    ).strip() or None
    if task_prompt:
        st.markdown(
            f'<div class="task-box"><div class="task-label">Active task</div>{task_prompt}</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# INPUT TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_mic, tab_upload, tab_demo = st.tabs([
    "  Record (microphone)  ",
    "  Upload file  ",
    "  Demo  ",
])

audio_bytes: bytes | None = None
audio_suffix = ".wav"

with tab_mic:
    st.markdown(
        "<p style='color:#8892A4;font-size:.83rem'>"
        "Click the microphone button to start. Speak 30–60 seconds, "
        "then click again to stop. Works best on Chrome or Edge.</p>",
        unsafe_allow_html=True,
    )
    try:
        from audio_recorder_streamlit import audio_recorder
        recorded = audio_recorder(text="", recording_color="#FF6B35",
                                   neutral_color="#8892A4", icon_size="3x",
                                   pause_threshold=3.0, sample_rate=16000)
        if not recorded:
            st.markdown("""
            <div class="rec-wrap">
              <div class="rec-dot-wrap"><div class="rec-ring"></div><div class="rec-dot"></div></div>
              <div class="rec-bars">
                <div class="rec-bar"></div><div class="rec-bar"></div>
                <div class="rec-bar"></div><div class="rec-bar"></div><div class="rec-bar"></div>
              </div>
              <div>
                <div class="rec-title">CLICK MIC TO START  ·  CLICK AGAIN TO STOP</div>
                <div class="rec-sub">Recommended: 30–60 seconds of natural English speech</div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            audio_bytes, audio_suffix = recorded, ".wav"
            st.markdown(
                '<div class="captured">Recording captured — click Analyse Speech below</div>',
                unsafe_allow_html=True,
            )
            st.audio(recorded, format="audio/wav")
    except ImportError:
        st.warning("Microphone requires `audio-recorder-streamlit`. "
                   "Run `pip install audio-recorder-streamlit` then restart.")

with tab_upload:
    st.markdown(
        "<p style='color:#8892A4;font-size:.83rem'>"
        "WAV · MP3 · M4A · OGG  ·  Recommended: 30–60 seconds  ·  "
        "Accented English fully supported.</p>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("aud", type=["wav","mp3","m4a","ogg"],
                                 label_visibility="collapsed")
    if uploaded:
        audio_bytes  = uploaded.read()
        audio_suffix = "." + uploaded.name.rsplit(".", 1)[-1].lower()
        st.audio(audio_bytes, format=f"audio/{audio_suffix.lstrip('.')}")

with tab_demo:
    st.markdown(
        "<p style='color:#8892A4;font-size:.83rem'>"
        "Explore the full dashboard with a pre-loaded 48-second synthetic sample.</p>",
        unsafe_allow_html=True,
    )
    run_demo = st.button("Run demo analysis")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
result: dict | None = None

if run_demo:
    result = make_demo()

elif audio_bytes:
    c_btn, c_info = st.columns([1, 3])
    with c_btn:
        run_analysis = st.button("Analyse Speech", type="primary", use_container_width=True)
    with c_info:
        backend = "OpenAI Whisper API" if openai_key else f"Local Whisper ({model_size})"
        ai_tier = ("Groq corrections mode" if (groq_key and correction_mode)
                   else "Groq coaching" if groq_key else "Rule-based coaching")
        st.markdown(
            f"<span style='font-size:.76rem;color:#8892A4'>"
            f"Transcription: <b style='color:#00D4AA'>{backend}</b>  ·  "
            f"Feedback: <b style='color:#00D4AA'>{ai_tier}</b></span>",
            unsafe_allow_html=True,
        )
    if not run_analysis:
        st.stop()

    with st.status("Analysing speech…", expanded=True) as status:
        try:
            import librosa
            from core.analyze    import (load_audio, detect_pauses, render_waveform,
                                         extract_pitch_variation, extract_whisper_confidence)
            from core.transcribe import transcribe

            st.write("Loading audio…")
            y, sr, duration_s, tmp_path = load_audio(audio_bytes, suffix=audio_suffix)
            duration_m = duration_s / 60

            st.write("Detecting pauses and pitch…")
            pause_count, total_pause_s, pauses, speech_time_s, _ = detect_pauses(
                y, sr, silence_db=silence_db, min_pause_s=min_pause / 1000)
            pause_rate = round(pause_count / max(duration_m, .01), 2)
            pitch_std, pitch_score = extract_pitch_variation(y, sr)

            st.write("Transcribing…")
            if openai_key:
                from core.transcribe import transcribe_api
                transcript_raw = transcribe_api(tmp_path, openai_key)
                whisper_result = None
            else:
                import whisper as _whisper
                from core.transcribe import _model_cache
                if model_size not in _model_cache:
                    _model_cache[model_size] = _whisper.load_model(model_size)
                whisper_result = _model_cache[model_size].transcribe(
                    tmp_path, language="en", fp16=False)
                transcript_raw = whisper_result["text"].strip()
            os.unlink(tmp_path)

            st.write("Computing all features…")
            word_count     = len(transcript_raw.split())
            wpm            = round(word_count / max(duration_m, .01), 1)
            art_rate       = round(word_count / max(speech_time_s / 60, .01), 1)
            filler_matches = detect_fillers(transcript_raw)
            filler_count   = len(filler_matches)
            filler_rate    = round(filler_count / max(duration_m, .01), 2)
            confidence     = extract_whisper_confidence(whisper_result)
            art_score      = compute_articulation_score(art_rate)

            disc = detect_discourse_connectors(transcript_raw)
            disc["discourse_score"] = score_discourse_coherence(disc, duration_s, word_count)
            grammar_issues = detect_grammar_issues(transcript_raw)

            scores = compute_scores(wpm, pause_rate, filler_rate)

            result = {
                "fluency_score":    scores["fluency"],
                "wpm": wpm,         "wpm_score": scores["wpm_score"],
                "pause_count":      pause_count, "pause_rate": pause_rate,
                "pause_score":      scores["pause_score"],
                "filler_count":     filler_count, "filler_rate": filler_rate,
                "filler_score":     scores["filler_score"],
                "articulation_rate":art_rate, "art_score": art_score,
                "pitch_std":        pitch_std,  "pitch_score": pitch_score,
                "confidence":       confidence, "duration_s":  duration_s,
                "transcript":       transcript_raw,
                "filler_matches":   filler_matches,
                "discourse":        disc,
                "grammar_issues":   grammar_issues,
                "demo": False, "pauses": pauses, "y": y, "sr": sr,
            }
            add_session(result, benchmark, username)
            status.update(label="Analysis complete", state="complete")

        except ImportError as e:
            status.update(label="Missing package", state="error")
            st.error(f"Missing: **{e}** — run `pip install -r requirements.txt`")
            st.stop()
        except Exception as e:
            status.update(label="Analysis failed", state="error")
            st.error(str(e)); st.stop()

else:
    st.markdown("""
    <div style='background:#162030;border:1px solid rgba(0,212,170,.1);
                border-radius:12px;padding:2.5rem 2rem;text-align:center;margin:1rem 0'>
      <div style='font-family:Space Mono,monospace;font-size:.95rem;
                  color:#F0F4F8;margin-bottom:.55rem'>Three ways to begin</div>
      <div style='color:#8892A4;font-size:.84rem;max-width:420px;margin:0 auto;line-height:1.85'>
        <b style='color:#00D4AA'>Record</b>  —  live via browser microphone<br>
        <b style='color:#00D4AA'>Upload</b>  —  WAV, MP3, M4A, or OGG file<br>
        <b style='color:#00D4AA'>Demo</b>  —  explore the dashboard instantly
      </div>
      <div style='color:rgba(136,146,164,.4);font-size:.72rem;margin-top:1.3rem'>
        No API key required  ·  Fully free by default
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
r  = result
sc = r["fluency_score"]
bm = BENCHMARKS[benchmark]
disc_data    = r.get("discourse", {})
grammar_data = r.get("grammar_issues", [])


# ── 1. Task prompt display ────────────────────────────────────────────────────
if task_prompt:
    st.markdown(
        f'<div class="task-box"><div class="task-label">Task</div>{task_prompt}</div>',
        unsafe_allow_html=True,
    )


# ── 2. Score overview ─────────────────────────────────────────────────────────
st.markdown("## Score Overview")
col_s, col_m = st.columns([1, 2], gap="large")

with col_s:
    st.markdown(
        f'<div class="score-badge {score_css(sc)}">{sc}<br>'
        f'<span style="font-size:.82rem;letter-spacing:.07em;opacity:.8">'
        f'{score_label(sc)}</span></div>',
        unsafe_allow_html=True,
    )
    delta   = round(sc - bm["min_score"], 1)
    d_color = "#00D4AA" if delta >= 0 else "#FF6B35"
    d_sign  = "+" if delta >= 0 else ""
    st.markdown(
        f"<div style='text-align:center;font-family:Space Mono,monospace;"
        f"font-size:.76rem;color:{d_color};margin-top:.25rem'>"
        f"{d_sign}{delta} vs {benchmark}</div>",
        unsafe_allow_html=True,
    )

with col_m:
    r1, r2, r3 = st.columns(3)
    r1.metric("Words / Min",   f"{r['wpm']:.0f}")
    r2.metric("Pauses / Min",  f"{r['pause_rate']:.1f}")
    r3.metric("Fillers / Min", f"{r['filler_rate']:.1f}")
    dur_m2 = int(r["duration_s"] // 60)
    dur_s2 = int(r["duration_s"] % 60)
    r4, r5, r6 = st.columns(3)
    r4.metric("Duration",    f"{dur_m2}m {dur_s2}s")
    r5.metric("Pauses",      str(r["pause_count"]))
    r6.metric("Fillers",     str(r["filler_count"]))

st.divider()


# ── 3. Core decomposition ─────────────────────────────────────────────────────
st.markdown("## Score Decomposition")
dc1, dc2, dc3 = st.columns(3)

def decomp_block(col, title, raw, weight, note):
    pct   = int(raw)
    bar_c = "fb-fill" if raw >= 68 else ("fb-fill-warn" if raw >= 50 else "fb-fill-bad")
    col.markdown(
        f"<div style='font-size:.68rem;color:#8892A4;text-transform:uppercase;"
        f"letter-spacing:.06em'>{title}</div>"
        f"<div style='font-family:Space Mono,monospace;font-size:1.5rem;color:#00D4AA'>"
        f"{raw:.0f}<span style='font-size:.78rem;color:#8892A4'>/100</span>"
        f"<span style='font-size:.68rem;color:#8892A4;margin-left:.45rem'>× {weight}</span></div>"
        f'<div class="fb-wrap"><div class="{bar_c}" style="width:{pct}%"></div></div>'
        f"<div style='font-size:.72rem;color:#8892A4'>{note}</div>",
        unsafe_allow_html=True,
    )

decomp_block(dc1, "Speaking Rate",  r["wpm_score"],    "0.40", f"{r['wpm']:.0f} WPM  ·  target 140–160")
decomp_block(dc2, "Pause Control",  r["pause_score"],  "0.35", f"{r['pause_count']} pauses detected")
decomp_block(dc3, "Filler Words",   r["filler_score"], "0.25", f"{r['filler_count']} fillers found")

st.divider()


# ── 4. Extended features ──────────────────────────────────────────────────────
st.markdown("## Extended Acoustic Analysis")
ec1, ec2, ec3 = st.columns(3)

def ext_block(col, title, val_str, raw, note):
    pct   = int(raw)
    bar_c = "fb-fill" if raw >= 68 else ("fb-fill-warn" if raw >= 50 else "fb-fill-bad")
    col.markdown(
        f"<div style='font-size:.68rem;color:#8892A4;text-transform:uppercase;"
        f"letter-spacing:.06em'>{title}</div>"
        f"<div style='font-family:Space Mono,monospace;font-size:1.15rem;color:#00D4AA'>"
        f"{val_str}</div>"
        f"<div style='font-size:.7rem;color:#8892A4;margin-bottom:2px'>Score: {raw:.0f}/100</div>"
        f'<div class="fb-wrap"><div class="{bar_c}" style="width:{pct}%"></div></div>'
        f"<div style='font-size:.7rem;color:#8892A4'>{note}</div>",
        unsafe_allow_html=True,
    )

ext_block(ec1, "Articulation Rate", f"{r.get('articulation_rate',0):.0f} WPM",
          r.get("art_score",50), "Speech-only WPM  ·  Kormos & Denes (2004)")
ext_block(ec2, "Pitch Variation",  f"F0 std {r.get('pitch_std',0):.0f} Hz",
          r.get("pitch_score",50), "F0 standard deviation  ·  Hincks (2005)")
ext_block(ec3, "Transcription Confidence", f"{r.get('confidence',50):.0f}/100",
          r.get("confidence",50), "Whisper avg_logprob — clarity proxy")

st.divider()


# ── 5. Discourse connector analysis ──────────────────────────────────────────
st.markdown("## Discourse Connector Analysis")
disc_score  = disc_data.get("discourse_score", 60.0)
disc_count  = disc_data.get("count", 0)
disc_types  = disc_data.get("types_used", 0)
disc_by_type = disc_data.get("by_type", {})

d_col1, d_col2 = st.columns([1, 2])
with d_col1:
    pct   = int(disc_score)
    bar_c = "fb-fill" if disc_score >= 65 else ("fb-fill-warn" if disc_score >= 45 else "fb-fill-bad")
    st.markdown(
        f"<div style='font-family:Space Mono,monospace;font-size:1.5rem;color:#00D4AA'>"
        f"{disc_score:.0f}<span style='font-size:.78rem;color:#8892A4'>/100</span></div>"
        f"<div style='font-size:.7rem;color:#8892A4;margin-bottom:2px'>"
        f"{disc_count} connectors · {disc_types} type(s)</div>"
        f'<div class="fb-wrap"><div class="{bar_c}" style="width:{pct}%"></div></div>'
        f"<div style='font-size:.7rem;color:#8892A4'>Discourse Coherence  ·  Schmidt (1990)</div>",
        unsafe_allow_html=True,
    )

with d_col2:
    # Type breakdown
    type_html = ""
    for ctype, words in disc_by_type.items():
        if words:
            examples = ", ".join(set(words[:3]))
            type_html += (
                f"<div style='margin-bottom:.25rem;font-size:.78rem'>"
                f"<span style='color:#00D4AA;font-family:Space Mono,monospace;"
                f"font-size:.68rem;text-transform:uppercase'>{ctype}</span>"
                f"  <span style='color:#8892A4'>{examples}</span></div>"
            )
    if not type_html:
        type_html = "<div style='font-size:.78rem;color:#8892A4'>No connectors detected.</div>"
    st.markdown(type_html, unsafe_allow_html=True)

st.divider()


# ── 6. Waveform ───────────────────────────────────────────────────────────────
st.markdown("## Waveform  ·  Pause Annotation")
if r.get("y") is not None:
    try:
        from core.analyze import render_waveform
        st.pyplot(render_waveform(r["y"], r["sr"], r["pauses"]),
                  use_container_width=True)
    except Exception:
        st.info("Waveform rendering unavailable.")
else:
    st.markdown(
        '<div style="background:#162030;border:1px solid rgba(0,212,170,.1);'
        'border-radius:8px;padding:1rem;text-align:center;color:#8892A4;font-size:.8rem">'
        'Demo mode — upload or record audio to see the annotated waveform.</div>',
        unsafe_allow_html=True,
    )

st.divider()


# ── 7. Annotated transcript ───────────────────────────────────────────────────
st.markdown("## Annotated Transcript")
filler_matches = r.get("filler_matches") or detect_fillers(r["transcript"])
conn_matches   = disc_data.get("matches", [])
annotated      = annotate_transcript_full(r["transcript"], filler_matches, conn_matches)

st.markdown(f'<div class="transcript-box">{annotated}</div>', unsafe_allow_html=True)
st.markdown("""
<div class="conn-legend">
  <span><span class="conn-dot-f"></span>Filler word</span>
  <span><span class="conn-dot-c"></span>Discourse connector</span>
</div>
""", unsafe_allow_html=True)

st.divider()


# ── 8. Coaching feedback ──────────────────────────────────────────────────────
st.markdown("## Coaching Feedback")

from core.feedback import rule_based_feedback, groq_coaching, groq_sentence_corrections

fw: dict[str, int] = {}
for m in filler_matches:
    w = m.group().lower()
    fw[w] = fw.get(w, 0) + 1
top_sorted = sorted(fw.items(), key=lambda x: -x[1])[:3]

fb = rule_based_feedback(
    wpm=r["wpm"], pause_rate=r["pause_rate"], filler_rate=r["filler_rate"],
    filler_count=r["filler_count"], filler_matches=filler_matches,
    wpm_score=r["wpm_score"], pause_score=r["pause_score"],
    filler_score=r["filler_score"], fluency_score=sc,
    transcript=r["transcript"], benchmark=benchmark,
    discourse_data={**disc_data, "discourse_score": disc_score},
    grammar_issues=grammar_data,
    task_prompt=task_prompt,
)

# Overall banner
banner_c = "#00D4AA" if sc >= 68 else ("#FFD166" if sc >= 50 else "#FF6B35")
st.markdown(
    f'<div class="surface" style="border-color:rgba(0,212,170,.22);color:{banner_c}">'
    f'{fb["overall"]}</div>',
    unsafe_allow_html=True,
)

# Task feedback if present
if fb.get("task"):
    st.markdown(
        f'<div class="fb-card" style="border-color:rgba(255,209,102,.25)">'
        f'<div style="font-size:.67rem;text-transform:uppercase;letter-spacing:.06em;'
        f'color:#8892A4;margin-bottom:.3rem">Task Feedback</div>'
        f'{fb["task"]}</div>',
        unsafe_allow_html=True,
    )

# Five feature cards: WPM, Pauses, Fillers, Discourse, Grammar
row1 = st.columns(3)
row2 = st.columns(2)

cards = [
    (row1[0], "wpm",      "Speaking Rate"),
    (row1[1], "pauses",   "Pause Control"),
    (row1[2], "fillers",  "Filler Words"),
    (row2[0], "discourse","Discourse Structure"),
    (row2[1], "grammar",  "Grammar"),
]
for col, key, title in cards:
    d = fb[key]
    col.markdown(
        f'<div class="fb-card">'
        f'<div style="font-size:.67rem;text-transform:uppercase;letter-spacing:.06em;'
        f'color:#8892A4;margin-bottom:.3rem">{d["icon"]}  {title}</div>'
        f'{d["advice"]}</div>',
        unsafe_allow_html=True,
    )

if top_sorted:
    ts_html = "  ·  ".join(
        f'<span class="filler">{w}</span> ×{n}' for w, n in top_sorted
    )
    st.markdown(
        f'<div style="font-size:.77rem;color:#8892A4;margin-top:-.2rem;margin-bottom:.5rem">'
        f'Most frequent fillers: {ts_html}</div>',
        unsafe_allow_html=True,
    )

# Priority
st.markdown(
    f'<div class="fb-card" style="border-color:rgba(255,209,102,.22)">'
    f'<div style="font-size:.67rem;text-transform:uppercase;letter-spacing:.06em;'
    f'color:#8892A4;margin-bottom:.3rem">Priority Focus</div>'
    f'{fb["priority"]}</div>',
    unsafe_allow_html=True,
)

# Practice drills
with st.expander("Practice Drills", expanded=False):
    for ex in fb["exercises"]:
        st.markdown(
            f'<div style="padding:.28rem 0;font-size:.84rem;color:#F0F4F8">{ex}</div>',
            unsafe_allow_html=True,
        )

# ── Groq AI feedback ──────────────────────────────────────────────────────────
if groq_key and not r.get("demo"):
    st.markdown("---")
    if correction_mode:
        st.markdown("### Sentence-Level Corrections  ·  Finishing School Protocol")
        st.markdown(
            "<div style='font-size:.78rem;color:#8892A4;margin-bottom:.7rem'>"
            "Each correction follows the protocol: identify the error → provide the correct form → "
            "prompt repetition. (Long, 1996 — corrective feedback in interaction.)</div>",
            unsafe_allow_html=True,
        )
        with st.spinner("Analysing sentences…"):
            try:
                corrections = groq_sentence_corrections(
                    r["transcript"], groq_key, task_prompt
                )
                if not corrections:
                    st.markdown(
                        '<div class="fb-card fb-card-ai">No grammatical errors found. '
                        'The transcript appears grammatically correct.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    for i, c in enumerate(corrections, 1):
                        st.markdown(
                            f'<div class="correction-card">'
                            f'<div style="font-size:.68rem;color:#8892A4;text-transform:uppercase;'
                            f'letter-spacing:.06em;margin-bottom:.4rem">Correction {i}  ·  {c.get("rule","")}</div>'
                            f'<span class="correction-original">{c.get("original","")}</span>'
                            f'  →  '
                            f'<span class="correction-fixed">{c.get("corrected","")}</span>'
                            f'<div class="repeat-prompt">'
                            f'Please repeat after me: "{c.get("repeat_after_me","")}"</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            except Exception as e:
                st.warning(f"Corrections unavailable: {e}")
    else:
        st.markdown("### AI Coaching  ·  Groq / Llama-3.3")
        with st.spinner("Generating coaching…"):
            try:
                advice = groq_coaching(
                    transcript=r["transcript"], wpm=r["wpm"],
                    pause_rate=r["pause_rate"], filler_rate=r["filler_rate"],
                    fluency_score=sc, filler_top=top_sorted, api_key=groq_key,
                    task_prompt=task_prompt,
                    discourse_score=disc_score,
                )
                st.markdown(f'<div class="fb-card fb-card-ai">{advice}</div>',
                            unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Groq unavailable: {e}")
elif groq_key and r.get("demo"):
    st.info("AI coaching disabled in demo mode (requires a real transcript).")
else:
    st.markdown(
        '<div style="font-size:.76rem;color:#8892A4;margin-top:.35rem">'
        'Add a free Groq API key in the sidebar for AI coaching or sentence-level corrections '
        '(free at <code>console.groq.com</code>, no credit card).</div>',
        unsafe_allow_html=True,
    )

st.divider()


# ── 9. Benchmark comparison ───────────────────────────────────────────────────
st.markdown("## Benchmark Comparison")
bm1, bm2 = st.columns(2)

with bm1:
    for bm_name, bm_vals in BENCHMARKS.items():
        active = "  →  " if bm_name == benchmark else "      "
        passes = sc >= bm_vals["min_score"]
        cls    = "tr-pass" if passes else "tr-fail"
        mark   = "PASS" if passes else "BELOW"
        st.markdown(
            f"<div class='tr'>"
            f"<span style='color:#8892A4'>{active}{bm_name}</span>"
            f"<span class='{cls}'>{mark}  (>= {bm_vals['min_score']})</span></div>",
            unsafe_allow_html=True,
        )

with bm2:
    bm_now = BENCHMARKS[benchmark]
    lo, hi = bm_now["wpm"]
    for label, target_str, actual, actual_str, passes in [
        ("WPM target", f"{lo}–{hi}", r["wpm"], f"{r['wpm']:.0f}",
         lo <= r["wpm"] <= hi + 20),
        ("Pauses/min", f"<= {bm_now['pause_pm']:.1f}", r["pause_rate"],
         f"{r['pause_rate']:.1f}", r["pause_rate"] <= bm_now["pause_pm"]),
        ("Fillers/min", f"<= {bm_now['filler_pm']:.1f}", r["filler_rate"],
         f"{r['filler_rate']:.1f}", r["filler_rate"] <= bm_now["filler_pm"]),
    ]:
        cls  = "tr-pass" if passes else "tr-fail"
        mark = "PASS" if passes else "BELOW"
        st.markdown(
            f"<div class='tr'><span style='color:#8892A4'>{label}</span>"
            f"<span class='{cls}'>{mark}  {actual_str} / {target_str}</span></div>",
            unsafe_allow_html=True,
        )

st.divider()


# ── 10. Progress dashboard ────────────────────────────────────────────────────
history = get_history(username)
st.markdown(f"## Progress Dashboard  ·  {len(history)} session{'s' if len(history)!=1 else ''}")

if len(history) < 2:
    st.markdown(
        '<div style="color:#8892A4;font-size:.84rem;padding:.7rem 0">'
        'Record at least 2 sessions to see your progress charts.</div>',
        unsafe_allow_html=True,
    )
else:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(12, 2.4))
    fig.patch.set_facecolor("#0D1F35")
    xs    = range(len(history))
    dates = [h.get("date", h["timestamp"][:6]) for h in history]

    panels = [
        (axes[0], [h["score"]       for h in history], "Fluency Score", "#00D4AA",
         [(68,"--","#FF6B35"),(80,"-.","#FFD166")]),
        (axes[1], [h["wpm"]         for h in history], "WPM",           "#00D4AA",
         [(140,"--","#FF6B35"),(160,"-.","#FFD166")]),
        (axes[2], [h["pause_rate"]  for h in history], "Pauses / Min",  "#FFD166",
         [(3.0,"--","#FF6B35")]),
        (axes[3], [h["filler_rate"] for h in history], "Fillers / Min", "#FF6B35",
         [(2.0,"--","#FF6B35")]),
    ]
    for ax, data, ylabel, color, refs in panels:
        ax.set_facecolor("#0a1726")
        ax.plot(list(xs), data, color=color, marker="o", linewidth=1.7, markersize=4)
        for rv, ls, rc in refs:
            ax.axhline(rv, color=rc, linestyle=ls, linewidth=.85, alpha=.6)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(dates, rotation=28, ha="right", fontsize=5.5, color="#8892A4")
        ax.tick_params(axis="y", colors="#8892A4", labelsize=6)
        ax.set_ylabel(ylabel, color="#8892A4", fontsize=6)
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        for s in ["bottom","left"]: ax.spines[s].set_edgecolor("#1E3A5F")
    plt.tight_layout(pad=.65)
    st.pyplot(fig, use_container_width=True)

    for h in reversed(history[-5:]):
        c = score_color(h["score"])
        st.markdown(
            f"<div class='tr'>"
            f"<span style='color:#8892A4;font-size:.75rem'>{h['timestamp']}</span>"
            f"<span style='font-family:Space Mono,monospace;font-size:.77rem;color:{c}'>"
            f"{h['score']}  {h['label']}</span>"
            f"<span style='color:#8892A4;font-size:.75rem'>"
            f"{h['wpm']:.0f} WPM  ·  {h['pause_rate']:.1f} P  ·  {h['filler_rate']:.1f} F</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()


# ── 11. Export ────────────────────────────────────────────────────────────────
st.markdown("## Export")

report_dict = {
    "metadata": {"tool": "Speech Fluency Analyzer · Diego Palencia Research",
                 "generated": datetime.now().isoformat(), "username": username,
                 "benchmark": benchmark, "task_prompt": task_prompt,
                 "demo": r.get("demo", False)},
    "scores": {"fluency_score": sc, "label": score_label(sc),
               "wpm_score": r["wpm_score"], "pause_score": r["pause_score"],
               "filler_score": r["filler_score"],
               "discourse_coherence": disc_score},
    "features": {"wpm": r["wpm"], "articulation_rate": r.get("articulation_rate",0),
                 "pitch_std_hz": r.get("pitch_std",0), "pitch_score": r.get("pitch_score",50),
                 "whisper_confidence": r.get("confidence",50),
                 "pause_count": r["pause_count"], "pause_rate_per_min": r["pause_rate"],
                 "filler_count": r["filler_count"], "filler_rate_per_min": r["filler_rate"],
                 "connector_count": disc_data.get("count",0),
                 "connector_types": disc_data.get("types_used",0),
                 "grammar_issues": len(grammar_data),
                 "duration_seconds": round(r["duration_s"],1)},
    "transcript": r["transcript"],
}

ex1, ex2, ex3 = st.columns(3)
with ex1:
    st.download_button("Download JSON Report",
                        data=json.dumps(report_dict, indent=2),
                        file_name=f"fluency_{username}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json", use_container_width=True)
with ex2:
    st.download_button("Download Transcript",
                        data=r["transcript"],
                        file_name=f"transcript_{username}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain", use_container_width=True)
with ex3:
    if st.button("Generate PDF Report", use_container_width=True):
        with st.spinner("Building PDF…"):
            try:
                from core.report import build_pdf
                pdf_bytes = build_pdf(r, fb, username, benchmark, top_sorted)
                st.download_button("Download PDF", data=pdf_bytes,
                                    file_name=f"report_{username}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                    mime="application/pdf", use_container_width=True)
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

with st.expander("Research Basis and Methodology"):
    st.markdown("""
| Feature | Source | Finding |
|---|---|---|
| Speaking Rate | Lennon (1990) | 130–180 WPM native English; below 100 = disfluent |
| Articulation Rate | Kormos & Denes (2004) | WPM over speech-only time — purer fluency signal |
| Pause Rate | Tavakoli & Skehan (2005) | L2 speakers pause 3–4x more; >400ms disrupts comprehension |
| Pitch Variation | Hincks (2005) | F0 std dev below 20 Hz signals monotone delivery |
| Filler Rate | Skehan (1996) | High filler rate signals real-time planning difficulty |
| Discourse Coherence | Schmidt (1990) | Noticing Hypothesis: connector use signals discourse planning |

```
Fluency = (0.40 × WPM score) + (0.35 × Pause score) + (0.25 × Filler score)
```
""")

st.markdown("""
<div class='footer'>
  Speech Fluency Analyzer  ·  Project 03  ·  Diego Jose Palencia Robles<br>
  github.com/diegopalencia-research  ·  Guatemala City  ·  2025<br>
  <span style='opacity:.4'>
    Lennon (1990)  ·  Skehan (1996)  ·  Tavakoli & Skehan (2005)  ·
    Kormos & Denes (2004)  ·  Hincks (2005)  ·  Schmidt (1990)
  </span>
</div>
""", unsafe_allow_html=True)
