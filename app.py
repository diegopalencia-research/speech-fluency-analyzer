"""
app.py — Speech Fluency Analyzer
══════════════════════════════════════════════════════════════════════════════
Project 03 · Diego Jose Palencia Robles · github.com/diegopalencia-research

Input modes       : Live microphone  |  File upload (WAV/MP3/M4A/OGG)
Transcription     : Local Whisper (free, no key)  |  OpenAI Whisper API
Core features     : WPM · Pause rate · Filler rate
Extended features : Articulation rate · Pitch variation · Whisper confidence
Feedback          : Rule-based (always)  |  Groq AI coaching (optional/free)
Identity          : Username-keyed persistent progress across sessions
Export            : JSON report  |  TXT transcript  |  Branded PDF report

Research basis    : Lennon (1990) · Skehan (1996) · Tavakoli & Skehan (2005)
                    Kormos & Denes (2004) · Hincks (2005)
══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import numpy as np
import json
import os
import tempfile
from datetime import datetime

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
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
  --navy:   #0D1F35;
  --panel:  #0a1726;
  --card:   #162030;
  --blue:   #1E3A5F;
  --accent: #00D4AA;
  --a2:     #FF6B35;
  --gold:   #FFD166;
  --light:  #F0F4F8;
  --gray:   #8892A4;
  --border: rgba(0,212,170,0.15);
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background: var(--navy) !important;
  color: var(--light) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--panel) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif !important; }

/* ── Metrics ── */
[data-testid="metric-container"] {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: .9rem 1.1rem;
}
[data-testid="stMetricValue"] {
  color: var(--accent) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 1.7rem !important;
}
[data-testid="stMetricLabel"] {
  color: var(--gray) !important;
  font-size: .72rem !important;
  letter-spacing: .05em;
  text-transform: uppercase;
}

/* ── Typography ── */
h1 {
  font-family: 'Space Mono', monospace !important;
  color: var(--accent) !important;
  font-size: 1.75rem !important;
  letter-spacing: -.02em;
}
h2 {
  font-family: 'Space Mono', monospace !important;
  color: var(--light) !important;
  border-bottom: 1px solid var(--border);
  padding-bottom: .4rem;
  font-size: 1rem !important;
  letter-spacing: -.01em;
}
h3 {
  font-family: 'Space Mono', monospace !important;
  color: var(--accent) !important;
  font-size: .88rem !important;
}

/* ── Score badge ── */
.score-badge {
  font-family: 'Space Mono', monospace;
  font-size: 3.2rem;
  font-weight: 700;
  text-align: center;
  padding: 1.3rem 1rem .9rem;
  border-radius: 12px;
  border: 2px solid;
  line-height: 1.1;
}
.score-low  { color:#FF6B35; border-color:rgba(255,107,53,.3); background:rgba(255,107,53,.06); }
.score-mid  { color:#FFD166; border-color:rgba(255,209,102,.3);background:rgba(255,209,102,.06);}
.score-high { color:#00D4AA; border-color:rgba(0,212,170,.3);  background:rgba(0,212,170,.06); }
.score-pro  { color:#00f0c0; border-color:rgba(0,240,192,.4);  background:rgba(0,240,192,.08); }

/* ── Progress bars ── */
.fb-wrap { background:rgba(0,212,170,.07); border-radius:4px; height:7px; width:100%; margin:5px 0 10px; }
.fb-fill      { border-radius:4px; height:7px; background:linear-gradient(90deg,#00D4AA,#00f0c0); }
.fb-fill-warn { border-radius:4px; height:7px; background:linear-gradient(90deg,#FFD166,#ffba00); }
.fb-fill-bad  { border-radius:4px; height:7px; background:linear-gradient(90deg,#FF6B35,#ff4f00); }

/* ── Filler highlight ── */
.filler {
  background: rgba(255,107,53,.18);
  color: #FF6B35;
  border-radius: 3px;
  padding: 1px 4px;
  font-weight: 600;
}

/* ── Card surfaces ── */
.surface {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.1rem 1.3rem;
  margin-bottom: .6rem;
}

/* ── Transcript ── */
.transcript-box {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.2rem 1.5rem;
  line-height: 2;
  font-size: .92rem;
}

/* ── Feedback card ── */
.fb-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem 1.3rem;
  margin-bottom: .6rem;
  font-size: .88rem;
  line-height: 1.65;
}
.fb-card-ai { border-color:rgba(0,212,170,.35); background:rgba(0,212,170,.04); }

/* ── Table rows ── */
.tr {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: .4rem 0;
  border-bottom: 1px solid rgba(0,212,170,.08);
  font-size: .84rem;
}
.tr-pass { color:#00D4AA; font-family:'Space Mono',monospace; font-size:.79rem; }
.tr-fail { color:#FF6B35; font-family:'Space Mono',monospace; font-size:.79rem; }

/* ── User identity ── */
.identity-bar {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: .6rem 1rem;
  display: flex;
  align-items: center;
  gap: .7rem;
  font-family: 'Space Mono', monospace;
  font-size: .8rem;
  margin-bottom: .5rem;
}
.identity-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--accent);
  flex-shrink: 0;
}

/* ── Sidebar section labels ── */
.sb-label {
  font-size: .67rem;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: .1em;
  font-family: 'Space Mono', monospace;
  margin: 0 0 8px;
  display: flex;
  align-items: center;
  gap: 7px;
}
.sb-label::after { content:''; flex:1; height:1px; background:var(--border); }

/* ── Key card in sidebar ── */
.key-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 9px 12px;
  margin-bottom: 7px;
}
.key-card-head {
  font-size: .69rem;
  color: var(--gray);
  text-transform: uppercase;
  letter-spacing: .06em;
  margin-bottom: 4px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.badge {
  font-size: .6rem;
  padding: 1px 6px;
  border-radius: 20px;
  text-transform: uppercase;
  letter-spacing: .04em;
}
.badge-free { background:rgba(0,212,170,.15); color:var(--accent); }
.badge-opt  { background:rgba(136,146,164,.12); color:var(--gray); }

/* ── Recording live indicator ── */
@keyframes pulse-ring  { 0%{transform:scale(.8);opacity:1} 100%{transform:scale(2.4);opacity:0} }
@keyframes pulse-dot   { 0%,100%{opacity:1} 50%{opacity:.35} }
@keyframes bar-bounce  { 0%,100%{height:5px} 50%{height:20px} }

.rec-wrap {
  display:flex; align-items:center; gap:14px;
  background:rgba(255,107,53,.08); border:1px solid rgba(255,107,53,.3);
  border-radius:10px; padding:13px 16px; margin:10px 0;
}
.rec-dot-wrap { position:relative; width:14px; height:14px; flex-shrink:0; }
.rec-dot  { position:absolute; inset:0; background:#FF6B35; border-radius:50%; animation:pulse-dot 1.2s ease-in-out infinite; }
.rec-ring { position:absolute; inset:0; border:2px solid #FF6B35; border-radius:50%; animation:pulse-ring 1.2s ease-out infinite; }
.rec-bars { display:flex; align-items:center; gap:3px; height:24px; }
.rec-bar  { width:3px; background:#FF6B35; border-radius:2px; animation:bar-bounce .8s ease-in-out infinite; }
.rec-bar:nth-child(1){animation-delay:0s}
.rec-bar:nth-child(2){animation-delay:.13s}
.rec-bar:nth-child(3){animation-delay:.26s}
.rec-bar:nth-child(4){animation-delay:.39s}
.rec-bar:nth-child(5){animation-delay:.13s}
.rec-title { font-family:'Space Mono',monospace; font-size:.79rem; color:#FF6B35; letter-spacing:.07em; }
.rec-sub   { font-size:.75rem; color:var(--gray); margin-top:2px; }

/* ── Captured state ── */
.captured {
  background:rgba(0,212,170,.07); border:1px solid rgba(0,212,170,.28);
  border-radius:10px; padding:11px 16px; margin:10px 0;
  font-family:'Space Mono',monospace; font-size:.79rem; color:var(--accent);
  display:flex; align-items:center; gap:10px;
}
.captured::before { content:'✓'; font-size:1rem; }

/* ── Eyebrow ── */
.eyebrow {
  font-family: 'Space Mono', monospace;
  font-size: .68rem;
  color: var(--accent);
  letter-spacing: .16em;
  text-transform: uppercase;
  display: flex;
  align-items: center;
  gap: .7rem;
  margin-bottom: .9rem;
}
.eyebrow::before { content:''; width:32px; height:1px; background:var(--accent); }

/* ── Feature eval cards ── */
.eval-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem 1.1rem;
  height: 100%;
}
.eval-title {
  font-family: 'Space Mono', monospace;
  font-size: .72rem;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: .06em;
  margin-bottom: .5rem;
}
.eval-body { font-size: .8rem; color: var(--gray); line-height: 1.6; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
  font-family: 'Space Mono', monospace;
  font-size: .78rem;
  letter-spacing: .03em;
}
.stTabs [aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  border: 2px dashed var(--border) !important;
  border-radius: 10px !important;
  background: var(--card) !important;
}

/* ── Buttons ── */
.stButton > button {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  color: var(--light) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: .75rem !important;
  border-radius: 6px !important;
  transition: all .18s !important;
}
.stButton > button:hover { border-color:var(--accent) !important; color:var(--accent) !important; }

/* ── Progress bar ── */
.stProgress > div > div > div > div { background: var(--accent) !important; }
.stAlert { border-radius:8px !important; font-size:.86rem !important; }
code { font-family:'Space Mono',monospace !important; color:var(--accent) !important;
       background:rgba(0,212,170,.07) !important; padding:.1em .35em; border-radius:3px; }

/* ── Footer ── */
.footer {
  text-align:center; padding:2rem 0 1rem;
  color:var(--gray); font-size:.72rem;
  font-family:'Space Mono',monospace;
  border-top:1px solid var(--border); margin-top:3rem;
  letter-spacing:.04em; line-height:1.8;
}

/* ── Responsive ── */
@media (max-width:768px) {
  .stTabs [data-baseweb="tab"] { font-size:.68rem !important; padding:5px 7px !important; }
  [data-testid="stMetricValue"] { font-size:1.3rem !important; }
  .score-badge { font-size:2.6rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ── IMPORTS ───────────────────────────────────────────────────────────────────
from core.score   import (detect_fillers, annotate_transcript, compute_scores,
                           compute_articulation_score, BENCHMARKS,
                           score_label, score_css, score_color)
from core.storage import get_history, add_session, clear_history, export_json, import_json


# ── HELPERS ───────────────────────────────────────────────────────────────────

def bar_html(score: float, width: str = "100%") -> str:
    cls = "fb-fill" if score >= 68 else ("fb-fill-warn" if score >= 50 else "fb-fill-bad")
    return (f'<div class="fb-wrap" style="width:{width}">'
            f'<div class="{cls}" style="width:{int(score)}%"></div></div>')


# ── DEMO RESULT ───────────────────────────────────────────────────────────────

def make_demo() -> dict:
    wpm, dur_s = 118.4, 48.0
    dur_m      = dur_s / 60
    pause_count, filler_count = 6, 7
    pause_rate  = round(pause_count / dur_m, 2)
    filler_rate = round(filler_count / dur_m, 2)
    art_rate    = round(wpm * dur_m / max(dur_s - 14.0, 1) * 60, 1)  # simulated
    scores      = compute_scores(wpm, pause_rate, filler_rate)
    transcript  = (
        "So, uh, I wanted to talk about the project timeline, you know, "
        "because basically we have a few things to consider. Um, the first "
        "thing is the deadline, which is, like, coming up pretty soon. And, "
        "uh, I think we need to, right, prioritise the core features. "
        "The main deliverable is the dashboard and the scoring module."
    )
    fm = detect_fillers(transcript)
    return {
        "fluency_score":    scores["fluency"],
        "wpm":              wpm,
        "wpm_score":        scores["wpm_score"],
        "pause_count":      pause_count,
        "pause_rate":       pause_rate,
        "pause_score":      scores["pause_score"],
        "filler_count":     filler_count,
        "filler_rate":      filler_rate,
        "filler_score":     scores["filler_score"],
        "articulation_rate":art_rate,
        "art_score":        compute_articulation_score(art_rate),
        "pitch_std":        38.4,
        "pitch_score":      74.0,
        "confidence":       68.5,
        "duration_s":       dur_s,
        "transcript":       transcript,
        "filler_matches":   fm,
        "demo":             True,
        "pauses": [(3.1,3.6,.5),(9.4,10.1,.7),(17.2,17.9,.7),
                   (24.1,24.7,.6),(31.0,31.7,.7),(38.0,38.5,.5)],
        "y": None, "sr": 16000,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='font-family:Space Mono,monospace;font-size:.95rem;"
        "color:#00D4AA;margin-bottom:2px'>Speech Fluency Analyzer</div>"
        "<div style='font-size:.72rem;color:#8892A4;font-family:Space Mono,monospace;"
        "margin-bottom:12px'>Project 03 · Palencia Research</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── User identity ──────────────────────────────────────────────────────
    st.markdown('<div class="sb-label">Identity</div>', unsafe_allow_html=True)
    username = st.text_input(
        "username_input",
        value=st.session_state.get("username", ""),
        placeholder="Enter your name or username",
        label_visibility="collapsed",
    ).strip() or "default"
    st.session_state["username"] = username
    st.markdown(
        f"<div style='font-size:.72rem;color:#8892A4;margin-top:2px;margin-bottom:8px'>"
        f"Progress is saved under <code>{username}</code></div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── API Keys ───────────────────────────────────────────────────────────
    st.markdown('<div class="sb-label">API Keys</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="key-card">
      <div class="key-card-head">
        OpenAI Whisper
        <span class="badge badge-opt">Optional</span>
      </div>
      <div style='font-size:.72rem;color:#8892A4;margin-bottom:5px'>
        Faster transcription (~5 s vs ~30 s local)
      </div>
    </div>
    """, unsafe_allow_html=True)
    openai_key = st.text_input("ok", type="password", placeholder="sk-…",
                                label_visibility="collapsed")

    st.markdown("""
    <div class="key-card">
      <div class="key-card-head">
        Groq  /  Llama-3.3
        <span class="badge badge-free">Free</span>
      </div>
      <div style='font-size:.72rem;color:#8892A4;margin-bottom:5px'>
        AI coaching · console.groq.com
      </div>
    </div>
    """, unsafe_allow_html=True)
    groq_key = st.text_input("gk", type="password", placeholder="gsk_…",
                              label_visibility="collapsed")

    st.markdown(
        "<div style='font-size:.69rem;color:rgba(136,146,164,.55);margin-top:3px'>"
        "The app runs fully free without any keys.</div>",
        unsafe_allow_html=True,
    )

    openai_key = openai_key or st.secrets.get("OPENAI_API_KEY", "")
    groq_key   = groq_key   or st.secrets.get("GROQ_API_KEY",   "")

    st.divider()

    # ── Settings ───────────────────────────────────────────────────────────
    st.markdown('<div class="sb-label">Settings</div>', unsafe_allow_html=True)
    benchmark  = st.selectbox("Benchmark", list(BENCHMARKS.keys()), index=0)
    silence_db = st.slider("Silence threshold (dB)", 15, 45, 30,
                            help="Lower = more sensitive to quiet speech")
    min_pause  = st.slider("Min pause (ms)", 200, 800, 400, step=50)
    model_size = st.selectbox("Local Whisper model", ["tiny","base","small"], index=0,
                               help="tiny = fastest  ·  small = most accurate")

    st.divider()

    # ── Reference ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style='font-size:.72rem;color:#8892A4;line-height:1.85'>
    <span style='color:#00D4AA;font-weight:600'>Score formula</span><br>
    40% WPM · 35% Pause · 25% Filler<br><br>
    <span style='color:#00D4AA;font-weight:600'>Targets</span><br>
    Call center: ≥ 68  ·  Professional: ≥ 80<br><br>
    <span style='color:#00D4AA;font-weight:600'>Research basis</span><br>
    Lennon (1990) · Skehan (1996)<br>
    Tavakoli & Skehan (2005)<br>
    Kormos & Denes (2004) · Hincks (2005)
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Session history controls ───────────────────────────────────────────
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
        st.markdown(
            "<div style='font-size:.72rem;color:#8892A4;margin-bottom:6px'>"
            "Upload a previously exported <code>fluency_{username}.json</code>.</div>",
            unsafe_allow_html=True,
        )
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

# Active user indicator
if username != "default":
    st.markdown(
        f'<div class="identity-bar">'
        f'<div class="identity-dot"></div>'
        f'<span style="color:#8892A4">Logged in as</span>'
        f'<span style="color:#F0F4F8">{username}</span>'
        f'<span style="color:#8892A4;margin-left:auto;font-size:.72rem">'
        f'{len(get_history(username))} session(s) saved</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    "<p style='color:#8892A4;font-size:.92rem;max-width:720px;"
    "margin-top:-.3rem;line-height:1.75'>"
    "Upload or record a 30–60 second English speech sample. The analyzer extracts "
    "six acoustic and linguistic features "
    "(speaking rate, articulation rate, pause frequency, pitch variation, "
    "filler word rate, and transcription confidence) "
    "and returns a composite <b style='color:#F0F4F8'>Fluency Score from 0 to 100</b> "
    "calibrated against professional and call center benchmarks. "
    "Results include an annotated transcript, waveform, coaching feedback, "
    "a progress dashboard, and a branded PDF report. "
    "<b style='color:#00D4AA'>No API key required.</b>"
    "</p>",
    unsafe_allow_html=True,
)

# ── Feature evaluation cards ──────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
for col, title, body in [
    (c1, "Speaking Rate  (WPM  /  Articulation Rate)",
     "Measures words per minute over total duration, and separately over speech-only time "
     "(excluding pauses). Native English: 130–180 WPM. "
     "Articulation rate removes pauses for a purer fluency signal."),
    (c2, "Pause Frequency  /  Pitch Variation",
     "Counts involuntary gaps longer than 400 ms per minute. "
     "Also extracts fundamental frequency standard deviation — "
     "monotone delivery signals low engagement and reduced fluency."),
    (c3, "Filler Words  /  Transcription Confidence",
     "Detects 'uh', 'um', 'like', 'you know', and similar markers. "
     "Also reports Whisper's phoneme-level confidence as a proxy "
     "for articulation clarity and pronunciation precision."),
]:
    with col:
        col.markdown(
            f'<div class="eval-card">'
            f'<div class="eval-title">{title}</div>'
            f'<div class="eval-body">{body}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()


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

# ── Microphone ────────────────────────────────────────────────────────────────
with tab_mic:
    st.markdown(
        "<p style='color:#8892A4;font-size:.85rem;margin-bottom:.5rem'>"
        "Click the microphone button to start. Speak for 30–60 seconds, "
        "then click again to stop. Recommended: Chrome or Edge.</p>",
        unsafe_allow_html=True,
    )
    try:
        from audio_recorder_streamlit import audio_recorder
        recorded = audio_recorder(
            text="",
            recording_color="#FF6B35",
            neutral_color="#8892A4",
            icon_size="3x",
            pause_threshold=3.0,
            sample_rate=16000,
        )
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
            </div>
            """, unsafe_allow_html=True)
        else:
            audio_bytes, audio_suffix = recorded, ".wav"
            st.markdown(
                '<div class="captured">Recording captured  —  click Analyse Speech below</div>',
                unsafe_allow_html=True,
            )
            st.audio(recorded, format="audio/wav")
    except ImportError:
        st.warning(
            "Microphone recording requires `audio-recorder-streamlit`.  \n"
            "Run `pip install audio-recorder-streamlit` then restart.  \n"
            "Use the Upload tab in the meantime."
        )

# ── Upload ────────────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown(
        "<p style='color:#8892A4;font-size:.85rem'>"
        "WAV  ·  MP3  ·  M4A  ·  OGG  ·  Recommended: 30–60 seconds  ·  "
        "Accented English fully supported.</p>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("audio_upload", type=["wav","mp3","m4a","ogg"],
                                 label_visibility="collapsed")
    if uploaded:
        audio_bytes  = uploaded.read()
        audio_suffix = "." + uploaded.name.rsplit(".", 1)[-1].lower()
        st.audio(audio_bytes, format=f"audio/{audio_suffix.lstrip('.')}")

# ── Demo ──────────────────────────────────────────────────────────────────────
with tab_demo:
    st.markdown(
        "<p style='color:#8892A4;font-size:.85rem'>"
        "Explore the full dashboard instantly with a pre-loaded synthetic sample "
        "(48 s, realistic metrics, no API key or microphone needed).</p>",
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
        ai_tier = "Groq AI coaching" if groq_key else "Rule-based coaching"
        st.markdown(
            f"<span style='font-size:.78rem;color:#8892A4'>"
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

            st.write("Detecting pauses and extracting pitch…")
            pause_count, total_pause_s, pauses, speech_time_s, _ = detect_pauses(
                y, sr, silence_db=silence_db, min_pause_s=min_pause / 1000,
            )
            pause_rate       = round(pause_count / max(duration_m, .01), 2)
            pitch_std, pitch_score = extract_pitch_variation(y, sr)

            st.write("Transcribing…")
            # Local Whisper returns a dict with segments; API returns a str
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
                    tmp_path, language="en", fp16=False
                )
                transcript_raw = whisper_result["text"].strip()
            os.unlink(tmp_path)

            st.write("Computing features…")
            word_count      = len(transcript_raw.split())
            wpm             = round(word_count / max(duration_m, .01), 1)
            art_rate        = round(word_count / max(speech_time_s / 60, .01), 1)
            filler_matches  = detect_fillers(transcript_raw)
            filler_count    = len(filler_matches)
            filler_rate     = round(filler_count / max(duration_m, .01), 2)
            confidence      = extract_whisper_confidence(whisper_result)
            art_score       = compute_articulation_score(art_rate)

            scores = compute_scores(wpm, pause_rate, filler_rate)

            result = {
                "fluency_score":    scores["fluency"],
                "wpm":              wpm,
                "wpm_score":        scores["wpm_score"],
                "pause_count":      pause_count,
                "pause_rate":       pause_rate,
                "pause_score":      scores["pause_score"],
                "filler_count":     filler_count,
                "filler_rate":      filler_rate,
                "filler_score":     scores["filler_score"],
                "articulation_rate":art_rate,
                "art_score":        art_score,
                "pitch_std":        pitch_std,
                "pitch_score":      pitch_score,
                "confidence":       confidence,
                "duration_s":       duration_s,
                "transcript":       transcript_raw,
                "filler_matches":   filler_matches,
                "demo":             False,
                "pauses":           pauses,
                "y":                y,
                "sr":               sr,
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
    <div style='background:#162030;border:1px solid rgba(0,212,170,.12);
                border-radius:12px;padding:2.8rem 2rem;text-align:center;margin:1rem 0'>
      <div style='font-family:Space Mono,monospace;font-size:1rem;
                  color:#F0F4F8;margin-bottom:.6rem'>Three ways to begin</div>
      <div style='color:#8892A4;font-size:.86rem;max-width:440px;margin:0 auto;line-height:1.8'>
        <b style='color:#00D4AA'>Record</b>  —  live via browser microphone<br>
        <b style='color:#00D4AA'>Upload</b>  —  WAV, MP3, M4A, or OGG file<br>
        <b style='color:#00D4AA'>Demo</b>  —  explore the dashboard instantly
      </div>
      <div style='color:rgba(136,146,164,.45);font-size:.74rem;margin-top:1.4rem'>
        No API key required  ·  Fully free by default
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
r  = result
sc = r["fluency_score"]
bm = BENCHMARKS[benchmark]


# ── SECTION 1: Score overview ─────────────────────────────────────────────────
st.markdown("## Score Overview")
col_score, col_metrics = st.columns([1, 2], gap="large")

with col_score:
    lbl = score_label(sc)
    clr = score_color(sc)
    st.markdown(
        f'<div class="score-badge {score_css(sc)}">'
        f'{sc}<br>'
        f'<span style="font-size:.85rem;letter-spacing:.07em;opacity:.8">{lbl}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    delta     = round(sc - bm["min_score"], 1)
    d_color   = "#00D4AA" if delta >= 0 else "#FF6B35"
    d_sign    = "+" if delta >= 0 else ""
    st.markdown(
        f"<div style='text-align:center;font-family:Space Mono,monospace;"
        f"font-size:.78rem;color:{d_color};margin-top:.3rem'>"
        f"{d_sign}{delta} vs {benchmark}</div>",
        unsafe_allow_html=True,
    )

with col_metrics:
    r1, r2, r3 = st.columns(3)
    r1.metric("Words / Min",   f"{r['wpm']:.0f}")
    r2.metric("Pauses / Min",  f"{r['pause_rate']:.1f}")
    r3.metric("Fillers / Min", f"{r['filler_rate']:.1f}")
    dur_m2 = int(r["duration_s"] // 60)
    dur_s2 = int(r["duration_s"] % 60)
    r4, r5, r6 = st.columns(3)
    r4.metric("Duration",      f"{dur_m2}m {dur_s2}s")
    r5.metric("Pause Count",   str(r["pause_count"]))
    r6.metric("Filler Count",  str(r["filler_count"]))

st.divider()


# ── SECTION 2: Core score decomposition ──────────────────────────────────────
st.markdown("## Score Decomposition  (Core Formula)")
dc1, dc2, dc3 = st.columns(3)

def decomp_block(col, title, raw, weight, note):
    pct   = int(raw)
    bar_c = "fb-fill" if raw >= 68 else ("fb-fill-warn" if raw >= 50 else "fb-fill-bad")
    col.markdown(
        f"<div style='font-size:.7rem;color:#8892A4;text-transform:uppercase;"
        f"letter-spacing:.06em'>{title}</div>"
        f"<div style='font-family:Space Mono,monospace;font-size:1.55rem;color:#00D4AA'>"
        f"{raw:.0f}<span style='font-size:.8rem;color:#8892A4'>/100</span>"
        f"<span style='font-size:.7rem;color:#8892A4;margin-left:.5rem'>× {weight}</span></div>"
        f'<div class="fb-wrap"><div class="{bar_c}" style="width:{pct}%"></div></div>'
        f"<div style='font-size:.75rem;color:#8892A4'>{note}</div>",
        unsafe_allow_html=True,
    )

decomp_block(dc1, "Speaking Rate",  r["wpm_score"],    "0.40", f"{r['wpm']:.0f} WPM  ·  target 140–160")
decomp_block(dc2, "Pause Control",  r["pause_score"],  "0.35", f"{r['pause_count']} pauses detected")
decomp_block(dc3, "Filler Words",   r["filler_score"], "0.25", f"{r['filler_count']} fillers found")

st.divider()


# ── SECTION 3: Extended acoustic features ────────────────────────────────────
st.markdown("## Extended Acoustic Analysis")
ec1, ec2, ec3 = st.columns(3)

def ext_block(col, title, value_str, raw, note):
    pct   = int(raw)
    bar_c = "fb-fill" if raw >= 68 else ("fb-fill-warn" if raw >= 50 else "fb-fill-bad")
    col.markdown(
        f"<div style='font-size:.7rem;color:#8892A4;text-transform:uppercase;"
        f"letter-spacing:.06em'>{title}</div>"
        f"<div style='font-family:Space Mono,monospace;font-size:1.2rem;color:#00D4AA'>"
        f"{value_str}</div>"
        f"<div style='font-size:.72rem;color:#8892A4;margin-bottom:3px'>"
        f"Score: {raw:.0f}/100</div>"
        f'<div class="fb-wrap"><div class="{bar_c}" style="width:{pct}%"></div></div>'
        f"<div style='font-size:.72rem;color:#8892A4'>{note}</div>",
        unsafe_allow_html=True,
    )

ext_block(ec1, "Articulation Rate",
          f"{r.get('articulation_rate',0):.0f} WPM",
          r.get("art_score", 50),
          "Speech-only WPM (pauses excluded)  ·  Kormos & Denes (2004)")
ext_block(ec2, "Pitch Variation",
          f"F0 std {r.get('pitch_std',0):.0f} Hz",
          r.get("pitch_score", 50),
          "F0 standard deviation  ·  Hincks (2005)")
ext_block(ec3, "Transcription Confidence",
          f"{r.get('confidence',50):.0f}/100",
          r.get("confidence", 50),
          "Whisper avg_logprob — phoneme clarity proxy")

st.divider()


# ── SECTION 4: Waveform ───────────────────────────────────────────────────────
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
        '<div style="background:#162030;border:1px solid rgba(0,212,170,.12);'
        'border-radius:8px;padding:1.1rem;text-align:center;color:#8892A4;font-size:.82rem">'
        'Demo mode  —  upload or record audio to see the annotated waveform.</div>',
        unsafe_allow_html=True,
    )

st.divider()


# ── SECTION 5: Annotated transcript ──────────────────────────────────────────
st.markdown("## Annotated Transcript")
filler_matches = r.get("filler_matches") or detect_fillers(r["transcript"])
annotated      = annotate_transcript(r["transcript"], filler_matches)
st.markdown(f'<div class="transcript-box">{annotated}</div>', unsafe_allow_html=True)
st.markdown(
    "<span style='font-size:.73rem;color:#8892A4'>"
    "<span class='filler'>highlighted</span>  =  filler word detected</span>",
    unsafe_allow_html=True,
)

st.divider()


# ── SECTION 6: Coaching feedback ─────────────────────────────────────────────
st.markdown("## Coaching Feedback")

from core.feedback import rule_based_feedback, groq_coaching

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
)

# Overall
banner_c = "#00D4AA" if sc >= 68 else ("#FFD166" if sc >= 50 else "#FF6B35")
st.markdown(
    f'<div class="surface" style="border-color:rgba(0,212,170,.25);color:{banner_c}">'
    f'{fb["overall"]}</div>',
    unsafe_allow_html=True,
)

# Feature cards
fc1, fc2, fc3 = st.columns(3)
for col, key, title in [(fc1,"wpm","Speaking Rate"),(fc2,"pauses","Pause Control"),(fc3,"fillers","Filler Words")]:
    d = fb[key]
    col.markdown(
        f'<div class="fb-card">'
        f'<div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;'
        f'color:#8892A4;margin-bottom:.35rem">{title}</div>'
        f'{d["advice"]}</div>',
        unsafe_allow_html=True,
    )

if top_sorted:
    ts_html = "  ·  ".join(f'<span class="filler">{w}</span> ×{n}' for w, n in top_sorted)
    st.markdown(
        f'<div style="font-size:.79rem;color:#8892A4;margin-top:-.2rem;margin-bottom:.6rem">'
        f'Most frequent fillers: {ts_html}</div>',
        unsafe_allow_html=True,
    )

# Priority
st.markdown(
    f'<div class="fb-card" style="border-color:rgba(255,209,102,.25)">'
    f'<div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;'
    f'color:#8892A4;margin-bottom:.35rem">Priority Focus</div>'
    f'{fb["priority"]}</div>',
    unsafe_allow_html=True,
)

# Practice drills
with st.expander("Practice Drills", expanded=True):
    for ex in fb["exercises"]:
        st.markdown(
            f'<div style="padding:.3rem 0;font-size:.86rem;color:#F0F4F8">{ex}</div>',
            unsafe_allow_html=True,
        )

# Groq AI coaching
if groq_key and not r.get("demo"):
    st.markdown("---")
    st.markdown("### AI Coaching  ·  Groq / Llama-3.3")
    with st.spinner("Generating coaching…"):
        try:
            advice = groq_coaching(
                transcript=r["transcript"], wpm=r["wpm"],
                pause_rate=r["pause_rate"], filler_rate=r["filler_rate"],
                fluency_score=sc, filler_top=top_sorted, api_key=groq_key,
            )
            st.markdown(f'<div class="fb-card fb-card-ai">{advice}</div>',
                        unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Groq unavailable: {e}")
elif groq_key and r.get("demo"):
    st.info("AI coaching disabled in demo mode (requires a real transcript).")
else:
    st.markdown(
        '<div style="font-size:.77rem;color:#8892A4;margin-top:.4rem">'
        'Add a free Groq API key in the sidebar for AI-powered, '
        'transcript-specific coaching (free at <code>console.groq.com</code>, no credit card).'
        '</div>',
        unsafe_allow_html=True,
    )

st.divider()


# ── SECTION 7: Benchmark comparison ──────────────────────────────────────────
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
            f"<span class='{cls}'>{mark}  (>= {bm_vals['min_score']})</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

with bm2:
    bm_now = BENCHMARKS[benchmark]
    lo, hi = bm_now["wpm"]
    for label, target_str, actual, actual_str, passes in [
        ("WPM target",
         f"{lo}–{hi}",
         r["wpm"],
         f"{r['wpm']:.0f}",
         lo <= r["wpm"] <= hi + 20),
        ("Pauses/min target",
         f"<= {bm_now['pause_pm']:.1f}",
         r["pause_rate"],
         f"{r['pause_rate']:.1f}",
         r["pause_rate"] <= bm_now["pause_pm"]),
        ("Fillers/min target",
         f"<= {bm_now['filler_pm']:.1f}",
         r["filler_rate"],
         f"{r['filler_rate']:.1f}",
         r["filler_rate"] <= bm_now["filler_pm"]),
    ]:
        cls  = "tr-pass" if passes else "tr-fail"
        mark = "PASS" if passes else "BELOW"
        st.markdown(
            f"<div class='tr'>"
            f"<span style='color:#8892A4'>{label}</span>"
            f"<span class='{cls}'>{mark}  {actual_str} / {target_str}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()


# ── SECTION 8: Progress dashboard ────────────────────────────────────────────
history = get_history(username)
st.markdown(f"## Progress Dashboard  ·  {len(history)} session{'s' if len(history)!=1 else ''}")

if len(history) < 2:
    st.markdown(
        '<div style="color:#8892A4;font-size:.86rem;padding:.8rem 0">'
        'Record at least 2 sessions to see your progress charts.</div>',
        unsafe_allow_html=True,
    )
else:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(12, 2.5))
    fig.patch.set_facecolor("#0D1F35")

    xs      = range(len(history))
    dates   = [h.get("date", h["timestamp"][:6]) for h in history]
    panels  = [
        (axes[0], [h["score"]           for h in history], "Fluency Score",    "#00D4AA",
         [(68,"--","#FF6B35"),(80,"-.","#FFD166")]),
        (axes[1], [h["wpm"]             for h in history], "WPM",              "#00D4AA",
         [(140,"--","#FF6B35"),(160,"-.","#FFD166")]),
        (axes[2], [h["pause_rate"]      for h in history], "Pauses / Min",     "#FFD166",
         [(3.0,"--","#FF6B35")]),
        (axes[3], [h["filler_rate"]     for h in history], "Fillers / Min",    "#FF6B35",
         [(2.0,"--","#FF6B35")]),
    ]

    for ax, data, ylabel, color, refs in panels:
        ax.set_facecolor("#0a1726")
        ax.plot(list(xs), data, color=color, marker="o", linewidth=1.8, markersize=4.5)
        for rv, ls, rc in refs:
            ax.axhline(rv, color=rc, linestyle=ls, linewidth=.9, alpha=.6)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(dates, rotation=28, ha="right", fontsize=6, color="#8892A4")
        ax.tick_params(axis="y", colors="#8892A4", labelsize=6.5)
        ax.set_ylabel(ylabel, color="#8892A4", fontsize=6.5)
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        for s in ["bottom","left"]: ax.spines[s].set_edgecolor("#1E3A5F")

    plt.tight_layout(pad=.7)
    st.pyplot(fig, use_container_width=True)

    for h in reversed(history[-6:]):
        c = score_color(h["score"])
        st.markdown(
            f"<div class='tr'>"
            f"<span style='color:#8892A4;font-size:.77rem'>{h['timestamp']}</span>"
            f"<span style='font-family:Space Mono,monospace;font-size:.79rem;color:{c}'>"
            f"{h['score']}  {h['label']}</span>"
            f"<span style='color:#8892A4;font-size:.77rem'>"
            f"{h['wpm']:.0f} WPM  ·  {h['pause_rate']:.1f} P  ·  {h['filler_rate']:.1f} F</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()


# ── SECTION 9: Export ─────────────────────────────────────────────────────────
st.markdown("## Export")

report_dict = {
    "metadata": {
        "tool":      "Speech Fluency Analyzer · Diego Palencia Research",
        "generated": datetime.now().isoformat(),
        "username":  username,
        "benchmark": benchmark,
        "demo":      r.get("demo", False),
    },
    "scores": {
        "fluency_score": sc,
        "label":         score_label(sc),
        "wpm_score":     r["wpm_score"],
        "pause_score":   r["pause_score"],
        "filler_score":  r["filler_score"],
    },
    "features": {
        "wpm":                 r["wpm"],
        "articulation_rate":   r.get("articulation_rate", 0),
        "pitch_std_hz":        r.get("pitch_std", 0),
        "pitch_score":         r.get("pitch_score", 50),
        "whisper_confidence":  r.get("confidence", 50),
        "pause_count":         r["pause_count"],
        "pause_rate_per_min":  r["pause_rate"],
        "filler_count":        r["filler_count"],
        "filler_rate_per_min": r["filler_rate"],
        "duration_seconds":    round(r["duration_s"], 1),
    },
    "transcript": r["transcript"],
    "formula":    "Fluency = 0.40×WPM + 0.35×Pause + 0.25×Filler",
}

ex1, ex2, ex3 = st.columns(3)
with ex1:
    st.download_button(
        "Download JSON Report", data=json.dumps(report_dict, indent=2),
        file_name=f"fluency_{username}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json", use_container_width=True,
    )
with ex2:
    st.download_button(
        "Download Transcript", data=r["transcript"],
        file_name=f"transcript_{username}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain", use_container_width=True,
    )
with ex3:
    if st.button("Generate PDF Report", use_container_width=True):
        with st.spinner("Building PDF…"):
            try:
                from core.report import build_pdf
                pdf_bytes = build_pdf(r, fb, username, benchmark, top_sorted)
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name=f"fluency_report_{username}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")


# ── Research expander ─────────────────────────────────────────────────────────
with st.expander("Research Basis and Methodology"):
    st.markdown("""
| Feature | Source | Empirical Finding |
|---|---|---|
| **Speaking Rate (WPM)** | Lennon (1990) | Native English: 130–180 WPM; below 100 = disfluent |
| **Articulation Rate** | Kormos & Denes (2004) | WPM over speech-only time — purer fluency signal than total WPM |
| **Pause Rate** | Tavakoli & Skehan (2005) | L2 speakers pause 3–4x more; pauses >400ms disrupt comprehension |
| **Pitch Variation** | Hincks (2005) | F0 std-dev below 20 Hz signals monotone delivery and low engagement |
| **Filler Words** | Skehan (1996) | High filler rate signals real-time planning difficulty |

**Core formula:**
```
Fluency = (0.40 × WPM score) + (0.35 × Pause score) + (0.25 × Filler score)
```
Extended features (articulation rate, pitch variation, confidence) are displayed
separately and not included in the composite to preserve the literature-validated weights.

Part of: **Palencia (2025). Computational Feature Extraction for Human Performance Prediction.**
""")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
  Speech Fluency Analyzer  ·  Project 03  ·  Diego Jose Palencia Robles<br>
  github.com/diegopalencia-research  ·  Guatemala City  ·  2025<br>
  <span style='opacity:.45'>
    Lennon (1990)  ·  Skehan (1996)  ·  Tavakoli & Skehan (2005)  ·
    Kormos & Denes (2004)  ·  Hincks (2005)
  </span>
</div>
""", unsafe_allow_html=True)
