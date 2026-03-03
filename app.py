"""
app.py — Speech Fluency Analyzer
══════════════════════════════════════════════════════════════════════════════
Project 03 · Diego Jose Palencia Robles · github.com/diegopalencia-research

Two input modes:
  • Upload  — WAV / MP3 / M4A file
  • Record  — Live microphone via browser

Two transcription backends:
  • Local Whisper (default, free, no API key)
  • OpenAI Whisper API (optional, faster)

Two feedback tiers:
  • Rule-based (always available, free)
  • Groq AI coaching (optional, free API at console.groq.com)

Research basis:
  Lennon (1990) · Skehan (1996) · Tavakoli & Skehan (2005)
══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import numpy as np
import json
import os
import io
import tempfile
from datetime import datetime

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Speech Fluency Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

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
  --border: rgba(0,212,170,0.18);
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
  border-radius: 10px;
  padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] {
  color: var(--accent) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 1.8rem !important;
}
[data-testid="stMetricLabel"] {
  color: var(--gray) !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

/* ── Typography ── */
h1 { font-family:'Space Mono',monospace !important; color:var(--accent) !important; font-size:1.9rem !important; }
h2 { font-family:'Space Mono',monospace !important; color:var(--light) !important;
     border-bottom:1px solid var(--border); padding-bottom:.5rem; font-size:1.1rem !important; }
h3 { font-family:'Space Mono',monospace !important; color:var(--accent) !important; font-size:.95rem !important; }

/* ── Score badge ── */
.score-badge {
  font-family: 'Space Mono', monospace;
  font-size: 3.5rem;
  font-weight: 700;
  text-align: center;
  padding: 1.4rem 1rem 1rem;
  border-radius: 14px;
  border: 2px solid;
  line-height: 1.1;
}
.score-low  { color:#FF6B35; border-color:rgba(255,107,53,.35); background:rgba(255,107,53,.07); }
.score-mid  { color:#FFD166; border-color:rgba(255,209,102,.35);background:rgba(255,209,102,.07);}
.score-high { color:#00D4AA; border-color:rgba(0,212,170,.35);  background:rgba(0,212,170,.07); }
.score-pro  { color:#00f0c0; border-color:rgba(0,240,192,.45);  background:rgba(0,240,192,.09); }

/* ── Feature progress bars ── */
.fb-wrap { background:rgba(0,212,170,.08); border-radius:5px; height:8px; width:100%; margin:5px 0 12px; }
.fb-fill  { border-radius:5px; height:8px; background:linear-gradient(90deg,#00D4AA,#00f0c0); }
.fb-fill-warn { background:linear-gradient(90deg,#FFD166,#ffba00); }
.fb-fill-bad  { background:linear-gradient(90deg,#FF6B35,#ff4f00); }

/* ── Filler highlight ── */
.filler { background:rgba(255,107,53,.22); color:#FF6B35; border-radius:4px;
          padding:1px 4px; font-weight:600; }

/* ── Transcript box ── */
.transcript-box {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.3rem 1.6rem;
  line-height: 2;
  font-size: .95rem;
}

/* ── Feedback card ── */
.fb-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.1rem 1.4rem;
  margin-bottom: .75rem;
  font-size: .9rem;
  line-height: 1.65;
}
.fb-card-groq {
  border-color: rgba(0,212,170,.4);
  background: rgba(0,212,170,.05);
}

/* ── Benchmark row ── */
.bm-row { display:flex; justify-content:space-between; align-items:center;
          padding:.45rem 0; border-bottom:1px solid rgba(0,212,170,.1); font-size:.86rem; }
.bm-pass { color:#00D4AA; font-family:'Space Mono',monospace; font-size:.82rem; }
.bm-fail { color:#FF6B35; font-family:'Space Mono',monospace; font-size:.82rem; }

/* ── Input area ── */
[data-testid="stFileUploader"] {
  border: 2px dashed var(--border) !important;
  border-radius: 10px !important;
  background: var(--card) !important;
}
.stTabs [data-baseweb="tab"] { font-family:'Space Mono',monospace; font-size:.82rem; }
.stTabs [aria-selected="true"] { color:var(--accent) !important; border-bottom-color:var(--accent) !important; }

/* ── Buttons ── */
.stButton > button {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  color: var(--light) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: .8rem !important;
  border-radius: 6px !important;
  transition: all .2s !important;
}
.stButton > button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }

/* ── Progress bar ── */
.stProgress > div > div > div > div { background: var(--accent) !important; }

/* ── Alert/info ── */
.stAlert { border-radius: 8px !important; font-size: .88rem !important; }

/* ── Progress chart container ── */
.progress-chart-wrap { background: var(--card); border:1px solid var(--border);
                       border-radius:10px; padding:1rem; }

/* ── Footer ── */
.footer {
  text-align:center; padding:2.5rem 0 1rem;
  color:var(--gray); font-size:.75rem;
  font-family:'Space Mono',monospace;
  border-top:1px solid var(--border); margin-top:3rem;
  letter-spacing:.04em;
}

/* ── Eyebrow label ── */
.eyebrow {
  font-family:'Space Mono',monospace;
  font-size:.72rem; color:var(--accent);
  letter-spacing:.15em; text-transform:uppercase;
  display:flex; align-items:center; gap:.7rem; margin-bottom:1rem;
}
.eyebrow::before { content:''; display:inline-block; width:36px; height:1px; background:var(--accent); }

/* ── Code ── */
code { font-family:'Space Mono',monospace !important; color:var(--accent) !important;
       background:rgba(0,212,170,.08) !important; padding:.1em .35em; border-radius:4px; }
</style>
""", unsafe_allow_html=True)


# ── IMPORTS (lazy-safe) ───────────────────────────────────────────────────────
from core.score    import (detect_fillers, annotate_transcript, compute_scores,
                            BENCHMARKS, score_label, score_css, score_color)
from core.storage  import get_history, add_session, clear_history, export_json, import_json


# ── HELPERS ───────────────────────────────────────────────────────────────────

def bar_html(score: float) -> str:
    pct  = int(score)
    cls  = "fb-fill" if score >= 68 else ("fb-fill-warn" if score >= 50 else "fb-fill-bad")
    return f'<div class="fb-wrap"><div class="{cls}" style="width:{pct}%"></div></div>'


def _save_tmp(audio_bytes: bytes, suffix: str = ".wav") -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        return f.name


# ── DEMO RESULT ───────────────────────────────────────────────────────────────

def make_demo() -> dict:
    np.random.seed(7)
    wpm         = 118.4
    dur_s       = 48.0
    dur_m       = dur_s / 60
    pause_count = 6
    pause_rate  = round(pause_count / dur_m, 2)
    filler_count= 7
    filler_rate = round(filler_count / dur_m, 2)
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
        "fluency_score": scores["fluency"],
        "wpm": wpm, "pause_count": pause_count, "pause_rate": pause_rate,
        "filler_count": filler_count, "filler_rate": filler_rate,
        "duration_s": dur_s, "transcript": transcript,
        "filler_matches": fm, "demo": True,
        **{k: scores[k] for k in ("wpm_score","pause_score","filler_score")},
        "pauses": [(3.1,3.6,.5),(9.4,10.1,.7),(17.2,17.9,.7),(24.1,24.7,.6),(31.0,31.7,.7),(38.0,38.5,.5)],
        "y": None, "sr": 16000,
    }


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎙️ Fluency Analyzer")
    st.markdown(
        "<span style='font-size:.75rem;color:#8892A4;font-family:Space Mono,monospace'>"
        "Project 03 · Palencia Research</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── API Keys (all optional) ──
    with st.expander("🔑 API Keys (all optional)", expanded=False):
        st.markdown(
            "<span style='font-size:.8rem;color:#8892A4'>"
            "The app runs fully free without any keys. "
            "Keys unlock faster transcription and AI coaching.</span>",
            unsafe_allow_html=True,
        )
        openai_key = st.text_input("OpenAI API key", type="password",
                                    placeholder="sk-… (faster Whisper API)",
                                    help="Optional. Enables OpenAI Whisper API (~5s vs ~30s local).")
        groq_key   = st.text_input("Groq API key", type="password",
                                    placeholder="gsk_… (free at console.groq.com)",
                                    help="Optional. Free tier. Enables AI coaching via Llama-3.3-70b.")
        # Fall back to Streamlit secrets
        openai_key = openai_key or st.secrets.get("OPENAI_API_KEY", "")
        groq_key   = groq_key   or st.secrets.get("GROQ_API_KEY",   "")

    st.divider()

    # ── Settings ──
    st.markdown("**Settings**")
    benchmark   = st.selectbox("Benchmark", list(BENCHMARKS.keys()), index=0)
    silence_db  = st.slider("Silence threshold (dB)", 15, 45, 30,
                             help="Lower = more sensitive to quiet speech")
    min_pause   = st.slider("Min pause (ms)", 200, 800, 400, step=50,
                             help="Pauses shorter than this are ignored")
    model_size  = st.selectbox("Local Whisper model",
                                ["tiny", "base", "small"],
                                index=0,
                                help="tiny=fastest/free · base=better accuracy · small=best accuracy")

    st.divider()

    # ── Reference card ──
    st.markdown("""
    <div style='font-size:.76rem;color:#8892A4;line-height:1.8'>
    <b style='color:#00D4AA'>Formula</b><br>
    40% WPM · 35% Pause · 25% Filler<br><br>
    <b style='color:#00D4AA'>Targets</b><br>
    Call center entry: ≥ 68<br>
    Professional: ≥ 80<br><br>
    <b style='color:#00D4AA'>References</b><br>
    Lennon (1990)<br>
    Skehan (1996)<br>
    Tavakoli & Skehan (2005)
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── History controls ──
    col_exp, col_clr = st.columns(2)
    with col_exp:
        st.download_button(
            "⬇️ Export",
            data=export_json(),
            file_name="fluency_history.json",
            mime="application/json",
            use_container_width=True,
            help="Download your full session history as JSON",
        )
    with col_clr:
        if st.button("🗑️ Clear", use_container_width=True):
            clear_history(); st.rerun()

    # ── Import ──
    with st.expander("📥 Import history"):
        uploaded_hist = st.file_uploader("Upload fluency_history.json",
                                          type="json", label_visibility="collapsed")
        if uploaded_hist:
            ok, msg = import_json(uploaded_hist.read().decode())
            (st.success if ok else st.error)(msg)
            if ok: st.rerun()


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="eyebrow">PROJECT 03 · SPEECH SCIENCE · L2 ASSESSMENT</div>',
            unsafe_allow_html=True)
st.title("Speech Fluency Analyzer")
st.markdown(
    "<p style='color:#8892A4;font-size:.95rem;max-width:660px;margin-top:-.4rem'>"
    "Record or upload English speech. The analyzer extracts acoustic features — "
    "words per minute, pause frequency, filler rate — and returns a composite fluency "
    "score grounded in peer-reviewed L2 research. "
    "<b style='color:#F0F4F8'>No API key required to get started.</b></p>",
    unsafe_allow_html=True,
)
st.divider()


# ── INPUT TABS ────────────────────────────────────────────────────────────────
tab_mic, tab_upload, tab_demo = st.tabs(
    ["🎙️  Record (live microphone)", "📁  Upload file", "🧪  Demo mode"]
)

audio_bytes: bytes | None = None
audio_suffix = ".wav"

# ── TAB 1: MICROPHONE ─────────────────────────────────────────────────────────
with tab_mic:
    st.markdown(
        "<p style='color:#8892A4;font-size:.88rem'>"
        "Click the microphone to start recording. Speak for 30–60 seconds, "
        "then click again to stop. Works on Chrome/Edge/Safari.</p>",
        unsafe_allow_html=True,
    )
    try:
        from audio_recorder_streamlit import audio_recorder
        recorded = audio_recorder(
            text="",
            recording_color="#00D4AA",
            neutral_color="#8892A4",
            icon_size="3x",
            pause_threshold=3.0,
            sample_rate=16000,
        )
        if recorded:
            audio_bytes  = recorded
            audio_suffix = ".wav"
            st.audio(recorded, format="audio/wav")
            st.success("✅ Recording captured — click **Analyse** below.")
    except ImportError:
        st.warning(
            "The microphone recorder requires `audio-recorder-streamlit`. "
            "Install it with: `pip install audio-recorder-streamlit`  \n"
            "Use the **Upload file** tab in the meantime."
        )

# ── TAB 2: FILE UPLOAD ────────────────────────────────────────────────────────
with tab_upload:
    st.markdown(
        "<p style='color:#8892A4;font-size:.88rem'>"
        "WAV · MP3 · M4A · OGG  ·  Recommended: 30–60 seconds  ·  "
        "Accented English is fully supported via Whisper.</p>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Drop audio file here", type=["wav","mp3","m4a","ogg"],
        label_visibility="collapsed",
    )
    if uploaded:
        audio_bytes  = uploaded.read()
        audio_suffix = "." + uploaded.name.rsplit(".", 1)[-1].lower()
        st.audio(audio_bytes, format=f"audio/{audio_suffix.lstrip('.')}")

# ── TAB 3: DEMO ───────────────────────────────────────────────────────────────
with tab_demo:
    st.markdown(
        "<p style='color:#8892A4;font-size:.88rem'>"
        "No microphone or file needed. Explore the full dashboard with a "
        "synthetic 48-second English sample — pre-loaded with realistic metrics.</p>",
        unsafe_allow_html=True,
    )
    run_demo = st.button("▶  Run demo analysis", use_container_width=False)


# ── ANALYSE BUTTON ────────────────────────────────────────────────────────────
st.divider()
result: dict | None = None

if run_demo:
    result = make_demo()

elif audio_bytes:
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_analysis = st.button("▶  Analyse speech", use_container_width=True, type="primary")
    with col_info:
        backend = "OpenAI Whisper API" if openai_key else f"Local Whisper ({model_size})"
        ai_tier = "Groq AI coaching" if groq_key else "Rule-based coaching (free)"
        st.markdown(
            f"<span style='font-size:.8rem;color:#8892A4'>"
            f"Transcription: <b style='color:#00D4AA'>{backend}</b> · "
            f"Feedback: <b style='color:#00D4AA'>{ai_tier}</b></span>",
            unsafe_allow_html=True,
        )
    if not run_analysis:
        st.stop()

    with st.status("🎙️ Analysing your speech…", expanded=True) as status:
        try:
            import librosa
            from core.analyze    import load_audio, detect_pauses, render_waveform
            from core.transcribe import transcribe

            # 1. Load audio
            st.write("⏳ Loading audio…")
            y, sr, duration_s, tmp_path = load_audio(audio_bytes, suffix=audio_suffix)
            duration_m = duration_s / 60

            # 2. Pause detection
            st.write("🔍 Detecting pauses…")
            pause_count, total_pause_s, pauses, _ = detect_pauses(
                y, sr,
                silence_db=silence_db,
                min_pause_s=min_pause / 1000,
            )
            pause_rate = round(pause_count / max(duration_m, 0.01), 2)

            # 3. Transcription
            st.write("📝 Transcribing…")
            transcript = transcribe(
                tmp_path,
                use_api=bool(openai_key),
                api_key=openai_key or None,
                model_size=model_size,
            )
            os.unlink(tmp_path)

            # 4. WPM + fillers
            st.write("📊 Computing features…")
            word_count   = len(transcript.split())
            wpm          = round(word_count / max(duration_m, 0.01), 1)
            filler_matches = detect_fillers(transcript)
            filler_count = len(filler_matches)
            filler_rate  = round(filler_count / max(duration_m, 0.01), 2)

            # 5. Score
            scores = compute_scores(wpm, pause_rate, filler_rate)

            result = {
                "fluency_score": scores["fluency"],
                "wpm": wpm, "pause_count": pause_count, "pause_rate": pause_rate,
                "filler_count": filler_count, "filler_rate": filler_rate,
                "duration_s": duration_s, "transcript": transcript,
                "filler_matches": filler_matches, "demo": False,
                "wpm_score": scores["wpm_score"],
                "pause_score": scores["pause_score"],
                "filler_score": scores["filler_score"],
                "pauses": pauses, "y": y, "sr": sr,
            }
            add_session(result, benchmark)
            status.update(label="✅ Analysis complete!", state="complete")

        except ImportError as e:
            status.update(label="❌ Missing package", state="error")
            st.error(f"Missing dependency: **{e}**  \nRun `pip install -r requirements.txt`")
            st.stop()
        except Exception as e:
            status.update(label="❌ Analysis failed", state="error")
            st.error(str(e))
            st.stop()

else:
    # ── Empty state ──
    st.markdown("""
    <div style='background:#162030;border:1px solid rgba(0,212,170,.14);
                border-radius:14px;padding:3rem;text-align:center;margin:1rem 0'>
      <div style='font-size:3.5rem;margin-bottom:1rem'>🎙️</div>
      <h3 style='color:#F0F4F8;margin-bottom:.5rem;font-family:Space Mono,monospace;font-size:1.1rem'>
        Three ways to start
      </h3>
      <p style='color:#8892A4;font-size:.88rem;max-width:480px;margin:.8rem auto 0'>
        <b style='color:#00D4AA'>Record</b> live via microphone ·
        <b style='color:#00D4AA'>Upload</b> a WAV or MP3 ·
        <b style='color:#00D4AA'>Demo</b> to explore the dashboard instantly
      </p>
      <p style='color:rgba(136,146,164,.6);font-size:.78rem;margin-top:1.5rem'>
        No API key required · Fully free by default
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════════════════════════════════════
# RESULTS DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
r  = result
sc = r["fluency_score"]
bm = BENCHMARKS[benchmark]

# ── SECTION 1: Score + Feature Metrics ───────────────────────────────────────
st.markdown("## Score Overview")
col_sc, col_mx = st.columns([1, 2], gap="large")

with col_sc:
    css_cls = score_css(sc)
    lbl     = score_label(sc)
    clr     = score_color(sc)
    st.markdown(
        f'<div class="score-badge {css_cls}">'
        f'{sc}<br>'
        f'<span style="font-size:.9rem;letter-spacing:.08em;opacity:.85">{lbl}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    delta = round(sc - bm["min_score"], 1)
    sign  = "+" if delta >= 0 else ""
    d_color = "#00D4AA" if delta >= 0 else "#FF6B35"
    st.markdown(
        f"<div style='text-align:center;font-family:Space Mono,monospace;"
        f"font-size:.82rem;color:{d_color};margin-top:.3rem'>"
        f"{sign}{delta} vs {benchmark}</div>",
        unsafe_allow_html=True,
    )

with col_mx:
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Words / Min",   f"{r['wpm']:.0f}",        help="Primary fluency indicator · Lennon (1990)")
    r1c2.metric("Pauses / Min",  f"{r['pause_rate']:.1f}",  help="Pauses >400ms · Tavakoli & Skehan (2005)")
    r1c3.metric("Fillers / Min", f"{r['filler_rate']:.1f}", help="Disfluency markers · Skehan (1996)")

    dur_m = int(r["duration_s"] // 60)
    dur_s = int(r["duration_s"] % 60)
    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.metric("Duration",     f"{dur_m}m {dur_s}s")
    r2c2.metric("Pause Count",  str(r["pause_count"]))
    r2c3.metric("Filler Count", str(r["filler_count"]))

st.divider()

# ── SECTION 2: Score Decomposition ───────────────────────────────────────────
st.markdown("## Score Decomposition")
dc1, dc2, dc3 = st.columns(3)

def decomp_block(col, icon, label, raw, weight, note):
    with col:
        pct   = int(raw)
        bar_c = "fb-fill" if raw >= 68 else ("fb-fill-warn" if raw >= 50 else "fb-fill-bad")
        col.markdown(
            f"<div style='font-size:.78rem;color:#8892A4;text-transform:uppercase;"
            f"letter-spacing:.06em'>{icon} {label}</div>"
            f"<div style='font-family:Space Mono,monospace;font-size:1.6rem;"
            f"color:#00D4AA'>{raw:.0f}<span style='font-size:.85rem;color:#8892A4'>/100</span>"
            f"<span style='font-size:.75rem;color:#8892A4;margin-left:.5rem'>×{weight}</span></div>"
            f'<div class="fb-wrap"><div class="{bar_c}" style="width:{pct}%"></div></div>'
            f"<div style='font-size:.78rem;color:#8892A4'>{note}</div>",
            unsafe_allow_html=True,
        )

decomp_block(dc1, "🗣", "Speaking Rate",  r["wpm_score"],    "0.40", f"{r['wpm']:.0f} WPM · target 140–160")
decomp_block(dc2, "⏸", "Pause Control",  r["pause_score"],  "0.35", f"{r['pause_count']} pauses detected")
decomp_block(dc3, "💬", "Filler Words",   r["filler_score"], "0.25", f"{r['filler_count']} fillers found")

st.divider()

# ── SECTION 3: Waveform ───────────────────────────────────────────────────────
st.markdown("## Waveform  ·  Pause Annotation")
if r.get("y") is not None and r["y"] is not None:
    try:
        from core.analyze import render_waveform
        fig = render_waveform(r["y"], r["sr"], r["pauses"])
        st.pyplot(fig, use_container_width=True)
    except Exception:
        st.info("Waveform rendering unavailable.")
else:
    st.markdown(
        '<div style="background:#162030;border:1px solid rgba(0,212,170,.15);'
        'border-radius:8px;padding:1.2rem;text-align:center;color:#8892A4;font-size:.85rem">'
        '〰️  Demo mode — waveform not available. Upload or record audio to see the annotated waveform.'
        '</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── SECTION 4: Annotated Transcript ──────────────────────────────────────────
st.markdown("## Annotated Transcript")
filler_matches = r.get("filler_matches") or detect_fillers(r["transcript"])
annotated      = annotate_transcript(r["transcript"], filler_matches)
st.markdown(
    f'<div class="transcript-box">{annotated}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "<span style='font-size:.76rem;color:#8892A4'>"
    "<span class='filler'>highlighted</span> = filler word</span>",
    unsafe_allow_html=True,
)

st.divider()

# ── SECTION 5: Coaching Feedback ─────────────────────────────────────────────
st.markdown("## Coaching Feedback")

from core.feedback import rule_based_feedback, groq_coaching

top_sorted = []
fw: dict[str, int] = {}
for m in filler_matches:
    w = m.group().lower()
    fw[w] = fw.get(w, 0) + 1
top_sorted = sorted(fw.items(), key=lambda x: -x[1])[:3]

fb = rule_based_feedback(
    wpm=r["wpm"], pause_rate=r["pause_rate"], filler_rate=r["filler_rate"],
    filler_count=r["filler_count"], filler_matches=filler_matches,
    wpm_score=r["wpm_score"], pause_score=r["pause_score"],
    filler_score=r["filler_score"], fluency_score=r["fluency_score"],
    transcript=r["transcript"], benchmark=benchmark,
)

# ── Overall banner ──
banner_col = "#00D4AA" if sc >= 68 else ("#FFD166" if sc >= 50 else "#FF6B35")
st.markdown(
    f'<div style="background:rgba(0,212,170,.07);border:1px solid rgba(0,212,170,.3);'
    f'border-radius:10px;padding:1rem 1.4rem;font-size:.95rem;color:{banner_col};margin-bottom:1rem">'
    f'📋 {fb["overall"]}</div>',
    unsafe_allow_html=True,
)

# ── Feature cards ──
fb_cols = st.columns(3)
for col, (key, title) in zip(fb_cols, [("wpm","Speaking Rate"),("pauses","Pause Control"),("fillers","Filler Words")]):
    with col:
        data = fb[key]
        st.markdown(
            f'<div class="fb-card">'
            f'<div style="font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;'
            f'color:#8892A4;margin-bottom:.4rem">{data["icon"]} {title}</div>'
            f'{data["advice"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Top fillers detail ──
if top_sorted:
    top_str = "  ·  ".join(f'<span class="filler">{w}</span> ×{n}' for w, n in top_sorted)
    st.markdown(
        f'<div style="font-size:.82rem;color:#8892A4;margin:-.2rem 0 .8rem">'
        f'Most frequent: {top_str}</div>',
        unsafe_allow_html=True,
    )

# ── Priority ──
st.markdown(
    f'<div class="fb-card" style="border-color:rgba(255,209,102,.3)">'
    f'<div style="font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;color:#8892A4;margin-bottom:.4rem">'
    f'🎯 Priority Focus</div>{fb["priority"]}</div>',
    unsafe_allow_html=True,
)

# ── Exercises ──
with st.expander("📋 Practice Drills", expanded=True):
    for ex in fb["exercises"]:
        st.markdown(
            f'<div style="padding:.35rem 0;font-size:.88rem;color:#F0F4F8">{ex}</div>',
            unsafe_allow_html=True,
        )

# ── Groq AI coaching (optional) ──
if groq_key and not r.get("demo"):
    st.markdown("---")
    st.markdown("### 🤖 AI Coaching  ·  Groq / Llama-3.3")
    with st.spinner("Generating personalised coaching…"):
        try:
            ai_advice = groq_coaching(
                transcript=r["transcript"],
                wpm=r["wpm"], pause_rate=r["pause_rate"],
                filler_rate=r["filler_rate"],
                fluency_score=r["fluency_score"],
                filler_top=top_sorted,
                api_key=groq_key,
            )
            st.markdown(
                f'<div class="fb-card fb-card-groq">{ai_advice}</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.warning(f"Groq coaching unavailable: {e}")
elif groq_key and r.get("demo"):
    st.info("🤖 AI coaching is disabled in demo mode — it needs a real transcript.")
else:
    st.markdown(
        '<div style="font-size:.8rem;color:#8892A4;margin-top:.5rem">'
        '💡 Add a free <b style="color:#F0F4F8">Groq API key</b> in the sidebar for AI-powered, '
        'transcript-specific coaching (free at <code>console.groq.com</code> — no credit card).'
        '</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── SECTION 6: Benchmark Comparison ──────────────────────────────────────────
st.markdown("## Benchmark Comparison")
bc1, bc2 = st.columns(2)

with bc1:
    for bm_name, bm_vals in BENCHMARKS.items():
        active    = "→ " if bm_name == benchmark else "   "
        passes    = sc >= bm_vals["min_score"]
        cls       = "bm-pass" if passes else "bm-fail"
        icon      = "✓" if passes else "✗"
        st.markdown(
            f"<div class='bm-row'>"
            f"<span style='color:#8892A4'>{active}{bm_name}</span>"
            f"<span class='{cls}'>{icon} ≥ {bm_vals['min_score']} required</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

with bc2:
    bm_now = BENCHMARKS[benchmark]
    lo, hi = bm_now["wpm"]
    items = [
        ("WPM target",          f"{lo}–{hi}",          r["wpm"],          f"{r['wpm']:.0f}"),
        ("Pauses/min target",   f"≤ {bm_now['pause_pm']:.1f}",  r["pause_rate"],   f"{r['pause_rate']:.1f}"),
        ("Fillers/min target",  f"≤ {bm_now['filler_pm']:.1f}", r["filler_rate"],  f"{r['filler_rate']:.1f}"),
    ]
    for label, target, actual, actual_str in items:
        if label == "WPM target":
            passes = lo <= actual <= hi + 20
        else:
            threshold = float(target.replace("≤ ",""))
            passes = actual <= threshold
        cls  = "bm-pass" if passes else "bm-fail"
        icon = "✓" if passes else "✗"
        st.markdown(
            f"<div class='bm-row'>"
            f"<span style='color:#8892A4'>{label}</span>"
            f"<span class='{cls}'>{icon} {actual_str} / {target}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ── SECTION 7: Progress Dashboard ────────────────────────────────────────────
history = get_history()
st.markdown(f"## Progress Dashboard  ·  {len(history)} session{'s' if len(history) != 1 else ''}")

if len(history) < 2:
    st.markdown(
        '<div style="color:#8892A4;font-size:.88rem;padding:1rem 0">'
        'Complete at least 2 sessions to see your progress chart.</div>',
        unsafe_allow_html=True,
    )
else:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores_hist  = [h["score"]      for h in history]
    wpms_hist    = [h["wpm"]        for h in history]
    pauses_hist  = [h["pause_rate"] for h in history]
    fillers_hist = [h["filler_rate"]for h in history]
    dates_hist   = [h.get("date", h["timestamp"][:6]) for h in history]
    xs           = range(len(history))

    fig, axes = plt.subplots(1, 4, figsize=(12, 2.6))
    fig.patch.set_facecolor("#0D1F35")

    panels = [
        (axes[0], scores_hist,  "Fluency Score", "#00D4AA", [(68,"--","#FF6B35","≥68"),(80,"-.","#FFD166","≥80")]),
        (axes[1], wpms_hist,    "WPM",           "#00D4AA", [(140,"--","#FF6B35","140"),(160,"-.","#FFD166","160")]),
        (axes[2], pauses_hist,  "Pauses/Min",    "#FFD166", [(3.0,"--","#FF6B35","≤3.0")]),
        (axes[3], fillers_hist, "Fillers/Min",   "#FF6B35", [(2.0,"--","#FF6B35","≤2.0")]),
    ]

    for ax, data, ylabel, color, refs in panels:
        ax.set_facecolor("#0a1726")
        ax.plot(xs, data, color=color, marker="o", linewidth=2, markersize=5)
        for ref_val, ls, rc, rlbl in refs:
            ax.axhline(ref_val, color=rc, linestyle=ls, linewidth=1, alpha=.6)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(dates_hist, rotation=30, ha="right", fontsize=6, color="#8892A4")
        ax.tick_params(axis="y", colors="#8892A4", labelsize=7)
        ax.set_ylabel(ylabel, color="#8892A4", fontsize=7)
        for s in ["top","right"]:
            ax.spines[s].set_visible(False)
        for s in ["bottom","left"]:
            ax.spines[s].set_edgecolor("#1E3A5F")

    plt.tight_layout(pad=0.8)
    st.pyplot(fig, use_container_width=True)

    # Recent table
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    for h in reversed(history[-6:]):
        lbl_color = score_color(h["score"])
        st.markdown(
            f"<div class='bm-row'>"
            f"<span style='color:#8892A4;font-size:.8rem'>{h['timestamp']}</span>"
            f"<span style='font-family:Space Mono,monospace;font-size:.82rem;"
            f"color:{lbl_color}'>{h['score']}  {h['label']}</span>"
            f"<span style='color:#8892A4;font-size:.8rem'>"
            f"{h['wpm']:.0f} WPM · {h['pause_rate']:.1f} P · {h['filler_rate']:.1f} F</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ── SECTION 8: Export ──────────────────────────────────────────────────────────
st.markdown("## Export")
report = {
    "metadata": {
        "tool": "Speech Fluency Analyzer · Diego Palencia Research",
        "generated": datetime.now().isoformat(),
        "benchmark": benchmark,
        "demo": r.get("demo", False),
    },
    "scores": {
        "fluency_score": r["fluency_score"],
        "label": score_label(sc),
        "wpm_score":    r["wpm_score"],
        "pause_score":  r["pause_score"],
        "filler_score": r["filler_score"],
    },
    "features": {
        "wpm": r["wpm"], "pause_count": r["pause_count"],
        "pause_rate_per_min": r["pause_rate"],
        "filler_count": r["filler_count"],
        "filler_rate_per_min": r["filler_rate"],
        "duration_seconds": round(r["duration_s"], 1),
    },
    "transcript": r["transcript"],
    "formula": "Fluency = 0.40×WPM_score + 0.35×Pause_score + 0.25×Filler_score",
    "references": [
        "Lennon, P. (1990). Investigating fluency in EFL. Language Learning, 40(3).",
        "Skehan, P. (1996). Task-based instruction framework. Applied Linguistics, 17(1).",
        "Tavakoli & Skehan (2005). Strategic planning and task performance.",
    ],
}
ex1, ex2 = st.columns(2)
with ex1:
    st.download_button("⬇️ Download Report (JSON)", data=json.dumps(report, indent=2),
                        file_name=f"fluency_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json", use_container_width=True)
with ex2:
    st.download_button("⬇️ Download Transcript (TXT)", data=r["transcript"],
                        file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain", use_container_width=True)


# ── RESEARCH EXPANDER ─────────────────────────────────────────────────────────
with st.expander("📚 Research Basis & Methodology"):
    st.markdown("""
#### Theoretical Grounding

| Feature | Source | Empirical Finding |
|---|---|---|
| **Speaking Rate (WPM)** | Lennon (1990) | Native English: 130–180 WPM; below 100 = disfluent |
| **Pause Rate** | Tavakoli & Skehan (2005) | L2 speakers pause 3–4× more; pauses >400ms disrupt comprehension |
| **Filler Words** | Skehan (1996) | High filler rate signals planning difficulty; pro target <2/min |

#### Composite Score Formula

```
wpm_score    = normalize(WPM, min=60, max=180) × 100
pause_score  = clip( (1 − pause_rate / 10) × 100, 0, 100 )
filler_score = clip( (1 − filler_rate / 10) × 100, 0, 100 )

Fluency_Score = (0.40 × wpm_score) + (0.35 × pause_score) + (0.25 × filler_score)
```

Part of: **Palencia (2025). Computational Feature Extraction for Human Performance Prediction.**  
OSF Preprints · `github.com/diegopalencia-research`
""")


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
  Speech Fluency Analyzer · Project 03 · Diego Jose Palencia Robles<br>
  github.com/diegopalencia-research · Guatemala City · 2025<br>
  <span style='opacity:.5'>Lennon (1990) · Skehan (1996) · Tavakoli & Skehan (2005)</span>
</div>
""", unsafe_allow_html=True)
