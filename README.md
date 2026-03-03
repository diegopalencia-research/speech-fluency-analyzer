# Speech Fluency Analyzer
`Audio Signal Processing · NLP · Speech Analytics · L2 Assessment`

---

## What It Does

Extracts **quantitative acoustic and linguistic features** from spoken English and computes a **composite fluency score (0–100)** grounded in L2 acquisition research — the same methodology used by TOEFL SpeakEasy and the Duolingo English Test, implemented in a free, deployable tool.

**Input:** Live microphone recording or uploaded WAV/MP3 (30–60 seconds)  
**Output:** Fluency Score · Feature breakdown · Annotated transcript · Benchmark comparison · Progress charts · Coaching feedback

---

## Key Features

| Feature | Description |
|---|---|
| 🎙️ Live recording | Browser microphone — record directly in the app |
| 📁 File upload | WAV · MP3 · M4A · OGG |
| 🆓 Free by default | Local Whisper (tiny model) — **no API key needed** |
| 📡 Optional fast transcription | OpenAI Whisper API (~5s vs ~30s local) |
| 📊 Composite score | Fluency Score 0–100 from 3 acoustic features |
| 〰️ Waveform | Amplitude plot with pause regions annotated |
| 📝 Annotated transcript | Filler words highlighted in the text |
| 🎯 Benchmark comparison | Score vs call center / professional standard |
| 📈 Progress dashboard | Session-to-session charts across 4 metrics |
| 🤖 AI coaching | Optional Groq (Llama-3.3) — **free API, no credit card** |
| 📋 Rule-based coaching | Specific, actionable advice — always free |
| ⬇️ Export | JSON report · TXT transcript · History JSON |
| 📥 Import history | Restore progress across deploys or devices |

---

## Zero-Friction Setup

The app is designed to work immediately with no configuration:

| Mode | Requirements | Transcription time |
|---|---|---|
| **Default (free)** | Nothing | ~20–30s (local Whisper tiny) |
| **Faster transcription** | OpenAI API key | ~5s |
| **AI coaching** | Groq API key (free) | +3–5s |

No API key? Click **Demo mode** and explore the full dashboard instantly.

---

## Folder Structure

```
speech-fluency-analyzer/
│
├── app.py                          ← Main Streamlit application (UI layer)
│
├── core/                           ← Business logic — imported by app.py
│   ├── __init__.py
│   ├── analyze.py                  ← librosa: load audio, detect pauses, waveform
│   ├── transcribe.py               ← Whisper local (free) + OpenAI API (optional)
│   ├── score.py                    ← Fluency formula, filler detection, labels
│   ├── feedback.py                 ← Rule-based coaching + Groq AI coaching
│   └── storage.py                  ← Session persistence (JSON file + import/export)
│
├── data/                           ← Auto-created on first run
│   └── sessions.json               ← Persistent progress history (gitignored)
│
├── .streamlit/
│   ├── config.toml                 ← Dark theme, portfolio colour palette
│   └── secrets.toml.example        ← Template — copy to secrets.toml locally
│
├── requirements.txt                ← Python dependencies
├── packages.txt                    ← System packages for Streamlit Cloud (ffmpeg)
├── README.md
└── .gitignore
```

**Why this structure?**  
`app.py` is the entry point Streamlit Cloud looks for at the repo root. All logic lives in `core/` so each concern is independently testable and the main file stays readable. `data/` is gitignored — it stores session history locally without polluting the repo.

---

## Fluency Score Formula

Based on Lennon (1990), Skehan (1996), and Tavakoli & Skehan (2005):

```python
wpm_score    = clip(normalize(WPM, min=60, max=180) × 100,  0, 100)
pause_score  = clip((1 − pause_rate / 10) × 100,            0, 100)
filler_score = clip((1 − filler_rate / 10) × 100,           0, 100)

Fluency_Score = (0.40 × wpm_score) + (0.35 × pause_score) + (0.25 × filler_score)
```

| Score | Label | Context |
|---|---|---|
| 0–49 | Developing | Significant fluency barriers |
| 50–67 | Emerging | Below call center threshold |
| 68–79 | Proficient | Call center entry target |
| 80–100 | Professional | Professional standard |

Weight rationale: WPM is the primary proxy (Lennon 1990); pause rate is the strongest secondary predictor (Tavakoli & Skehan 2005); filler rate is tertiary (Skehan 1996).

---

## Audio Processing Pipeline

```
User records / uploads audio
          ↓
librosa.load(audio, sr=16000, mono=True)
          ↓
librosa.effects.split(y, top_db=30)     ← non-silent regions
          ↓
gaps between regions > 400ms            ← pause_count, pause_rate
          ↓
Whisper transcription (local or API)    ← transcript text
          ↓
word_count / (duration_seconds / 60)   ← WPM
          ↓
regex filler pattern on transcript     ← filler_count, filler_rate
          ↓
Fluency Score = weighted formula        ← 0–100
          ↓
Render dashboard + coaching feedback
```

---

## Local Setup

```bash
# 1. Clone
git clone https://github.com/diegopalencia-research/speech-fluency-analyzer
cd speech-fluency-analyzer

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install ffmpeg (required for Whisper)
# macOS:
brew install ffmpeg
# Ubuntu/Debian:
sudo apt install ffmpeg
# Windows: https://ffmpeg.org/download.html

# 4. (Optional) Add API keys
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml — all keys are optional

# 5. Run
streamlit run app.py
```

On first run, the local Whisper `tiny` model (~75 MB) downloads automatically and is cached for future use.

---

## Streamlit Cloud Deployment

```
1. Push repo to GitHub (public or private)
2. Go to share.streamlit.io
3. New app → your repo → main file: app.py
4. (Optional) App Settings → Secrets → add:
      OPENAI_API_KEY = "sk-..."
      GROQ_API_KEY   = "gsk_..."
5. Deploy
```

`packages.txt` handles `ffmpeg` installation automatically on Streamlit Cloud.

---

## Coaching Feedback System

**Tier 1 — Rule-based (always free)**  
Analyses all three features against your chosen benchmark and produces specific advice — referencing your actual numbers, not generic tips.

**Tier 2 — Groq AI coaching (optional, free API)**  
Sends your transcript and metrics to `llama-3.3-70b-versatile` via Groq. Returns personalised, transcript-specific coaching: identifies the single highest-impact change and ends with a concrete 5-minute drill. Free tier at [console.groq.com](https://console.groq.com) — no credit card required.

---

## Research Basis

| Feature | Source | Finding |
|---|---|---|
| **WPM** | Lennon (1990) | Native English: 130–180 WPM; below 100 = disfluent |
| **Pause Rate** | Tavakoli & Skehan (2005) | L2 speakers pause 3–4× more; >400ms disrupts comprehension |
| **Filler Rate** | Skehan (1996) | High filler rate signals planning difficulty |

**Full references**

- Lennon, P. (1990). Investigating fluency in EFL: A quantitative approach. *Language Learning, 40*(3), 387–417.
- Skehan, P. (1996). A framework for the implementation of task-based instruction. *Applied Linguistics, 17*(1), 38–62.
- Tavakoli, P., & Skehan, P. (2005). Strategic planning, task structure, and performance testing. In R. Ellis (Ed.), *Planning and Task Performance in a Second Language.*

Part of: **Palencia (2025). Computational Feature Extraction for Human Performance Prediction.**  
A Multi-Domain Portfolio: Phonological · Operational · Acoustic · Chronobiological Systems.

---

**Live App:** https://speech-fluency-analyzer.streamlit.app/
&nbsp;&nbsp;·&nbsp;&nbsp;
**GitHub:** github.com/diegopalencia-speech-fluency-analyzer

## Author

**Diego José Palencia Robles**
*Data Science & NLP Projects — Applied AI & Analytics + Machine Learning*

- GitHub; @diegopalencia-research: https://github.com/diegopalencia-research
- LinkedIn: https://www.linkedin.com/in/diego-jose-palencia-robles/

---

## License

MIT License
