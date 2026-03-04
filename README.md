# Speech Fluency Analyzer
`Audio Signal Processing · NLP · Speech Analytics · L2 Assessment`

**Computational acoustic and linguistic analysis of spoken English.**  
Extracts six acoustic features and three linguistic dimensions from a speech sample, returns a composite Fluency Score calibrated against professional benchmarks, and delivers structured coaching in two tiers.

---

## What It Does

Upload or record 30–60 seconds of English speech. The analyzer:

1. Transcribes the audio using OpenAI Whisper (locally, no API key needed)
2. Extracts acoustic features via librosa
3. Detects filler words, discourse connectors, and grammar patterns in the transcript
4. Computes a **Fluency Score from 0 to 100** using a research-validated formula
5. Returns an annotated transcript, waveform, benchmark comparison, and coaching
6. Saves your progress across sessions by username
7. Generates a branded one-page **PDF report** (for coaches and HR teams)

No API key required to start. All transcription and scoring runs locally and free.

---

## Features Measured

### Acoustic Features
| Feature | Method | Research Basis |
|---|---|---|
| Speaking Rate (WPM) | Word count / total duration | Lennon (1990) |
| Articulation Rate | WPM over speech-only time | Kormos & Denes (2004) |
| Pause Frequency | Librosa silence detection | Tavakoli & Skehan (2005) |
| Pitch Variation (F0) | Librosa pyin algorithm | Hincks (2005) |
| Transcription Confidence | Whisper avg_logprob | Proxy for articulation clarity |

### Linguistic Features
| Feature | Method | Research Basis |
|---|---|---|
| Filler Word Rate | Regex detection (uh, um, like, you know…) | Skehan (1996) |
| Discourse Connectors | Classification into 6 functional types | Schmidt (1990), Celce-Murcia |
| Grammar Patterns | Rule-based L2 error detection | Long (1996) |

### Composite Score Formula
```
Fluency Score = (0.40 × WPM score) + (0.35 × Pause score) + (0.25 × Filler score)
```
Weights reflect relative effect sizes reported in Lennon (1990), Tavakoli & Skehan (2005), and Skehan (1996).

---

## Score Thresholds

| Score | Label | Context |
|---|---|---|
| 0–49 | Developing | Significant fluency barriers |
| 50–67 | Emerging | Progressing with consistent practice |
| 68–79 | Proficient | Call center entry standard |
| 80–100 | Professional | Full professional communication |

---

## Feedback System

**Tier 1 — Rule-based (always on, always free)**
- Five feature cards: Speaking Rate, Pause Control, Filler Words, Discourse Structure, Grammar
- Specific numerical advice referencing your actual scores
- Practice drills tailored to your weakest feature

**Tier 2 — Groq AI coaching (optional, free API)**

Two modes selectable in the sidebar:

- **Narrative mode** — paragraph coaching that references actual phrases from your transcript
- **Correction mode** — sentence-level errors in Finishing School protocol format:
  > *"Almost. Say it like this: \_\_\_. Please repeat after me."*

Groq is free at [console.groq.com](https://console.groq.com) — no credit card required.

---

## Setup

### Local

```bash
git clone https://github.com/diegopalencia-research/speech-fluency-analyzer
cd speech-fluency-analyzer
pip install -r requirements.txt
streamlit run app.py
```

First run downloads the Whisper `tiny` model (~75 MB). This takes 30–60 seconds once, then is cached.

### Streamlit Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Connect your fork → set main file to `app.py`
4. Deploy

To add Groq AI coaching: open the app → Manage app → Secrets → add:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```

---

## Folder Structure

```
speech-fluency-analyzer/
├── app.py                    ← Streamlit UI (entry point)
├── core/
│   ├── __init__.py           ← Empty package marker (required)
│   ├── analyze.py            ← Audio loading, pause detection, pitch, waveform
│   ├── score.py              ← Fluency formula, filler/connector/grammar detection
│   ├── feedback.py           ← Rule-based coaching + Groq AI tiers
│   ├── storage.py            ← Username-keyed session persistence
│   ├── transcribe.py         ← Whisper local + OpenAI API backends
│   └── report.py             ← PDF report generation (reportlab)
├── data/                     ← Auto-created, gitignored
│   └── {username}.json       ← Session history per user
├── .streamlit/
│   ├── config.toml           ← Dark theme
│   └── secrets.toml.example  ← API key template
├── requirements.txt
├── packages.txt              ← ffmpeg for Streamlit Cloud
├── README.md
└── .gitignore
```

---

## Requirements

```
streamlit>=1.32.0
openai-whisper>=20231117
openai>=1.14.0
groq>=0.9.0
librosa>=0.10.1
numpy>=1.24.0
matplotlib>=3.8.0
soundfile>=0.12.1
audio-recorder-streamlit>=0.0.8
reportlab>=4.0.0
scipy>=1.11.0
```

`packages.txt` (Streamlit Cloud only):
```
ffmpeg
```

---

## Optional Task Prompt

Expand the **"Set a task prompt"** panel before recording to give the AI context:

> *"Describe your morning routine using sequencing words."*  
> *"Retell the story using first, then, finally."*  
> *"Why do people follow daily routines?"*

When a task is set, the Groq coaching evaluates both fluency and task completion, and the correction mode checks whether sequencing connectors appear in the response.

---

## Research Basis

| Framework | Citation |
|---|---|
| Speaking Rate | Lennon, P. (1990). Investigating fluency in EFL. *Language Learning*, 40(3). |
| Pause Rate | Tavakoli, P. & Skehan, P. (2005). Strategic planning and task performance. |
| Filler Rate | Skehan, P. (1996). A framework for task-based instruction. *Applied Linguistics*, 17(1). |
| Articulation Rate | Kormos, J. & Denes, M. (2004). Exploring measures of fluency. *System*, 32(1). |
| Pitch Variation | Hincks, R. (2005). Measures and perceptions of liveliness. *Interspeech*. |
| Discourse Connectors | Schmidt, R. (1990). The role of consciousness. *Applied Linguistics*, 11(2). |
| Corrective Feedback | Long, M. (1996). The role of the linguistic environment. *Handbook of SLA*. |

Part of: **Palencia (2025). Computational Feature Extraction for Human Performance Prediction.**  

---

**Live App:** https://speech-fluency-analyzer.streamlit.app/
&nbsp;&nbsp;·&nbsp;&nbsp;
**GitHub:** github.com/diegopalencia-speech-fluency-analyzer

## Author

**Diego José Palencia Robles**
*Data Science & NLP Projects — Applied AI & Analytics + Machine Learning + Chronobiological Systems*

- GitHub; @diegopalencia-research: https://github.com/diegopalencia-research
- LinkedIn: https://www.linkedin.com/in/diego-jose-palencia-robles/

---

## License

MIT License
