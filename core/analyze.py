"""
core/analyze.py
───────────────
Acoustic feature extraction via librosa.

Features extracted
──────────────────
  pause_count       — number of silent gaps > min_pause_s
  total_pause_s     — cumulative pause duration
  articulation_rate — WPM over speech-only time (excludes pauses)
  pitch_var         — std-dev of voiced F0 (Hz), normalised to 0-100 score
  confidence        — mean Whisper segment avg_logprob → 0-100 score

Research basis
──────────────
  Pause detection:    Tavakoli & Skehan (2005)
  Articulation rate:  Kormos & Denes (2004) — speech rate vs articulation rate
  Pitch variation:    Hincks (2005) — F0 variation as fluency/engagement signal
"""

from __future__ import annotations
import tempfile
import os
import numpy as np


# ── LOAD ─────────────────────────────────────────────────────────────────────

def load_audio(audio_bytes: bytes, suffix: str = ".wav") -> tuple:
    """
    Load audio bytes with librosa at 16 kHz mono.
    Returns (y, sr, duration_seconds, tmp_path).
    Caller must delete tmp_path after use.
    """
    import librosa
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    y, sr = librosa.load(tmp_path, sr=16000, mono=True)
    return y, sr, float(len(y) / sr), tmp_path


# ── PAUSE DETECTION ───────────────────────────────────────────────────────────

def detect_pauses(
    y: np.ndarray,
    sr: int,
    silence_db: int = 30,
    min_pause_s: float = 0.4,
) -> tuple:
    """
    Identify silent gaps between non-silent speech segments.

    Returns
    -------
    pause_count   : int
    total_pause_s : float
    pauses        : list of (start_s, end_s, duration_s)
    speech_time_s : float — total duration of actual speech (not pauses)
    intervals     : np.ndarray — raw librosa split intervals
    """
    import librosa
    intervals = librosa.effects.split(y, top_db=silence_db)

    if len(intervals) == 0:
        return 0, 0.0, [], float(len(y) / sr), intervals

    pauses: list[tuple] = []
    for i in range(1, len(intervals)):
        gap_s = intervals[i - 1][1] / sr
        gap_e = intervals[i][0] / sr
        gap_d = gap_e - gap_s
        if gap_d >= min_pause_s:
            pauses.append((gap_s, gap_e, gap_d))

    total_pause_s = sum(p[2] for p in pauses)
    speech_time_s = (len(y) / sr) - total_pause_s

    return len(pauses), total_pause_s, pauses, max(speech_time_s, 0.1), intervals


# ── PITCH VARIATION ───────────────────────────────────────────────────────────

def extract_pitch_variation(y: np.ndarray, sr: int) -> tuple[float, float]:
    """
    Extract fundamental frequency (F0) using pyin algorithm.
    Returns (pitch_std_hz, pitch_score_0_100).

    Scoring rationale (Hincks, 2005):
      - Monotone speech (std < 20 Hz) signals low engagement / disfluency
      - High variation (std > 80 Hz) may indicate uncontrolled pitch
      - Optimal professional range: 30–70 Hz std dev → score 70–100
    """
    import librosa
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),   # ~65 Hz
            fmax=librosa.note_to_hz("C7"),   # ~2093 Hz
            sr=sr,
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

        if len(voiced_f0) < 10:
            return 0.0, 50.0   # insufficient voiced frames — neutral score

        std_hz = float(np.std(voiced_f0))

        # Normalise: 30–70 Hz → 70–100, outside → decreasing
        if std_hz < 15:
            score = max(0.0, std_hz / 15 * 55)          # 0–55
        elif std_hz <= 70:
            score = 55 + (std_hz - 15) / 55 * 45        # 55–100
        else:
            score = max(40.0, 100 - (std_hz - 70) * 0.8) # declining above 70

        return round(std_hz, 1), round(min(score, 100.0), 1)

    except Exception:
        return 0.0, 50.0


# ── WHISPER CONFIDENCE ────────────────────────────────────────────────────────

def extract_whisper_confidence(whisper_result: dict | None) -> float:
    """
    Derive a phoneme-level confidence proxy from Whisper's segment avg_logprob.

    avg_logprob range: [-2.5, 0]  (0 = perfect, more negative = less confident)
    We normalise to [0, 100] using clip(-2.5, 0) → linear map.

    Only available when using local Whisper (returns a dict with segments).
    Returns 50.0 (neutral) if result is None or API-mode (plain string).
    """
    if whisper_result is None or not isinstance(whisper_result, dict):
        return 50.0

    segments = whisper_result.get("segments", [])
    if not segments:
        return 50.0

    log_probs = [s.get("avg_logprob", -1.0) for s in segments]
    mean_lp   = float(np.mean(log_probs))
    # Clip to [-2.5, 0] → normalise to [0, 100]
    score = float(np.clip((mean_lp + 2.5) / 2.5 * 100, 0, 100))
    return round(score, 1)


# ── WAVEFORM ─────────────────────────────────────────────────────────────────

def render_waveform(
    y: np.ndarray,
    sr: int,
    pauses: list[tuple],
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """
    Amplitude waveform with pause regions shaded.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    duration = len(y) / sr
    t = np.linspace(0, duration, len(y))

    fig, ax = plt.subplots(figsize=(10, 2.4))
    fig.patch.set_facecolor("#0D1F35")
    ax.set_facecolor("#0a1726")

    step = max(1, len(y) // 8000)
    ax.plot(t[::step], y[::step], color="#00D4AA", linewidth=0.5, alpha=0.9)

    for ps, pe, _ in pauses:
        ax.axvspan(ps, pe, color="#FF6B35", alpha=0.22, linewidth=0)

    ax.set_xlabel("Time (s)", color="#8892A4", fontsize=8)
    ax.set_ylabel("Amplitude",  color="#8892A4", fontsize=8)
    ax.tick_params(colors="#8892A4", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E3A5F")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        handles=[
            mpatches.Patch(color="#00D4AA", label="Speech"),
            mpatches.Patch(color="#FF6B35", alpha=0.5, label="Pause > 400 ms"),
        ],
        loc="upper right",
        facecolor="#0D1F35", edgecolor="#1E3A5F",
        labelcolor="#8892A4", fontsize=7,
    )
    plt.tight_layout(pad=0.4)
    return fig
