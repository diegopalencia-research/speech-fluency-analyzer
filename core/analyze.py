"""
core/analyze.py
───────────────
Acoustic feature extraction using librosa.
Detects pauses, computes speech intervals, renders waveform.

Research basis:
  Tavakoli & Skehan (2005) — pause threshold 400ms, L2 pause frequency
"""

from __future__ import annotations
import io
import tempfile
import os
import numpy as np


# ── LOAD ─────────────────────────────────────────────────────────────────────

def load_audio(audio_bytes: bytes, suffix: str = ".wav") -> tuple:
    """
    Load audio bytes with librosa at 16kHz mono.
    Returns (y, sr, duration_seconds, tmp_path).
    Caller is responsible for deleting tmp_path.
    """
    import librosa
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    y, sr = librosa.load(tmp_path, sr=16000, mono=True)
    duration_s = len(y) / sr
    return y, sr, duration_s, tmp_path


# ── PAUSE DETECTION ───────────────────────────────────────────────────────────

def detect_pauses(
    y: np.ndarray,
    sr: int,
    silence_db: int = 30,
    min_pause_s: float = 0.4,
) -> tuple[int, float, list[tuple], np.ndarray]:
    """
    Detect silent pauses using librosa.effects.split.

    Returns:
        pause_count      — number of pauses >= min_pause_s
        total_pause_s    — cumulative pause duration
        pauses           — list of (start_s, end_s, duration_s) tuples
        intervals        — raw non-silent frame intervals from librosa
    """
    import librosa
    intervals = librosa.effects.split(y, top_db=silence_db)

    if len(intervals) == 0:
        return 0, 0.0, [], intervals

    pauses: list[tuple] = []
    for i in range(1, len(intervals)):
        gap_start = intervals[i - 1][1] / sr
        gap_end   = intervals[i][0] / sr
        gap_dur   = gap_end - gap_start
        if gap_dur >= min_pause_s:
            pauses.append((gap_start, gap_end, gap_dur))

    total_pause_s = sum(p[2] for p in pauses)
    return len(pauses), total_pause_s, pauses, intervals


# ── WAVEFORM ─────────────────────────────────────────────────────────────────

def render_waveform(
    y: np.ndarray,
    sr: int,
    pauses: list[tuple],
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """
    Render amplitude waveform with orange pause regions.
    Uses portfolio navy/teal color system.
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

    # Downsample for performance
    step = max(1, len(y) // 8000)
    ax.plot(t[::step], y[::step], color="#00D4AA", linewidth=0.5, alpha=0.9)

    for ps, pe, _ in pauses:
        ax.axvspan(ps, pe, color="#FF6B35", alpha=0.22, linewidth=0)

    ax.set_xlabel("Time (s)", color="#8892A4", fontsize=8)
    ax.set_ylabel("Amplitude", color="#8892A4", fontsize=8)
    ax.tick_params(colors="#8892A4", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E3A5F")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        mpatches.Patch(color="#00D4AA", label="Speech"),
        mpatches.Patch(color="#FF6B35", alpha=0.5, label="Pause  >400 ms"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper right",
        facecolor="#0D1F35", edgecolor="#1E3A5F",
        labelcolor="#8892A4", fontsize=7,
    )
    plt.tight_layout(pad=0.4)
    return fig
