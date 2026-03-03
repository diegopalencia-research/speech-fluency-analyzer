"""
core/transcribe.py
──────────────────
Two transcription backends:

1. LOCAL (default, free)
   Uses openai-whisper library with the "tiny" model (~75 MB, cached after
   first download). No API key. Runs on CPU. ~10-30 s for a 60 s clip.

2. API (optional, faster)
   Uses openai Python client → Whisper API endpoint.
   Requires OPENAI_API_KEY.  ~2-5 s for same clip.

The caller chooses the backend via use_api flag.
"""

from __future__ import annotations
import os
import tempfile
import functools


# ── LOCAL WHISPER ─────────────────────────────────────────────────────────────

def _get_local_model(model_size: str = "tiny"):
    """
    Load and cache the local Whisper model.
    Decorated with functools.cache so the model loads only once per process.
    """
    import whisper  # openai-whisper package
    return whisper.load_model(model_size)


# Use a module-level dict as a simple process cache (compatible with Streamlit)
_model_cache: dict = {}


def transcribe_local(audio_path: str, model_size: str = "tiny") -> str:
    """
    Transcribe using the open-source Whisper model running locally.
    First call downloads the model (~75 MB for 'tiny').
    """
    import whisper
    if model_size not in _model_cache:
        _model_cache[model_size] = whisper.load_model(model_size)
    model = _model_cache[model_size]
    result = model.transcribe(audio_path, language="en", fp16=False)
    return result["text"].strip()


# ── OPENAI WHISPER API ────────────────────────────────────────────────────────

def transcribe_api(audio_path: str, api_key: str) -> str:
    """
    Transcribe via OpenAI Whisper API.
    Faster than local but requires a paid API key.
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="en",
            response_format="text",
        )
    return result.strip() if isinstance(result, str) else result


# ── UNIFIED ENTRY POINT ───────────────────────────────────────────────────────

def transcribe(
    audio_path: str,
    use_api: bool = False,
    api_key: str | None = None,
    model_size: str = "tiny",
) -> str:
    """
    Route to the appropriate backend.
    Falls back to local if use_api=True but api_key is missing.
    """
    if use_api and api_key:
        return transcribe_api(audio_path, api_key)
    return transcribe_local(audio_path, model_size=model_size)
