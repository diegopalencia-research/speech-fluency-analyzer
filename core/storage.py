"""
core/storage.py
───────────────
Persistent progress tracking across browser sessions.

Strategy
────────
Streamlit Cloud does not provide a persistent filesystem between deploys,
but files *do* persist across page refreshes within the same deployment.
We use a JSON file at DATA_PATH as the primary store and
st.session_state as the in-memory cache.

Users can also export/import their full history as JSON, which survives
re-deploys and lets them move data between devices.
"""

from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path

DATA_PATH = Path("data/sessions.json")


def _load_from_disk() -> list[dict]:
    if DATA_PATH.exists():
        try:
            with open(DATA_PATH) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_to_disk(sessions: list[dict]) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "w") as f:
        json.dump(sessions, f, indent=2)


def get_history() -> list[dict]:
    """Return full session history (disk + in-memory merged)."""
    import streamlit as st
    if "history" not in st.session_state:
        st.session_state["history"] = _load_from_disk()
    return st.session_state["history"]


def add_session(result: dict, benchmark: str) -> None:
    """Append a completed analysis to history and persist to disk."""
    import streamlit as st
    from core.score import score_label

    entry = {
        "id":          datetime.now().strftime("%Y%m%d_%H%M%S"),
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        "date":        datetime.now().strftime("%b %d"),
        "score":       result["fluency_score"],
        "wpm":         result["wpm"],
        "pause_rate":  result["pause_rate"],
        "filler_rate": result["filler_rate"],
        "label":       score_label(result["fluency_score"]),
        "benchmark":   benchmark,
        "duration_s":  round(result.get("duration_s", 0), 1),
    }

    history = get_history()
    history.append(entry)
    st.session_state["history"] = history
    _save_to_disk(history)


def clear_history() -> None:
    import streamlit as st
    st.session_state["history"] = []
    if DATA_PATH.exists():
        DATA_PATH.unlink()


def export_json() -> str:
    return json.dumps(get_history(), indent=2)


def import_json(raw: str) -> tuple[bool, str]:
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            return False, "Expected a JSON array."
        import streamlit as st
        st.session_state["history"] = data
        _save_to_disk(data)
        return True, f"Imported {len(data)} sessions."
    except Exception as e:
        return False, str(e)
