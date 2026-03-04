"""
core/storage.py
───────────────
Username-keyed session persistence.

Each user gets their own file: data/{username}.json
This survives page refreshes within a deployment and can be exported/imported
across deployments or devices via JSON download.

Username is stored in st.session_state["username"] and set at app startup.
"""

from __future__ import annotations
import json
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("data")


def _safe_name(username: str) -> str:
    """Sanitise username for use as a filename."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", username.strip().lower())[:40]


def _path(username: str) -> Path:
    return DATA_DIR / f"{_safe_name(username)}.json"


def _load(username: str) -> list[dict]:
    p = _path(username)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return []
    return []


def _save(username: str, sessions: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _path(username).write_text(json.dumps(sessions, indent=2))


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def get_history(username: str = "default") -> list[dict]:
    import streamlit as st
    key = f"history_{_safe_name(username)}"
    if key not in st.session_state:
        st.session_state[key] = _load(username)
    return st.session_state[key]


def add_session(result: dict, benchmark: str, username: str = "default") -> None:
    import streamlit as st
    from core.score import score_label
    key = f"history_{_safe_name(username)}"

    entry = {
        "id":               datetime.now().strftime("%Y%m%d_%H%M%S"),
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M"),
        "date":             datetime.now().strftime("%b %d"),
        "username":         username,
        "score":            result["fluency_score"],
        "wpm":              result["wpm"],
        "articulation_rate":result.get("articulation_rate", 0.0),
        "pitch_std":        result.get("pitch_std", 0.0),
        "pitch_score":      result.get("pitch_score", 50.0),
        "confidence":       result.get("confidence", 50.0),
        "pause_rate":       result["pause_rate"],
        "filler_rate":      result["filler_rate"],
        "label":            score_label(result["fluency_score"]),
        "benchmark":        benchmark,
        "duration_s":       round(result.get("duration_s", 0), 1),
    }

    history = get_history(username)
    history.append(entry)
    st.session_state[key] = history
    _save(username, history)


def clear_history(username: str = "default") -> None:
    import streamlit as st
    key = f"history_{_safe_name(username)}"
    st.session_state[key] = []
    p = _path(username)
    if p.exists():
        p.unlink()


def export_json(username: str = "default") -> str:
    return json.dumps(get_history(username), indent=2)


def import_json(raw: str, username: str = "default") -> tuple[bool, str]:
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            return False, "Expected a JSON array."
        import streamlit as st
        key = f"history_{_safe_name(username)}"
        st.session_state[key] = data
        _save(username, data)
        return True, f"Imported {len(data)} sessions for '{username}'."
    except Exception as e:
        return False, str(e)


def list_users() -> list[str]:
    """Return all usernames that have saved data."""
    if not DATA_DIR.exists():
        return []
    return [p.stem for p in DATA_DIR.glob("*.json")]
