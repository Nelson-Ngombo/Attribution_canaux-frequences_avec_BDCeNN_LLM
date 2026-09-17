"""
Non-blocking toast notification system for the dashboard.
Wraps Streamlit's st.toast with categorized presets.
"""

import streamlit as st


def success(message: str):
    """Shows a green success toast."""
    try:
        st.toast(f"[OK] {message}", icon=None)
    except Exception:
        st.success(message)


def error(message: str):
    """Shows a red error toast."""
    try:
        st.toast(f"[ERREUR] {message}", icon=None)
    except Exception:
        st.error(message)


def warning(message: str):
    """Shows a yellow warning toast."""
    try:
        st.toast(f"[ATTENTION] {message}", icon=None)
    except Exception:
        st.warning(message)


def info(message: str):
    """Shows an informational toast."""
    try:
        st.toast(f"[INFO] {message}", icon=None)
    except Exception:
        st.info(message)