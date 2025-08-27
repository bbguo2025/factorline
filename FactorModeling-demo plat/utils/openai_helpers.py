import os
import sys
import subprocess
import streamlit as st
from types import ModuleType

__all__ = ["get_openai"]


def _retrieve_api_key() -> str | None:
    """Return OpenAI key from Streamlit secrets or environment variables (if any)."""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY")


def _ensure_openai_installed() -> ModuleType:
    """Import openai, installing it on the fly if necessary (quietly)."""
    try:
        import openai  # type: ignore
    except ModuleNotFoundError:
        with st.spinner("Installing openai …"):
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", "openai>=1.30.1"],
                check=False,
            )
            import openai  # type: ignore  # noqa: E401
    return openai  # type: ignore


def get_openai():  # noqa: D401
    """Return the configured `openai` module and ensure a valid API key.

    • Retrieves key from `st.secrets["OPENAI_API_KEY"]` or `$OPENAI_API_KEY`.
    • If absent, prompts the user in the sidebar to enter it (session-only).
    • Stops the Streamlit script until a key is provided.
    • Ensures the `openai` package is installed (installs if missing).
    """
    openai = _ensure_openai_installed()

    api_key = _retrieve_api_key()

    with st.sidebar:
        st.header("🔑 OpenAI API Key")
        if api_key:
            st.success("API key loaded from secrets/env ✔️")
        else:
            api_key = st.text_input(
                "Enter your OpenAI API key", value="", type="password", placeholder="sk-…"
            )
            if api_key:
                st.success("API key set ✔️ (session only)")

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to proceed.")
        st.stop()

    openai.api_key = api_key  # type: ignore[attr-defined]
    return openai 