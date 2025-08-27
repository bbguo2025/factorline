import streamlit as st
import os
import json
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from utils.openai_helpers import get_openai

from styles.design_system import DesignSystem
from performance_monitor import PerformanceMonitor
from utils.streamlit_helpers import initialize_session_state

# Directory to persist simulation records
SAVE_DIR = "saved_simulations"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _to_serializable(obj: Any) -> Any:
    """Ensure all objects are JSON-serialisable (fall back to str)."""
    try:
        json.dumps(obj)  # type: ignore
        return obj
    except TypeError:
        return str(obj)

def _make_serializable(data: Any) -> Any:
    """Recursively convert complex objects in nested structures."""
    if isinstance(data, dict):
        return {k: _make_serializable(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_make_serializable(v) for v in data]
    return _to_serializable(data)

# -----------------------------------------------------------------------------
# OpenAI configuration (centralised helper)
# -----------------------------------------------------------------------------

openai = get_openai()

# -----------------------------------------------------------------------------
# OpenAI helper
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner=True)  # Cache so we only pay once per unique code snippet
def _code_to_feature_name(code: str) -> str:
    """Generate a succinct factor name (≤5 words, TitleCase) that captures the code's intent."""
    if not code:
        return "Unnamed Feature"

    # Truncate very long code snippets to keep the prompt light (~4k chars)
    max_chars = 4000
    snippet = code[:max_chars]

    try:
        # Handle both OpenAI ≥1.0 ("chat" namespace) and <1.0 (ChatCompletion) clients
        if hasattr(openai, "chat"):
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert quantitative developer. "
                            "Given a Python factor-construction snippet, craft a short, descriptive "
                            "TitleCase name (no spaces, <=5 words) that clearly conveys the core idea. "
                            "Return ONLY the name – no punctuation, no extra commentary."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"""Generate a concise feature name for this code:\n\n{snippet}\n""",
                    },
                ],
                max_tokens=12,
                temperature=0.5,
            )
        else:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert quantitative developer. "
                            "Given a Python factor-construction snippet, craft a short, descriptive "
                            "TitleCase name (no spaces, <=5 words) that clearly conveys the core idea. "
                            "Return ONLY the name – no punctuation, no extra commentary."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"""Generate a concise feature name for this code:\n\n{snippet}\n""",
                    },
                ],
                max_tokens=12,
                temperature=0.5,
            )
        name = response.choices[0].message.content.strip()
        # Guardrail: fall back if model returns something too long
        return name if len(name.split()) <= 5 else "Generated Feature"
    except Exception as exc:
        st.warning(f"OpenAI summarisation failed: {exc}")
        return "Summarisation Error"

def _save_record(code: str, settings: Dict[str, Any], summary: Dict[str, Any]) -> None:
    """Persist a single simulation run to disk as JSON."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = f"{timestamp.replace(':', '').replace(' ', '_')}.json"
    payload = {
        "timestamp": timestamp,
        "code": code,
        "settings": _make_serializable(settings),
        "summary": _make_serializable(summary),
    }
    with open(os.path.join(SAVE_DIR, filename), "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _load_records() -> List[Dict[str, Any]]:
    """Load all saved simulation records from disk."""
    records: List[Dict[str, Any]] = []
    for fname in sorted(os.listdir(SAVE_DIR), reverse=True):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(SAVE_DIR, fname), "r", encoding="utf-8") as fp:
                record = json.load(fp)
                record["_filename"] = fname  # Internal reference
                records.append(record)
        except Exception as exc:
            print(f"Failed to load {fname}: {exc}")
    return records


def _delete_record(filename: str) -> None:
    """Delete a saved record file."""
    try:
        os.remove(os.path.join(SAVE_DIR, filename))
    except Exception as exc:
        st.error(f"Failed to delete {filename}: {exc}")

# -----------------------------------------------------------------------------
# Streamlit page
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Simulation Results Repository",
        page_icon="📂",
        layout="wide",
    )

    # Apply shared UI helpers
    DesignSystem.inject_global_styles()
    DesignSystem.create_page_header(
        title="Factor Analysis Results Repository",
        description="Store simulation settings, code & performance so you don't have to rerun heavy simulations.",
        icon="📂",
    )
    PerformanceMonitor.display_system_status()
    initialize_session_state()

    # (Removed manual save section – simulations are now auto-saved in Page 2)

    # Visual separator
    st.markdown("---")
    # ---------------------------------------------------------------------
    # Section: Browse saved simulations
    # ---------------------------------------------------------------------
    st.markdown("## 📚 Stored Simulations")

    records = _load_records()
    if not records:
        st.info("No simulations saved yet.")
        return

    # Build summary DataFrame for quick overview
    overview_rows = []
    for rec in records:
        summ = rec.get("summary", {})
        settings = rec.get("settings", {})
        feature_name = _code_to_feature_name(rec.get("code", ""))

        overview_rows.append({
            "Feature": feature_name,
            "Ann. Return": summ.get("Annualized Return"),
            "Sharpe": summ.get("Sharpe Ratio"),
            "Rebalance": settings.get("rebalance_period"),
            "Timestamp": rec.get("timestamp"),
        })

    overview_df = pd.DataFrame(overview_rows)[["Feature", "Ann. Return", "Sharpe", "Rebalance", "Timestamp"]]
    st.dataframe(overview_df, use_container_width=True)

    selected_feature = st.selectbox(
        "Select a simulation to inspect:",
        overview_df["Feature"].tolist(),
        index=0,
    )

    selected_record = next(
        (r for r in records if _code_to_feature_name(r.get("code", "")) == selected_feature),
        None,
    )
    if selected_record:
        st.subheader(f"Details for {selected_feature}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Settings")
            st.json(selected_record["settings"])
        with col2:
            st.markdown("### Performance Summary")
            st.json(selected_record.get("summary", {}))

        # Display saved performance image if available
        if img_file := selected_record.get("image_file"):
            img_path = os.path.join(SAVE_DIR, img_file)
            if os.path.exists(img_path):
                st.markdown("### Performance Chart")
                st.image(img_path, use_column_width=True)

        st.markdown("### Feature Code")
        st.code(selected_record.get("code", "# No code stored"), language="python")

        if st.button("🗑️ Delete this record"):
            _delete_record(selected_record["_filename"])
            # Trigger a rerun to refresh the list (Streamlit ≥1.25 provides st.rerun)
            try:
                st.rerun()
            except AttributeError:
                # Fallback for older Streamlit versions
                st.experimental_rerun()


if __name__ == "__main__":
    main() 