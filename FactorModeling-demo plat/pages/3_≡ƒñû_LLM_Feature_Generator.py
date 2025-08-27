# Standard libraries
import os
import re
import importlib.util
import json
from utils.openai_helpers import get_openai

# Third-party
import streamlit as st

# Attempt to import configure_page from project's config if available
try:
    from config import configure_page
except ImportError:

    def configure_page(title: str, icon: str = "💬", layout: str = "wide"):
        """Fallback page configurator when config.configure_page is unavailable."""
        st.set_page_config(page_title=title, page_icon=icon, layout=layout)


# --------------------------- Page Configuration --------------------------- #
configure_page("LLM Feature Generator", icon="💬")

st.title("🤖 LLM-Powered Feature Generator")
st.markdown("""
    Type an English-language description of the feature you want (e.g. **"Earnings Yield"**).
    The assistant will generate ready-to-paste Python code that constructs the factor in the
    same style as the existing snippets.
    """)

# --------------------------- Helper Functions ----------------------------- #
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # Project root


@st.cache_resource(show_spinner=False)
def load_feature_snippets(num_examples: int = 3):
    """Load feature snippets from JSON if available, otherwise fall back to the original Python module."""
    json_path = os.path.join(ROOT_DIR, "reference", "feature_snippets.json")
    snippets = None
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                snippets = json.load(f)
        except Exception as _:
            snippets = None  # fallback

    if snippets is None:
        snippet_path = os.path.join(ROOT_DIR, "reference",
                                    "feature_snippets.py")
        spec = importlib.util.spec_from_file_location("feature_snippets",
                                                      snippet_path)
        module = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(module)  # type: ignore
        snippets = module.SNIPPETS

    keys = list(snippets.keys())[:num_examples]
    return {k: snippets[k] for k in keys}


@st.cache_resource(show_spinner=False)
def load_available_columns():
    """Return the list of available dataset columns from JSON file if present, else parse the Python script."""
    json_path = os.path.join(ROOT_DIR, "reference", "dataset_columns.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("general_columns", []) + data.get(
                "fundamental_columns", [])
        except Exception:
            pass  # fallback to legacy parsing

    # Legacy parsing of the Python script (original behaviour)
    script_path = os.path.join(ROOT_DIR, "reference",
                               "create_dataset_subset.py")
    if not os.path.exists(script_path):
        return []

    with open(script_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    def _extract(start_token):
        capture = False
        cols = []
        for line in lines:
            if start_token in line and line.strip().startswith(start_token):
                capture = True
                continue
            if capture:
                if "]" in line:
                    break
                cols += re.findall(r"'([^']+)'", line)
        return cols

    return _extract("general_columns") + _extract("fundamental_columns")


# Cache heavy loads
EXAMPLE_SNIPPETS = load_feature_snippets()
AVAILABLE_COLUMNS = load_available_columns()

# --------------------------- OpenAI Configuration ------------------------- #

openai = get_openai()


# --------------------------- Helper Function Names ------------------------- #
@st.cache_resource(show_spinner=False)
def load_operation_functions():
    """Return a list of helper function names defined in operations.py."""
    op_path = os.path.join(ROOT_DIR, "operations.py")
    if not os.path.exists(op_path):
        return []
    with open(op_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Match function definitions like: def function_name(
    return sorted(set(re.findall(r"^def (\w+)\(", text, flags=re.MULTILINE)))


HELPER_FUNCS = load_operation_functions()

# --------------------------- Prompt Construction -------------------------- #


def build_system_prompt(raw_user_input: str) -> str:
    """Create the full few-shot prompt sent to the LLM."""

    examples_block = "\n\n".join([
        f"--- Example: {name.replace('_', ' ').title()} ---\n{code}"
        for name, code in EXAMPLE_SNIPPETS.items()
    ])

    cols_formatted = ", ".join(
        AVAILABLE_COLUMNS
    ) if AVAILABLE_COLUMNS else "(column list unavailable)"
    ops_formatted = ", ".join(
        HELPER_FUNCS
    ) if HELPER_FUNCS else "ts_backfill, group_rank_normalized, group_neutralize, cs_winsor, ts_sum, ts_delay"

    prompt = f"""
You are a financial factor-engineering assistant. Given a user description, output **only** a
Python code block that constructs a new factor called `custom_feature` using **only** the helper
functions available (listed below). Do not add explanations or markdown fences—output raw code.

Helper functions you can use: {ops_formatted}

Here are some examples:\n\n{examples_block}

Available columns in the dataset: {cols_formatted}

Assume the following variables are predefined:
    df : pandas DataFrame containing the columns above
    subindustry : pandas Series mapping each row to its sub-industry

Return only the factor-construction code. Do **not** wrap the output in markdown fences.
"""
    return prompt.strip()


# --------------------------- Chat Interface ------------------------------- #
user_input = st.text_input(
    "🔎 Feature prompt", placeholder="Earnings Yield, Net Debt to EBITDA, etc.")
submit = st.button("Generate Feature Code", type="primary")

if submit and user_input:
    with st.spinner("Generating factor code..."):
        try:
            messages = [
                {
                    "role": "system",
                    "content": build_system_prompt(user_input)
                },
                {
                    "role": "user",
                    "content": user_input.strip()
                },
            ]
            # Support both OpenAI ≥1.0 and legacy <1.0 clients
            if hasattr(openai, "chat"):
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=600,
                )
            else:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=600,
                )
            generated_code = response.choices[0].message.content.strip()
            st.success("✅ Feature code generated!")
            st.code(generated_code, language="python")
        except Exception as e:
            st.error(f"Error while calling OpenAI: {e}")
elif submit and not user_input:
    st.warning("Please enter a prompt to generate the feature.")

# --------------------------- Sample Feature Snippets -------------------- #
if load_feature_snippets(num_examples=1000):
    st.markdown("---")
    with st.expander("📚 Browse example feature snippets", expanded=False):
        all_snippets = load_feature_snippets(num_examples=1000)
        snippet_names = list(all_snippets.keys())
        selected_snippet = st.selectbox("Select a sample feature",
                                        snippet_names)
        st.code(all_snippets[selected_snippet], language="python")
