"""Utility module to persist and retrieve simulation run details.

This centralises logic so that both the Factor Analysis notebook (page 2)
and the Simulation Results repository (page 4) can share the same storage
mechanisms without code duplication.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from matplotlib.figure import Figure  # type: ignore

# Directory where JSON files will be stored
SAVE_DIR = "saved_simulations"
# Ensure directory exists when module is imported
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Helper – JSON serialisation safety
# -----------------------------------------------------------------------------

def _to_serialisable(obj: Any) -> Any:  # noqa: D401 – helper
    """Return *obj* if it can be JSON-dumped, else convert to ``str``."""
    try:
        json.dumps(obj)  # type: ignore[arg-type]
        return obj
    except TypeError:
        return str(obj)

def make_serialisable(data: Any) -> Any:  # noqa: D401 – helper
    """Recursively convert nested structures so that ``json.dumps`` works."""
    if isinstance(data, dict):
        return {k: make_serialisable(v) for k, v in data.items()}
    if isinstance(data, list):
        return [make_serialisable(v) for v in data]
    # Scalar / unsupported types
    return _to_serialisable(data)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def save_simulation_record(
    code: str,
    settings: Dict[str, Any],
    summary: Dict[str, Any],
    fig: Optional[Figure] = None,
) -> str:
    """Persist a simulation run to ``SAVE_DIR``.

    Parameters
    ----------
    code:
        The factor-generation code used to build the custom feature.
    settings:
        Dict returned by ``simulation_settings`` representing run parameters.
    summary:
        Performance summary (e.g. from ``PortfolioAnalyzer.summary()``).

    Returns
    -------
    str
        The filename (relative path) in which the record was stored.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # First determine base filename (without extension)
    base_name = timestamp.replace(":", "").replace(" ", "_")

    payload = {
        "timestamp": timestamp,
        "code": code,
        "settings": make_serialisable(settings),
        "summary": make_serialisable(summary),
    }

    # ------------------------------------------------------------------
    # Save figure if provided
    # ------------------------------------------------------------------
    if fig is not None:
        try:
            img_filename = f"{base_name}.png"
            fig_path = os.path.join(SAVE_DIR, img_filename)
            # Use a relatively high DPI for clarity, but modest file size
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            payload["image_file"] = img_filename
        except Exception as exc:  # pragma: no cover
            print(f"[simulation_storage] Could not save figure: {exc}")

    filepath = os.path.join(SAVE_DIR, f"{base_name}.json")
    with open(filepath, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    return filepath


def load_simulation_records() -> List[Dict[str, Any]]:
    """Return a list of stored simulation records (newest first)."""
    records: List[Dict[str, Any]] = []
    if not os.path.isdir(SAVE_DIR):
        return records

    for fname in sorted(os.listdir(SAVE_DIR), reverse=True):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(SAVE_DIR, fname), "r", encoding="utf-8") as fp:
                record = json.load(fp)
                record["_filename"] = fname  # internal reference
                records.append(record)
        except Exception as exc:  # pragma: no cover
            print(f"[simulation_storage] Failed to load '{fname}': {exc}")

    return records


def delete_simulation_record(filename: str) -> None:
    """Delete a stored record by filename (in ``SAVE_DIR``)."""
    try:
        os.remove(os.path.join(SAVE_DIR, filename))
    except FileNotFoundError:
        pass
    except Exception as exc:  # pragma: no cover
        print(f"[simulation_storage] Failed to delete '{filename}': {exc}") 