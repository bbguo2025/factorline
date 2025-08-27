"""Market & Economic Dashboard (refactored)

This page provides three polished views powered by the Financial Modeling Prep
API:
1. Major Market Indices – headline metrics + optional historical chart
2. US Treasury Yield Curve – latest curve and quick historical comparison
3. Economic Calendar – filterable by date-range, country and impact level

The code is organised for readability, re-use and graceful error handling.
"""

# ---------------------------------------------------------------------------
# Imports & Path setup
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Allow helper modules living in project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from styles.design_system import DesignSystem
from utils.streamlit_helpers import create_metric_card

# ---------------------------------------------------------------------------
# Page configuration & styles
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Market & Economic Dashboard", page_icon="📈", layout="wide")
DesignSystem.inject_global_styles()
DesignSystem.create_page_header(
    title="Market & Economic Dashboard",
    description="Key market snapshots & macro data (FinancialModelingPrep API)",
    icon="📈",
)

# ---------------------------------------------------------------------------
# Constants & helper utilities
# ---------------------------------------------------------------------------

BASE_V3 = "https://financialmodelingprep.com/api/v3"
BASE_V4 = "https://financialmodelingprep.com/api/v4"  # keep for other v4 endpoints
BASE_STABLE = "https://financialmodelingprep.com/stable"
HEADERS = {"User-Agent": "FactorModeling-Dashboard"}


def _load_fmp_key() -> str:
    """Retrieve FMP API key from secrets or environment (fallback to demo)."""
    try:
        key = st.secrets["FMP_API_KEY"]
    except Exception:
        key = os.getenv("FMP_API_KEY")
    if not key:
        st.warning("Using FMP demo key – data is limited. Add `FMP_API_KEY` to secrets for full access.")
    return key or "demo"


API_KEY = _load_fmp_key()


@st.cache_data(ttl=600)
def _fetch_json(url: str) -> list | dict | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as err:
        st.error(f"FMP request failed → {err}")
        return None


# ---------------------------------------------------------------------------
# Data fetchers (all cached)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def get_indices() -> pd.DataFrame:
    data = _fetch_json(f"{BASE_V3}/quotes/index?apikey={API_KEY}")
    return pd.DataFrame(data) if data else pd.DataFrame()


@st.cache_data(ttl=1800)
def get_historical_index(symbol: str, lookback: int = 365) -> pd.DataFrame:
    url = f"{BASE_V3}/historical-price-full/{symbol}?timeseries={lookback}&serietype=line&apikey={API_KEY}"
    raw = _fetch_json(url)
    if not raw or "historical" not in raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw["historical"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


@st.cache_data(ttl=1800)
def get_treasury() -> pd.DataFrame:
    """Fetch daily treasury curve from /stable/treasury-rates endpoint."""
    data = _fetch_json(f"{BASE_STABLE}/treasury-rates?apikey={API_KEY}")
    df = pd.DataFrame(data)
    if df.empty:
        return df
    # ensure proper ordering
    df = df.sort_values("date", ascending=False)
    return df


@st.cache_data(ttl=3600)
def get_econ_calendar(start: date, end: date) -> pd.DataFrame:
    url = f"{BASE_V3}/economic_calendar?from={start}&to={end}&apikey={API_KEY}"
    data = _fetch_json(url)
    df = pd.DataFrame(data)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def metric_grid(rows: List[dict]):
    """Render a responsive metric grid from list of dicts {title, value, delta}."""
    cols = st.columns(len(rows))
    for col, row in zip(cols, rows):
        card = create_metric_card(row["title"], row["value"], delta=row["delta"], delta_color="normal")
        col.markdown(card, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs setup
# ---------------------------------------------------------------------------

tab_idx, tab_yield, tab_sector, tab_cal = st.tabs([
    "📊 Indices",
    "💵 Yield Curve",
    "🏭 Sectors",
    "�� Econ Calendar",
])

# ---------------------------------------------------------------------------
# 1. Market Indices
# ---------------------------------------------------------------------------

with tab_idx:
    st.subheader("Major Market Indices")
    idx_df = get_indices()

    if idx_df.empty:
        st.info("No data.")
    else:
        # ---- Categorised index map ---- #
        INDEX_CATEGORIES = {
            "US": {
                "^GSPC": "S&P 500",
                "^DJI": "Dow 30",
                "^IXIC": "Nasdaq Comp.",
                "^RUT": "Russell 2K",
            },
            "Global": {
                "^FTSE": "FTSE 100",
                "^GDAXI": "DAX 40",
                "^N225": "Nikkei 225",
                "^HSI": "Hang Seng",
            },
        }

        cat_choice = st.radio("Category", list(INDEX_CATEGORIES.keys()), horizontal=True)
        focus = INDEX_CATEGORIES[cat_choice]

        rows = []
        for ticker, label in focus.items():
            row = idx_df.loc[idx_df["symbol"] == ticker]
            if row.empty:
                continue
            price = row.iloc[0]["price"]
            pct = row.iloc[0]["changesPercentage"]
            rows.append({"title": label, "value": f"{price:,.2f}", "delta": f"{pct:.2f}%"})
        metric_grid(rows)

        st.divider()

        # ---- Interactive chart ---- #
        left, right = st.columns([2, 1])

        with right:
            sel_options = [t for t in focus.keys() if t in idx_df["symbol"].values]
            selected = st.selectbox("🔍 Select index to chart", options=sel_options, format_func=lambda x: focus.get(x, x))
            lookback_days = st.slider("Look-back (days)", 30, 365, 180, step=30)

        with left:
            hist = get_historical_index(selected, lookback=lookback_days)
            if hist.empty:
                st.warning("No historical data available.")
            else:
                fig = px.line(hist, x="date", y="close", title=f"{focus.get(selected, selected)} – last {lookback_days} days", labels={"close": "Close"})
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# 2. Treasury Yield Curve
# ---------------------------------------------------------------------------

with tab_yield:
    st.subheader("US Treasury Yield Curve")
    ydf = get_treasury()

    if ydf.empty:
        st.info("No data.")
    else:
        latest = ydf.iloc[0]
        maturities = [
            ("month1", "1M"),
            ("month3", "3M"),
            ("month6", "6M"),
            ("year1", "1Y"),
            ("year2", "2Y"),
            ("year3", "3Y"),
            ("year5", "5Y"),
            ("year7", "7Y"),
            ("year10", "10Y"),
            ("year20", "20Y"),
            ("year30", "30Y"),
        ]
        y_vals = [latest.get(src) for src, _ in maturities]
        label_vals = [label for _, label in maturities]
        curve_df = pd.DataFrame({"Maturity": label_vals, "Yield": y_vals})

        base_fig = px.area(curve_df, x="Maturity", y="Yield")
        base_fig.update_traces(mode="lines+markers", marker_size=6)
        base_fig.update_layout(title=f"US Treasury Yield Curve — {latest['date']}", yaxis_title="Yield (%)", hovermode="x unified")
        st.plotly_chart(base_fig, use_container_width=True)

        # 2s10s spread metric
        try:
            spread_2s10s = latest["year10"] - latest["year2"]
            spread_card = create_metric_card("2s10s Spread", f"{spread_2s10s:.2f}%", delta="", delta_color="normal")
            st.markdown(spread_card, unsafe_allow_html=True)
        except Exception:
            pass

        # Quick compare with previous curve
        st.markdown("### Compare with previous date")
        prev_dates = ydf["date"].iloc[1:6].tolist()  # last 5 prior dates
        if prev_dates:
            pdate = st.selectbox("Previous date", prev_dates)
            prev_row = ydf[ydf["date"] == pdate].iloc[0]
            prev_vals = [prev_row.get(src) for src, _ in maturities]
            overlay = go.Figure()
            overlay.add_trace(go.Scatter(x=label_vals, y=y_vals, mode="lines+markers", name=str(latest["date"])))
            overlay.add_trace(go.Scatter(x=label_vals, y=prev_vals, mode="lines+markers", name=str(pdate)))
            overlay.update_layout(title="Yield Curve Comparison", yaxis_title="Yield (%)")
            st.plotly_chart(overlay, use_container_width=True)

        st.caption("Source: FMP Treasury Rates API")

# ---------------------------------------------------------------------------
# 3. Sector Performance
# ---------------------------------------------------------------------------


@st.cache_data(ttl=1800)
def get_sector_perf() -> pd.DataFrame:
    data = _fetch_json(f"{BASE_V3}/stock/sectors-performance?apikey={API_KEY}")
    if not data or "sectorPerformance" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["sectorPerformance"])
    # Convert change% to float
    df["changesPercentage"] = df["changesPercentage"].str.replace("%", "").astype(float)
    return df


with tab_sector:
    st.subheader("US Sector Performance – 1D")
    sec_df = get_sector_perf()
    if sec_df.empty:
        st.info("Data not available.")
    else:
        fig_sec = px.bar(sec_df, x="sector", y="changesPercentage", color="changesPercentage", color_continuous_scale="RdYlGn")
        fig_sec.update_layout(yaxis_title="% Change", xaxis_title="Sector", coloraxis_showscale=False)
        st.plotly_chart(fig_sec, use_container_width=True)
        st.dataframe(sec_df.rename(columns={"sector": "Sector", "changesPercentage": "%Chg"}), use_container_width=True)
        st.caption("Source: FMP Sector Performance API")

# ---------------------------------------------------------------------------
# 4. Economic Calendar
# ---------------------------------------------------------------------------

with tab_cal:
    st.subheader("Global Economic Calendar")

    # ---- Sidebar-like controls ---- #
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        start_d = st.date_input("From", value=date.today())
    with ctrl2:
        end_d = st.date_input("To", value=date.today() + timedelta(days=14))
    with ctrl3:
        country_filter = st.text_input("Country (e.g. US, DE, CN) – leave blank for all").upper().strip()

    if start_d > end_d:
        st.error("Start date must be before end date.")
    else:
        cal_df = get_econ_calendar(start_d, end_d)
        if cal_df.empty:
            st.info("No events in range.")
        else:
            if country_filter:
                cal_df = cal_df[cal_df["country"] == country_filter]
            # Optional impact filter if field exists
            if "impact" in cal_df.columns:
                impact_levels = sorted(cal_df["impact"].dropna().unique().tolist())
                chosen_impacts = st.multiselect("Impact", impact_levels, default=impact_levels)
                cal_df = cal_df[cal_df["impact"].isin(chosen_impacts)]

            cal_df = cal_df.sort_values("date")
            display_cols = [c for c in ["date", "event", "country", "impact", "actual", "previous"] if c in cal_df.columns]
            st.dataframe(cal_df[display_cols].rename(columns={"impact": "Imp."}), use_container_width=True)
            st.caption("Source: FMP Economic Calendar API") 