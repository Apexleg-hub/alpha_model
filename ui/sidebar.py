"""
ui/sidebar.py
──────────────
Renders the Streamlit sidebar and returns a typed SidebarParams dataclass.
All widget state lives here; no st.sidebar calls elsewhere.
"""

from __future__ import annotations
import streamlit as st
from dataclasses import dataclass

from config.config import SYMBOLS, TIMEFRAMES


@dataclass
class SidebarParams:
    symbol:         str
    timeframe:      str
    n_bars:         int
    n_regimes:      int
    account_equity: float
    max_risk_pct:   float   # already divided by 100
    kelly_fraction: float
    max_dd_pct:     float   # already divided by 100
    run:            bool    # True when the "Run Pipeline" button was clicked


def render_sidebar() -> SidebarParams:
    with st.sidebar:
        st.markdown("##  Parameters")
        st.markdown("---")

        symbol    = st.selectbox("Symbol",    SYMBOLS,    index=0)
        timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=0)
        n_bars    = st.slider("Bars", 100, 500, 250, 50)
        n_regimes = st.slider("HMM States", 2, 4, 3)

        st.markdown("---")
        st.markdown("###  Risk Settings")
        account_equity  = st.number_input("Account Equity ($)", 1_00, 2_00)
        max_risk_pct    = st.slider("Max Risk / Trade (%)", 0.5, 5.0, 2.0, 0.5) / 100
        kelly_fraction  = st.slider("Kelly Fraction",       0.1, 1.0, 0.25, 0.05)
        max_dd_pct      = st.slider("Max Drawdown Limit (%)", 5, 50, 15, 5) / 100

        st.markdown("---")
        st.markdown("### Data Source")
        st.caption(" Live MT5 data required. Ensure MT5 is running.")

        run = st.button("  Run Pipeline", use_container_width=True, type="primary")

    return SidebarParams(
        symbol=symbol,
        timeframe=timeframe,
        n_bars=n_bars,
        n_regimes=n_regimes,
        account_equity=float(account_equity),
        max_risk_pct=max_risk_pct,
        kelly_fraction=kelly_fraction,
        max_dd_pct=max_dd_pct,
        run=run,
    )
