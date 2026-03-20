"""
ui/tab_targets.py
-----------------
Price target view: likely buy/sell prices derived from math (ATR + signal strength).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from models.aggregator import AggResult
from ui.utils.charts import apply_base_layout


def render(df: pd.DataFrame, agg: AggResult) -> None:
    st.markdown(
        '<div class="pipeline-header">'
        'Price Targets — Math-Based Levels'
        '</div>',
        unsafe_allow_html=True,
    )

    if "ATR_14" not in df.columns or len(df) == 0:
        st.warning("ATR_14 not available. Run the pipeline to compute features.")
        return

    last = df.iloc[-1]
    close = float(last["Close"])
    atr = float(last["ATR_14"])

    signal = int(agg.signal[-1]) if len(agg.signal) else 0
    strength = float(agg.strength[-1]) if len(agg.strength) else 0.0

    entry_mult = 0.25 + 0.25 * strength
    target_mult = 1.0 + strength

    if signal == 1:
        action = "LONG"
        buy_price = close - atr * entry_mult
        sell_price = close + atr * target_mult
    elif signal == -1:
        action = "SHORT"
        sell_price = close + atr * entry_mult
        buy_price = close - atr * target_mult
    else:
        action = "FLAT"
        buy_price = close - atr * entry_mult
        sell_price = close + atr * entry_mult

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Current Close", _fmt_price(close))
    with c2:
        st.metric("ATR (14)", _fmt_price(atr))
    with c3:
        st.metric("Signal", action, f"Strength {strength:.2f}")

    st.markdown("#### Likely Prices (Math-Based)")
    b1, b2 = st.columns(2)
    with b1:
        st.metric("Likely Buy Price", _fmt_price(buy_price), _fmt_delta(buy_price - close))
    with b2:
        st.metric("Likely Sell Price", _fmt_price(sell_price), _fmt_delta(sell_price - close))

    st.caption(
        "Formula: entry buffer = ATR * (0.25 + 0.25 * strength). "
        "Target move = ATR * (1.0 + strength)."
    )

    fig = _target_chart(df, buy_price, sell_price)
    st.plotly_chart(fig, use_container_width=True)


def _target_chart(df: pd.DataFrame, buy: float, sell: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        line=dict(color="#94a3b8", width=1.3),
        name="Close",
    ))
    fig.add_hline(y=buy, line=dict(color="#34d399", width=1, dash="dot"),
                  annotation_text="Buy", annotation_position="top left")
    fig.add_hline(y=sell, line=dict(color="#f59e0b", width=1, dash="dot"),
                  annotation_text="Sell", annotation_position="top left")
    apply_base_layout(fig, height=320, title="Price Targets vs Close")
    return fig


def _fmt_price(x: float) -> str:
    if x >= 100:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:,.5f}"
    return f"{x:,.6f}"


def _fmt_delta(d: float) -> str:
    sign = "+" if d >= 0 else ""
    return f"{sign}{_fmt_price(d)}"
