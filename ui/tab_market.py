"""
ui/tab_market.py
─────────────────
Tab 1 — Market Data & Regime Detection

Shows:
  • Candlestick + EMA overlays + volume + RSI
  • HMM regime bands shaded on price chart
  • Regime distribution bar chart
  • Latest feature values table
  • HMM transition matrix (expandable)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from alpha_model.utils.charts import (
    CHART_LAYOUT, EMA_COLORS, CANDLE_UP, CANDLE_DOWN,
    REGIME_COLOR, add_regime_bands, apply_base_layout,
)


def render(df: pd.DataFrame, regimes: np.ndarray,
           trans_mat: np.ndarray | None) -> None:

    st.markdown(
        '<div class="pipeline-header">'
        'Step 1 → Feature Engineering → Step 3: Regime Detection (HMM)'
        '</div>',
        unsafe_allow_html=True,
    )

    st.plotly_chart(_price_chart(df, regimes), use_container_width=True)

    col_explain, col_dist = st.columns([2, 1])
    with col_explain:
        st.markdown(
            '<div class="explain-box">'
            '<b>📖 Plain English:</b> Candles show OHLC price. '
            'EMA 9 (blue), 21 (amber), and 50 (purple) track short-, '
            'medium-, and long-term trend. Volume bars confirm moves. '
            'RSI measures momentum (overbought &gt; 70, oversold &lt; 30). '
            'Coloured background bands are HMM regime labels: '
            '<b style="color:#34d399">Trending</b> = clean directional move, '
            '<b style="color:#63b3ed">Ranging</b> = choppy mean-reversion, '
            '<b style="color:#f87171">Volatile</b> = elevated vol / anomalous volume.'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_dist:
        st.markdown("**Regime Distribution**")
        st.plotly_chart(_regime_dist_chart(regimes), use_container_width=True)

    # Latest feature row
    feat_cols = [
        "Close", "EMA_9", "EMA_21", "RSI_14",
        "EWMA_Vol", "Vol_Ratio", "BB_Pos", "ATR_14", "Regime",
    ]
    st.markdown("**Latest Feature Values**")
    row = df[feat_cols].tail(1).copy()
    row["EWMA_Vol"]  = (row["EWMA_Vol"] * 100).round(4)
    row["Vol_Ratio"] = row["Vol_Ratio"].round(2)
    row["BB_Pos"]    = row["BB_Pos"].round(3)
    row["RSI_14"]    = row["RSI_14"].round(1)
    st.dataframe(row, use_container_width=True)

    # HMM transition matrix
    if trans_mat is not None:
        with st.expander("HMM Transition Matrix"):
            labels = [f"State {i}" for i in range(trans_mat.shape[0])]
            tm_df  = pd.DataFrame(trans_mat, index=labels, columns=labels)
            st.dataframe(tm_df.round(4), use_container_width=True)
            st.markdown(
                '<div class="explain-box">'
                'Each cell = probability of moving from one regime to another on the '
                'next bar. High diagonal values = sticky regimes (desirable — once in '
                'a trend, you stay in it). The HMM uses log-returns, EWMA volatility, '
                'volume Z-score, and 5-bar momentum as its observation features.'
                '</div>',
                unsafe_allow_html=True,
            )


# ── Charts ────────────────────────────────────────────────────────────────────

def _price_chart(df: pd.DataFrame, regimes: np.ndarray) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.02,
    )

    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color=CANDLE_UP,
        decreasing_line_color=CANDLE_DOWN,
        name="OHLC", showlegend=False,
    ), row=1, col=1)

    for span, color in EMA_COLORS.items():
        col = f"EMA_{span}"
        if col in df.columns and span in (9, 21, 50):
            fig.add_trace(go.Scatter(
                x=df["Date"], y=df[col],
                line=dict(width=1, color=color),
                name=f"EMA {span}",
            ), row=1, col=1)

    bar_colors = [
        CANDLE_UP if c >= o else CANDLE_DOWN
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Volume"],
        marker_color=bar_colors,
        name="Volume", showlegend=False, opacity=0.7,
    ), row=2, col=1)

    if "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["RSI_14"],
            line=dict(color="#818cf8", width=1.5),
            name="RSI 14",
        ), row=3, col=1)
        for lvl, clr in [(70, "rgba(248,113,113,0.5)"),
                         (30, "rgba(52,211,153,0.5)")]:
            fig.add_hline(y=lvl,
                          line=dict(color=clr, width=1, dash="dot"),
                          row=3, col=1)

    add_regime_bands(fig, df["Date"], regimes, row=1, col=1)

    apply_base_layout(fig, height=530)
    return fig


def _regime_dist_chart(regimes: np.ndarray) -> go.Figure:
    counts = pd.Series(regimes).value_counts()
    colors = [REGIME_COLOR.get(r, "#94a3b8") for r in counts.index]
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=colors, opacity=0.85,
    ))
    apply_base_layout(fig, height=220)
    fig.update_layout(margin=dict(l=30, r=10, t=20, b=30))
    return fig
