"""
ui/tab_diagnostics.py
──────────────────────
Tab 6 — Diagnostics

Shows:
  • Feature correlation heatmap
  • SVM ↔ LSTM signal agreement metric
  • Log-return distribution histogram
  • Raw feature DataFrame (last 20 bars)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from alpha_model.models.svm_signal  import SVMResult
from alpha_model.models.lstm_signal import LSTMResult
from alpha_model.utils.charts       import CHART_LAYOUT, apply_base_layout


_FEAT_COLS = [
    "LogRet", "RSI_14", "EWMA_Vol", "Vol_Ratio",
    "EMA_cross", "Mom_5", "BB_Pos", "ATR_14",
]

_DISPLAY_COLS = [
    "Date", "Close", "RSI_14", "EMA_9", "EMA_21",
    "EWMA_Vol", "ATR_14", "Mom_5", "Vol_Ratio",
    "BB_Pos", "Regime", "Signal",
]


def render(df: pd.DataFrame,
           svm: SVMResult,
           lstm: LSTMResult) -> None:

    # ── Feature correlation heatmap ───────────────────────────────────────────
    st.markdown("### 🔬 Feature Correlation Heatmap")
    corr = df[_FEAT_COLS].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    )
    fig_corr.update_layout(**CHART_LAYOUT, height=420,
                           title="Feature Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Signal agreement ──────────────────────────────────────────────────────
    st.markdown("### 📊 Signal Agreement")
    mask  = (svm.signal != 0) | (lstm.signal != 0)
    if mask.sum() > 0:
        agree = ((svm.signal[mask] == lstm.signal[mask]) &
                 (svm.signal[mask] != 0)).mean() * 100
    else:
        agree = 0.0
    st.metric("SVM ↔ LSTM Agreement (on signal bars)", f"{agree:.1f}%")

    # ── Return distribution ────────────────────────────────────────────────────
    st.markdown("### 📈 Log-Return Distribution")
    rets = df["LogRet"].values * 100
    fig_ret = go.Figure()
    fig_ret.add_trace(go.Histogram(
        x=rets, nbinsx=50,
        marker_color="#60a5fa", opacity=0.75,
        name="Log Returns (%)",
    ))
    apply_base_layout(fig_ret, height=270, title="Log Return Distribution (%)")
    fig_ret.update_layout(margin=dict(l=40, r=10, t=44, b=30))
    st.plotly_chart(fig_ret, use_container_width=True)

    # ── Raw feature table ──────────────────────────────────────────────────────
    st.markdown("### 📄 Raw Feature DataFrame — Last 20 Bars")
    cols = [c for c in _DISPLAY_COLS if c in df.columns]
    st.dataframe(df[cols].tail(20).round(5), use_container_width=True)
