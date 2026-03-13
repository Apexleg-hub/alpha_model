"""
ui/tab_signals.py
──────────────────
Tab 2 — Signal Models (SVM · LSTM · Isolation Forest)

Shows stacked signal panels for each model plus plain-English explanations
and per-model statistics.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from alpha_model.models.svm_signal   import SVMResult
from alpha_model.models.lstm_signal  import LSTMResult
from alpha_model.models.iso_forest   import IsoResult
from alpha_model.utils.charts        import CHART_LAYOUT, apply_base_layout


def render(df: pd.DataFrame,
           svm: SVMResult,
           lstm: LSTMResult,
           iso: IsoResult,
           regimes: np.ndarray) -> None:

    st.markdown(
        '<div class="pipeline-header">'
        'Step 4: Signal Models — SVM · LSTM · Isolation Forest'
        '</div>',
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        _signals_chart(df, svm.signal, lstm.signal, iso.signal),
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### SVM Classifier")
        st.metric("Out-of-Sample Accuracy", f"{svm.accuracy * 100:.1f}%")
        st.markdown(
            '<div class="explain-box">'
            'SVM draws an RBF hyperplane to separate bullish and bearish bar '
            'environments using RSI, EMA crossover, momentum, EWMA volatility, '
            'Bollinger Band position, volume ratio, and ATR. Trained on the '
            'first 75% of bars; signals shown on the remaining 25%. '
            'Best in <b>Trending</b> regimes where linear decision boundaries hold.'
            '</div>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown("#### LSTM Network")
        long_pct  = (lstm.signal ==  1).mean() * 100
        short_pct = (lstm.signal == -1).mean() * 100
        flat_pct  = (lstm.signal ==  0).mean() * 100
        st.metric("% Long",  f"{long_pct:.0f}%")
        st.metric("% Short", f"{short_pct:.0f}%")
        st.metric("% Flat",  f"{flat_pct:.0f}%")
        st.markdown(
            '<div class="explain-box">'
            'Gated recurrent unit over a 20-bar window. Forget, input, and output '
            'gates control how much past context flows into the prediction. '
            'Outputs a raw score thresholded at ±0.15 to filter noise. '
            'Best in <b>Ranging</b> regimes where sequence context adds most value.'
            '</div>',
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown("#### Isolation Forest")
        n_anom   = iso.is_anomaly.sum()
        anom_pct = n_anom / max(len(iso.is_anomaly), 1) * 100
        st.metric("Anomalies Detected",
                  f"{n_anom}", f"{anom_pct:.1f}% of bars")
        st.markdown(
            '<div class="explain-box">'
            'Isolation Forest isolates anomalous bars by randomly partitioning '
            'the feature space — anomalies are isolated in fewer splits. '
            'Input: Volume, Vol Ratio, Vol Z-score, EWMA vol, ATR. '
            'An anomalous bar combined with a directional price move produces '
            '+1 / -1. Best in <b>Volatile</b> regimes driven by volume shocks.'
            '</div>',
            unsafe_allow_html=True,
        )


# ── Charts ────────────────────────────────────────────────────────────────────

def _signals_chart(df: pd.DataFrame,
                   svm_sig: np.ndarray,
                   lstm_sig: np.ndarray,
                   iso_sig: np.ndarray) -> go.Figure:

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.34, 0.33, 0.33],
        vertical_spacing=0.025,
        subplot_titles=[
            "SVM — Directional Classifier",
            "LSTM — Temporal Pattern",
            "Isolation Forest — Volume Anomaly",
        ],
    )

    pairs = [
        (svm_sig,  1, "#60a5fa", "#f59e0b"),
        (lstm_sig, 2, "#a78bfa", "#fb7185"),
        (iso_sig,  3, "#34d399", "#f87171"),
    ]

    for sig, row, pos_col, neg_col in pairs:
        colors = [
            pos_col if s > 0 else (neg_col if s < 0 else "#4a5568")
            for s in sig
        ]
        fig.add_trace(go.Bar(
            x=df["Date"], y=sig,
            marker_color=colors,
            showlegend=False, opacity=0.85,
        ), row=row, col=1)

    apply_base_layout(fig, height=480)
    return fig
