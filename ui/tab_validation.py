"""
ui/tab_validation.py
────────────────────
Prediction vs. outcome dashboard and out-of-sample validation metrics.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.aggregator import AggResult
from models.svm_signal import SVMResult
from models.lstm_signal import LSTMResult
from models.iso_forest import IsoResult
from ui.utils.charts import apply_base_layout


def render(df: pd.DataFrame,
           agg: AggResult,
           svm: SVMResult,
           lstm: LSTMResult,
           iso: IsoResult,
           oos_ratio: float = 0.25) -> None:
    st.markdown(
        '<div class="pipeline-header">'
        'Validation — Prediction vs Outcome'
        '</div>',
        unsafe_allow_html=True,
    )

    if len(df) < 30:
        st.warning("Not enough bars for validation. Increase the bar count.")
        return

    next_ret = df["LogRet"].shift(-1).fillna(0.0).values
    n = len(df)
    oos_start = max(int(n * (1.0 - oos_ratio)), 1)
    oos_end = max(n - 1, oos_start + 1)
    sl = slice(oos_start, oos_end)

    st.caption(f"Out-of-sample window: last {int(oos_ratio * 100)}% of bars")

    # ── OOS metrics table ─────────────────────────────────────────────────────
    rows = []
    for name, sig in [
        ("Aggregator", agg.signal),
        ("SVM", svm.signal),
        ("LSTM", lstm.signal),
        ("IsoForest", iso.signal),
    ]:
        metrics = _signal_metrics(sig[sl], next_ret[sl])
        rows.append({"Model": name, **metrics})

    metrics_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(metrics_df, use_container_width=True)

    # ── Prediction vs outcome chart ───────────────────────────────────────────
    st.markdown("### Prediction vs Outcome (OOS)")
    fig_pred = _prediction_outcome_chart(df.iloc[sl], agg.signal[sl], next_ret[sl])
    st.plotly_chart(fig_pred, use_container_width=True)

    # ── Confidence buckets ────────────────────────────────────────────────────
    st.markdown("### Confidence vs Outcome (OOS)")
    fig_conf = _confidence_bucket_chart(
        agg.strength[sl], agg.signal[sl], next_ret[sl]
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    # ── Walk-forward slices (approx.) ─────────────────────────────────────────
    st.markdown("### Walk-Forward Slices (Approx.)")
    fig_walk = _walk_forward_chart(df["Date"], agg.signal, next_ret)
    st.plotly_chart(fig_walk, use_container_width=True)


def _signal_metrics(signal: np.ndarray, next_ret: np.ndarray) -> dict:
    signal = np.asarray(signal)
    next_ret = np.asarray(next_ret)
    mask = signal != 0
    coverage = float(mask.mean() * 100)

    if mask.sum() == 0:
        return {
            "Accuracy": "0.0%",
            "Win Rate": "0.0%",
            "Coverage": f"{coverage:.1f}%",
            "Edge (avg %)": "0.00",
            "Profit Factor": "0.00",
        }

    target = np.sign(next_ret)
    correct = (signal[mask] == target[mask])
    accuracy = correct.mean() * 100

    signed = signal[mask] * next_ret[mask]
    win_rate = (signed > 0).mean() * 100
    edge = signed.mean() * 100

    pos = signed[signed > 0]
    neg = signed[signed < 0]
    if neg.size == 0:
        profit_factor = float("inf")
    else:
        profit_factor = float(pos.sum() / abs(neg.sum())) if pos.size else 0.0

    pf_display = "∞" if profit_factor == float("inf") else f"{profit_factor:.2f}"

    return {
        "Accuracy": f"{accuracy:.1f}%",
        "Win Rate": f"{win_rate:.1f}%",
        "Coverage": f"{coverage:.1f}%",
        "Edge (avg %)": f"{edge:.2f}",
        "Profit Factor": pf_display,
    }


def _prediction_outcome_chart(df: pd.DataFrame,
                              signal: np.ndarray,
                              next_ret: np.ndarray) -> go.Figure:
    signal = np.asarray(signal)
    next_ret = np.asarray(next_ret)
    target = np.sign(next_ret)

    mask = signal != 0
    correct = mask & (signal == target)
    wrong = mask & (signal != target)

    long = signal == 1
    short = signal == -1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        line=dict(color="#94a3b8", width=1.2),
        name="Close",
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"][correct & long],
        y=df["Close"][correct & long],
        mode="markers",
        marker=dict(symbol="triangle-up", size=9, color="#34d399"),
        name="Correct Long",
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"][correct & short],
        y=df["Close"][correct & short],
        mode="markers",
        marker=dict(symbol="triangle-down", size=9, color="#34d399"),
        name="Correct Short",
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"][wrong & long],
        y=df["Close"][wrong & long],
        mode="markers",
        marker=dict(symbol="triangle-up", size=9, color="#f87171"),
        name="Wrong Long",
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"][wrong & short],
        y=df["Close"][wrong & short],
        mode="markers",
        marker=dict(symbol="triangle-down", size=9, color="#f87171"),
        name="Wrong Short",
    ))

    apply_base_layout(fig, height=360, title="Signals vs Next-Bar Outcome")
    return fig


def _confidence_bucket_chart(strength: np.ndarray,
                              signal: np.ndarray,
                              next_ret: np.ndarray) -> go.Figure:
    bins = np.linspace(0.0, 1.0, 6)
    labels = [f"{bins[i]:.1f}–{bins[i+1]:.1f}" for i in range(len(bins) - 1)]

    hit_rates = []
    edges = []
    counts = []

    for i in range(len(bins) - 1):
        b0, b1 = bins[i], bins[i + 1]
        mask = (strength >= b0) & (strength < b1) & (signal != 0)
        counts.append(int(mask.sum()))
        if mask.sum() == 0:
            hit_rates.append(0.0)
            edges.append(0.0)
            continue
        signed = signal[mask] * next_ret[mask]
        hit_rates.append((signed > 0).mean() * 100)
        edges.append(signed.mean() * 100)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=labels, y=hit_rates, name="Hit Rate (%)",
        marker_color="#60a5fa",
        hovertemplate="Hit Rate: %{y:.1f}%<br>Count: %{customdata}",
        customdata=counts,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=labels, y=edges, name="Edge (avg %)",
        mode="lines+markers",
        line=dict(color="#34d399", width=2),
    ), secondary_y=True)

    apply_base_layout(fig, height=300, title="Confidence Buckets")
    fig.update_yaxes(title_text="Hit Rate (%)", secondary_y=False)
    fig.update_yaxes(title_text="Edge (avg %)", secondary_y=True)
    return fig


def _walk_forward_chart(dates: pd.Series,
                        signal: np.ndarray,
                        next_ret: np.ndarray,
                        n_slices: int = 4) -> go.Figure:
    n = len(signal)
    size = max(n // n_slices, 1)

    slice_labels = []
    edges = []

    for i in range(n_slices):
        start = i * size
        end = n if i == n_slices - 1 else min((i + 1) * size, n)
        if end - start <= 2:
            continue

        sl = slice(start, end - 1)
        sig = signal[sl]
        ret = next_ret[sl]
        mask = sig != 0
        if mask.sum() == 0:
            edge = 0.0
        else:
            edge = float((sig[mask] * ret[mask]).mean() * 100)

        label = f"{dates.iloc[start].date()} → {dates.iloc[end-1].date()}"
        slice_labels.append(label)
        edges.append(edge)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=slice_labels, y=edges,
        marker_color="#a78bfa",
        name="Edge (avg %)",
    ))
    apply_base_layout(fig, height=260, title="Walk-Forward Edge by Slice")
    fig.update_xaxes(tickangle=20)
    return fig
