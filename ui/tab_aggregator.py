"""
ui/tab_aggregator.py
─────────────────────
Tab 3 — Signal Aggregator

Shows:
  • Current aggregated signal badge
  • Regime-dependent weight table
  • Raw score time series with threshold lines
  • Plain-English explanation of the aggregation logic
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from alpha_model.models.aggregator   import AggResult
from alpha_model.config              import AggregatorConfig, DEFAULT_CONFIG
from alpha_model.utils.charts        import REGIME_COLOR, apply_base_layout


def render(df: pd.DataFrame,
           result: AggResult,
           cur_regime: str,
           cfg: AggregatorConfig | None = None) -> None:

    if cfg is None:
        cfg = DEFAULT_CONFIG.aggregator

    st.markdown(
        '<div class="pipeline-header">'
        'Step 5: Signal Aggregator — Regime-Weighted Consensus'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Current signal badge ──────────────────────────────────────────────────
    cur_sig = int(result.signal[-1])
    sig_text  = "▲ LONG" if cur_sig == 1 else ("▼ SHORT" if cur_sig == -1 else "— FLAT")
    badge_cls = "badge-long" if cur_sig == 1 else ("badge-short" if cur_sig == -1 else "badge-flat")
    reg_color = REGIME_COLOR.get(cur_regime, "#94a3b8")

    st.markdown(
        f'<p style="font-size:15px; margin-bottom:4px;">'
        f'Current aggregated signal: '
        f'<span class="{badge_cls}">{sig_text}</span>&nbsp;&nbsp;'
        f'in regime <b style="color:{reg_color}">{cur_regime}</b>'
        f'</p>',
        unsafe_allow_html=True,
    )

    # ── Weights table ─────────────────────────────────────────────────────────
    st.markdown("#### Regime-Dependent Model Weights")
    rows = []
    for regime, weights in cfg.regime_weights.items():
        rows.append({
            "Regime":     regime,
            "SVM":        weights[0],
            "LSTM":       weights[1],
            "IsoForest":  weights[2],
        })
    weights_df = pd.DataFrame(rows).set_index("Regime")
    st.dataframe(weights_df, use_container_width=True)

    # ── Explanation ───────────────────────────────────────────────────────────
    threshold = cfg.signal_threshold
    st.markdown(
        f'<div class="explain-box">'
        f'<b>📖 Plain English:</b> Each model votes at every bar. Votes are '
        f'scaled by the model\'s confidence and then combined using '
        f'regime-dependent weights (table above). The raw weighted score must '
        f'exceed <b>±{threshold}</b> to produce a trade signal — below that '
        f'the model stays flat. In <b>Trending</b> markets, SVM and LSTM carry '
        f'most weight. In <b>Volatile</b> markets, the Isolation Forest takes '
        f'over because volume shocks dominate price discovery. In '
        f'<b>Ranging</b> markets, weights are balanced.'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Raw score chart ───────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=result.raw_score,
        line=dict(color="#60a5fa", width=1.5),
        fill="tozeroy", fillcolor="rgba(96,165,250,0.07)",
        name="Raw Score",
    ))
    fig.add_hline(y= threshold, line=dict(color="#34d399", width=1, dash="dot"))
    fig.add_hline(y=-threshold, line=dict(color="#f87171", width=1, dash="dot"))
    apply_base_layout(
        fig, height=240,
        title=f"Aggregator Raw Score  (±{threshold} threshold lines)",
    )
    st.plotly_chart(fig, use_container_width=True)
