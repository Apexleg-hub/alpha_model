"""
ui/tab_risk.py
───────────────
Tab 4 — Risk Engine & Equity Curve

Shows:
  • Top-line performance KPIs
  • Equity curve / drawdown / position-size chart
  • Rolling Kelly fraction chart
  • EWMA annualised volatility chart
  • Kelly formula explanation box
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from risk.engine         import RiskResult
from execution.simulator import ExecResult
from ui.utils.charts        import apply_base_layout, CHART_LAYOUT


def render(df: pd.DataFrame,
           risk: RiskResult,
           exec_result: ExecResult,
           kelly_fraction: float) -> None:

    st.markdown(
        '<div class="pipeline-header">'
        'Step 6: Risk Engine — Kelly Criterion + EWMA Volatility Scaling'
        '</div>',
        unsafe_allow_html=True,
    )

    stats = exec_result.stats

    # ── KPIs ──────────────────────────────────────────────────────────────────
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Final Equity",  f"${stats['final_equity']:,.0f}",
              f"{stats['total_return']:+.1f}%")
    r2.metric("Sharpe Ratio",  f"{stats['sharpe']:.2f}")
    r3.metric("Max Drawdown",  f"{stats['max_drawdown']:.1f}%")
    r4.metric("Profit Factor", f"{stats['profit_factor']:.2f}")

    # ── Main triple chart ─────────────────────────────────────────────────────
    st.plotly_chart(
        _equity_chart(exec_result.equity_curve,
                      exec_result.drawdown,
                      df, risk.pos_size),
        use_container_width=True,
    )

    col_explain, col_kelly = st.columns(2)
    with col_explain:
        st.markdown(
            f'<div class="explain-box">'
            f'<b> Kelly Criterion:</b> Given win probability <i>p</i> and '
            f'payoff ratio <i>b</i> = avg_win / avg_loss, the optimal fraction '
            f'to risk is <b>f = (p·b − (1−p)) / b</b>. Full Kelly maximises '
            f'long-run growth but causes extreme drawdowns. We use '
            f'<b>{kelly_fraction}× fractional Kelly</b> as a safety multiplier. '
            f'EWMA volatility (λ=0.94) then further scales position size down '
            f'when realised vol exceeds the 0.5% daily target — you bet smaller '
            f'when the market is noisy.'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_kelly:
        st.plotly_chart(_kelly_chart(df, risk.kelly_f), use_container_width=True)

    st.plotly_chart(_vol_chart(df), use_container_width=True)


# ── Charts ────────────────────────────────────────────────────────────────────

def _equity_chart(equity: np.ndarray, drawdown: np.ndarray,
                  df: pd.DataFrame, pos_size: np.ndarray) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.30, 0.25],
        vertical_spacing=0.03,
        subplot_titles=["Equity Curve", "Drawdown (%)", "Position Size (% Equity)"],
    )
    date = df["Date"]

    fig.add_trace(go.Scatter(
        x=date, y=equity,
        line=dict(color="#60a5fa", width=2),
        fill="tozeroy", fillcolor="rgba(96,165,250,0.07)",
        name="Equity",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=date, y=drawdown,
        line=dict(color="#f87171", width=1.5),
        fill="tozeroy", fillcolor="rgba(248,113,113,0.10)",
        name="Drawdown",
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=date, y=pos_size * 100,
        marker_color="#a78bfa", opacity=0.7,
        name="Pos Size",
    ), row=3, col=1)

    apply_base_layout(fig, height=490)
    return fig


def _kelly_chart(df: pd.DataFrame, kelly_f: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=kelly_f * 100,
        line=dict(color="#a78bfa", width=1.5),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.08)",
        name="Kelly f (%)",
    ))
    apply_base_layout(fig, height=230,
                      title="Rolling Fractional Kelly (%)")
    fig.update_layout(margin=dict(l=40, r=10, t=40, b=30))
    return fig


def _vol_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["EWMA_Vol_Ann"] * 100,
        line=dict(color="#f59e0b", width=1.5),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.07)",
        name="Ann. Vol (%)",
    ))
    apply_base_layout(fig, height=210,
                      title="EWMA Volatility — Annualised % (RiskMetrics λ=0.94)")
    fig.update_layout(margin=dict(l=40, r=10, t=40, b=30))
    return fig
