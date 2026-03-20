"""
ui/tab_trades.py
─────────────────
Tab 5 — Execution & Trade Log

Shows:
  • Per-trade P&L bar chart
  • Summary winner/loser statistics
  • Sortable full trade log table
  • Execution explanation box
"""

from __future__ import annotations
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from execution.simulator import ExecResult
from ui.utils.charts        import apply_base_layout


def render(exec_result: ExecResult) -> None:
    st.markdown(
        '<div class="pipeline-header">'
        'Step 7: Execution — Simulated MT5 Trade Log'
        '</div>',
        unsafe_allow_html=True,
    )

    trades = exec_result.trades
    stats  = exec_result.stats

    if len(trades) == 0:
        st.warning(
            "No trades generated. Try adjusting the signal threshold, "
            "Kelly fraction, or increasing the number of bars."
        )
        return

    # ── P&L chart ─────────────────────────────────────────────────────────────
    fig = _pnl_chart(trades)
    st.plotly_chart(fig, use_container_width=True)

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    wins  = trades[trades["pnl_abs"] > 0]
    loses = trades[trades["pnl_abs"] <= 0]

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Trades", stats["n_trades"])
    s2.metric("Winners", len(wins),
              f"Avg +${wins['pnl_abs'].mean():.0f}" if len(wins) else "–")
    s3.metric("Losers", len(loses),
              f"Avg −${abs(loses['pnl_abs'].mean()):.0f}" if len(loses) else "–")
    s4.metric("Profit Factor", f"{stats['profit_factor']:.2f}")

    # ── Trade log table ───────────────────────────────────────────────────────
    display_cols = [
        "entry_date", "exit_date", "direction",
        "entry_price", "exit_price", "pnl_pct", "pnl_abs",
    ]
    tdf = trades[display_cols].copy()
    tdf["pnl_pct"]     = tdf["pnl_pct"].round(3)
    tdf["pnl_abs"]     = tdf["pnl_abs"].round(2)
    tdf["entry_price"] = tdf["entry_price"].round(5)
    tdf["exit_price"]  = tdf["exit_price"].round(5)

    st.dataframe(
        tdf.sort_values("exit_date", ascending=False),
        use_container_width=True, height=300,
    )

    # ── Explanation ───────────────────────────────────────────────────────────
    st.markdown(
        '<div class="explain-box">'
        '<b>📖 Execution:</b> In live mode, approved signals are forwarded to '
        'the MT5 API. Each entry applies 0.02% slippage; round-trip commission '
        'is 0.005%. Positions are sized by the risk engine. In this simulation, '
        'trades enter at the bar after the signal fires and exit when the signal '
        'flips or disappears.'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Chart ─────────────────────────────────────────────────────────────────────

def _pnl_chart(trades: pd.DataFrame) -> go.Figure:
    colors = ["#34d399" if p > 0 else "#f87171" for p in trades["pnl_abs"]]
    fig = go.Figure(go.Bar(
        x=trades["exit_date"],
        y=trades["pnl_abs"],
        marker_color=colors, opacity=0.85,
        text=[f"{p:.1f}" for p in trades["pnl_pct"]],
        textposition="outside",
        textfont=dict(size=10),
    ))
    apply_base_layout(fig, height=290, title="Trade P&L ($)")
    fig.update_layout(margin=dict(l=40, r=10, t=44, b=30))
    return fig
