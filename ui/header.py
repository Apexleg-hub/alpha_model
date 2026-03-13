"""
ui/header.py
─────────────
Page title, live/demo status badge, and the six top-level KPI metric cards.
"""

from __future__ import annotations
import numpy as np
import streamlit as st
from typing import Any, Dict

from alpha_model.utils.charts import REGIME_COLOR


def render_header(is_live: bool) -> None:
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("# 📈 Alpha Model v1")
        st.markdown(
            "*Quantitative Signal Pipeline — MT5 · HMM · SVM · LSTM · Isolation Forest*"
        )
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        if is_live:
            st.markdown('<span class="status-live">● LIVE</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-demo">● DEMO</span>',
                        unsafe_allow_html=True)
    st.markdown("---")


def render_kpis(symbol: str, timeframe: str,
                cur_sig: int, cur_regime: str,
                strength: np.ndarray, regimes: np.ndarray,
                stats: Dict[str, Any]) -> None:
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    sig_label  = "▲ LONG" if cur_sig == 1 else ("▼ SHORT" if cur_sig == -1 else "— FLAT")
    regime_pct = f"{(regimes == cur_regime).mean() * 100:.0f}% of bars"
    reg_color  = REGIME_COLOR.get(cur_regime, "#94a3b8")

    k1.metric("Symbol",         symbol,     timeframe)
    k2.metric("Current Signal", sig_label,  f"Strength {strength[-1]:.2f}")
    k3.metric("Regime",         cur_regime, regime_pct)
    k4.metric("Total Return",   f"{stats['total_return']:+.1f}%",
              f"Sharpe {stats['sharpe']:.2f}")
    k5.metric("Win Rate",       f"{stats['win_rate']:.0f}%",
              f"{stats['n_trades']} trades")
    k6.metric("Max Drawdown",   f"{stats['max_drawdown']:.1f}%",
              f"PF {stats['profit_factor']:.2f}")

    st.markdown("---")
