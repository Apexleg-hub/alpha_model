"""
app.py
───────
Alpha Model v1 — Streamlit entry point.

This file is intentionally thin: it wires the sidebar, pipeline, and tabs.
All business logic lives in alpha_model/; all UI logic lives in alpha_model/ui/.

Run (from the alpha_model parent directory):
    streamlit run alpha_model/app.py
"""

import streamlit as st


from config.config             import PipelineConfig, RiskConfig, RegimeConfig
from pipeline.pipeline         import run_pipeline
from ui.utils.styles    import inject_css
from ui.sidebar         import render_sidebar
from ui.header          import render_header, render_kpis
from ui            import tab_market      as _tm
from ui            import tab_signals     as _ts
from ui            import tab_aggregator  as _ta
from ui            import tab_risk        as _tr
from ui            import tab_trades      as _tt
from ui            import tab_diagnostics as _td
from ui            import tab_validation  as _tv
from ui            import tab_targets     as _tp

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alpha Model v1",
    page_icon=" Eve",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _show_pipeline_overview() -> None:
    st.markdown("### Pipeline Architecture")
    steps = [
        ("📡", "Market Data",  "MT5 OHLCV bars"),
        ("🔧", "Features",     "RSI · EMA · Vol · ATR"),
        ("🌐", "Regimes",      "HMM · 3 states"),
        ("🤖", "Signals",      "SVM · LSTM · IsoForest"),
        ("⚖️", "Aggregator",   "Regime-weighted consensus"),
        ("🛡️", "Risk Engine",  "Kelly · EWMA sizing"),
        ("🚀", "Execution",    "MT5 API"),
    ]
    cols = st.columns(len(steps))
    for col, (icon, title, sub) in zip(cols, steps):
        with col:
            st.markdown(f"**{icon} {title}**")
            st.caption(sub)


# ── Sidebar ───────────────────────────────────────────────────────────────────
params = render_sidebar()

# ── Header (always visible) ───────────────────────────────────────────────────
live_state = st.session_state.get("is_live", False)
render_header(live_state)

# ── Welcome screen ────────────────────────────────────────────────────────────
if not params.run and "result" not in st.session_state:
    st.info("Configure parameters in the sidebar and click **▶  Run Pipeline**.")
    _show_pipeline_overview()
    st.stop()

# ── Run pipeline ──────────────────────────────────────────────────────────────
if params.run:
    cfg = PipelineConfig(
        regime=RegimeConfig(n_states=params.n_regimes),
        risk=RiskConfig(
            account_equity=params.account_equity,
            max_risk_pct=params.max_risk_pct,
            kelly_fraction=params.kelly_fraction,
            max_drawdown_pct=params.max_dd_pct,
        ),
    )

    with st.spinner("Running pipeline…"):
        try:
            result = run_pipeline(
                symbol=params.symbol,
                timeframe=params.timeframe,
                n_bars=params.n_bars,
                use_live=True,
                cfg=cfg,
            )
        except RuntimeError as exc:
            st.error(f"MT5 connection failed: {exc}")
            st.stop()

    st.session_state["result"]  = result
    st.session_state["params"]  = params
    st.session_state["is_live"] = result.is_live

# ── Render results ────────────────────────────────────────────────────────────
if "result" in st.session_state:
    R  = st.session_state["result"]
    P  = st.session_state["params"]
    df = R.df

    cur_sig    = int(R.agg.signal[-1])
    cur_regime = R.regime.labels[-1]

    render_kpis(
        symbol=R.symbol, timeframe=R.timeframe,
        cur_sig=cur_sig, cur_regime=cur_regime,
        strength=R.agg.strength, regimes=R.regime.labels,
        stats=R.execution.stats,
    )

    tabs = st.tabs([
        " Market & Regimes",
        " Signal Models",
        " Aggregator",
        " Price Targets",
        " Risk & Equity",
        " Trades",
        " Diagnostics",
        " Validation",
    ])

    with tabs[0]:
        _tm.render(df, R.regime.labels, R.regime.trans_mat)

    with tabs[1]:
        _ts.render(df, R.svm, R.lstm, R.iso, R.regime.labels)

    with tabs[2]:
        _ta.render(df, R.agg, cur_regime)

    with tabs[3]:
        _tp.render(df, R.agg)

    with tabs[4]:
        _tr.render(df, R.risk, R.execution, P.kelly_fraction)

    with tabs[5]:
        _tt.render(R.execution)

    with tabs[6]:
        _td.render(df, R.svm, R.lstm)

    with tabs[7]:
        _tv.render(df, R.agg, R.svm, R.lstm, R.iso)
