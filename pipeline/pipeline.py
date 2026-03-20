"""
pipeline.py
────────────
Central orchestrator: runs every step in order and returns a PipelineResult.

Usage
─────
    from alpha_model.pipeline import run_pipeline, PipelineResult
    from alpha_model.config   import PipelineConfig

    result = run_pipeline(symbol="EURUSD", timeframe="D1",
                          n_bars=250, use_live=True,
                          cfg=PipelineConfig())

No Streamlit dependency here — this module is safe to call from unit tests,
notebooks, or batch scripts.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict

from config              import PipelineConfig, DEFAULT_CONFIG
from data.loader         import load_bars
from features.engineering import add_features
from models.regime        import detect_regimes,         RegimeResult
from models.svm_signal    import run_svm,                SVMResult
from models.lstm_signal   import run_lstm,               LSTMResult
from models.iso_forest    import run_isolation_forest,   IsoResult
from models.aggregator    import aggregate,              AggResult
from risk.engine          import compute_risk,           RiskResult
from execution.simulator  import simulate_execution,     ExecResult


@dataclass
class PipelineResult:
    # Inputs
    symbol:    str
    timeframe: str
    is_live:   bool

    # Data
    df:        pd.DataFrame     # OHLCV + all features + Regime + Signal columns

    # Step outputs (kept separate for UI introspection)
    regime:    RegimeResult
    svm:       SVMResult
    lstm:      LSTMResult
    iso:       IsoResult
    agg:       AggResult
    risk:      RiskResult
    execution: ExecResult


def run_pipeline(symbol:    str,
                 timeframe: str,
                 n_bars:    int,
                 use_live:  bool,
                 cfg: PipelineConfig | None = None) -> PipelineResult:
    """Execute the full pipeline end-to-end and return a PipelineResult."""

    if cfg is None:
        cfg = DEFAULT_CONFIG

    # ── 1. Market data ────────────────────────────────────────────────────────
    df_raw, is_live = load_bars(symbol, timeframe, n_bars, use_live)

    # ── 2. Feature engineering ────────────────────────────────────────────────
    df = add_features(df_raw, cfg.feature)

    # ── 3. Regime detection ───────────────────────────────────────────────────
    regime = detect_regimes(df, cfg.regime)
    df["Regime"] = regime.labels

    # ── 4. Signal models ──────────────────────────────────────────────────────
    svm  = run_svm(df, cfg.svm)
    lstm = run_lstm(df, cfg.lstm)
    iso  = run_isolation_forest(df, cfg.iso_forest)

    # ── 5. Signal aggregator ──────────────────────────────────────────────────
    agg = aggregate(svm, lstm, iso, regime.labels, cfg.aggregator)
    df["Signal"] = agg.signal

    # ── 6. Risk engine ────────────────────────────────────────────────────────
    # Apply sidebar overrides to risk config
    risk_cfg = cfg.risk
    risk     = compute_risk(df, agg.signal, agg.strength, risk_cfg)

    # ── 7. Execution ──────────────────────────────────────────────────────────
    execution = simulate_execution(
        df, agg.signal, risk.pos_size,
        cfg.execution, risk_cfg.account_equity,
    )

    return PipelineResult(
        symbol=symbol, timeframe=timeframe, is_live=is_live,
        df=df, regime=regime,
        svm=svm, lstm=lstm, iso=iso, agg=agg,
        risk=risk, execution=execution,
    )
