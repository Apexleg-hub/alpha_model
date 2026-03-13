"""
data/loader.py
──────────────
Step 1 of the pipeline: fetch OHLCV bars.

Priority:
  1. MetaTrader 5 API (live / backtest)
  2. Realistic synthetic multi-regime data (demo / offline)

All callers receive an identical DataFrame schema regardless of source:
  Date | Open | High | Low | Close | Volume
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple

import streamlit as st

from alpha_model.config import BASE_PRICES, TF_VOL, TF_FREQ


# ── MT5 timeframe map (imported lazily to avoid hard dep) ────────────────────
_MT5_TF_MAP: dict | None = None

def _mt5_tf_map() -> dict:
    global _MT5_TF_MAP
    if _MT5_TF_MAP is None:
        import MetaTrader5 as mt5
        _MT5_TF_MAP = {
            "H4":  mt5.TIMEFRAME_H4,
            "D1":  mt5.TIMEFRAME_D1,
            "W1":  mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1,
        }
    return _MT5_TF_MAP


# ── Public API ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def load_bars(symbol: str, timeframe: str,
              n_bars: int, use_live: bool) -> Tuple[pd.DataFrame, bool]:
    """
    Returns (df, is_live).
    df columns: Date, Open, High, Low, Close, Volume
    is_live: True when data came from MT5
    """
    if use_live:
        df = _load_mt5(symbol, timeframe, n_bars)
        if df is not None:
            return df, True

    return _load_synthetic(symbol, timeframe, n_bars), False


# ── MT5 loader ────────────────────────────────────────────────────────────────

def _load_mt5(symbol: str, timeframe: str, n_bars: int) -> pd.DataFrame | None:
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return None

        tf_map = _mt5_tf_map()
        rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, n_bars)
        mt5.shutdown()

        if rates is None or len(rates) < 10:
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df.rename(columns={
            "time": "Date", "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "tick_volume": "Volume",
        })[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()

    except Exception:
        return None


# ── Synthetic loader ──────────────────────────────────────────────────────────

def _load_synthetic(symbol: str, timeframe: str, n_bars: int) -> pd.DataFrame:
    """
    Generates OHLCV bars using a 3-state Hidden Markov-like process:
      State 0 = trending up   (low vol, positive drift)
      State 1 = trending down (low vol, negative drift)
      State 2 = volatile      (high vol, no drift)
    """
    seed = abs(hash(symbol + timeframe)) % (2 ** 31)
    rng  = np.random.default_rng(seed)

    S0    = BASE_PRICES.get(symbol, 1.1000)
    sigma = TF_VOL.get(timeframe, 0.003)

    # Transition matrix
    trans = np.array([
        [0.96, 0.03, 0.01],
        [0.04, 0.93, 0.03],
        [0.05, 0.05, 0.90],
    ])
    mu_r  = [0.0001, -0.0001, 0.0]
    sig_r = [sigma * 0.7, sigma * 0.7, sigma * 1.8]

    regimes = np.zeros(n_bars, dtype=int)
    ret     = np.zeros(n_bars)
    state   = 0
    for i in range(n_bars):
        state       = rng.choice(3, p=trans[state])
        regimes[i]  = state
        ret[i]      = rng.normal(mu_r[state], sig_r[state])

    closes = S0 * np.exp(np.cumsum(ret))
    spread = sigma * 0.4
    highs  = closes * np.exp(np.abs(rng.normal(0, spread, n_bars)))
    lows   = closes * np.exp(-np.abs(rng.normal(0, spread, n_bars)))
    opens  = np.roll(closes, 1); opens[0] = S0

    freq  = TF_FREQ.get(timeframe, "D")
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq=freq)

    vol_base = 5000 + 2000 * np.abs(rng.standard_normal(n_bars))
    volumes  = (vol_base * (1 + 2 * (regimes == 2))).astype(int)

    return pd.DataFrame({
        "Date": dates, "Open": opens, "High": highs,
        "Low": lows, "Close": closes, "Volume": volumes,
    })
