"""
data/loader.py
----------------
Step 1 of the pipeline: fetch OHLCV bars.

Live data only:
  1. MetaTrader 5 API (live / backtest)

All callers receive an identical DataFrame schema:
  Date | Open | High | Low | Close | Volume
"""

from __future__ import annotations
import pandas as pd
from typing import Tuple

import streamlit as st


# -- MT5 timeframe map (imported lazily to avoid hard dep) ---------------------
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


# -- Public API ----------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def load_bars(symbol: str, timeframe: str,
              n_bars: int, use_live: bool) -> Tuple[pd.DataFrame, bool]:
    """
    Returns (df, is_live).
    df columns: Date, Open, High, Low, Close, Volume
    is_live: True when data came from MT5
    """
    if not use_live:
        st.warning("Live MT5 Data is required. Attempting MT5 connection anyway.")

    df = _load_mt5(symbol, timeframe, n_bars)
    return df, True


# -- MT5 loader ----------------------------------------------------------------

def _load_mt5(symbol: str, timeframe: str, n_bars: int) -> pd.DataFrame:
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError("MT5 initialize() failed. Ensure MT5 is installed and running.")

    try:
        tf_map = _mt5_tf_map()
        rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, n_bars)
    finally:
        mt5.shutdown()

    if rates is None or len(rates) < 10:
        raise RuntimeError("MT5 returned no data for the requested symbol/timeframe.")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.rename(columns={
        "time": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "tick_volume": "Volume",
    })[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
