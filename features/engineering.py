"""
features/engineering.py
────────────────────────
Step 2 of the pipeline: derive predictive features from raw OHLCV bars.

All features are added as new columns on a copy of the input DataFrame.
No side effects on the caller's data.

Features produced
─────────────────
Momentum    : RSI_7, RSI_14, Mom_5, Mom_20, EMA_cross
Trend       : EMA_9, EMA_21, EMA_50, EMA_200
Volatility  : LogRet, EWMA_Vol, EWMA_Vol_Ann, ATR_14, BB_Width, BB_Pos
Volume      : Vol_MA20, Vol_Ratio, Vol_Z
Bands       : BB_Upper, BB_Lower
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config.config import FeatureConfig, DEFAULT_CONFIG


# ── Public API ────────────────────────────────────────────────────────────────

def add_features(df: pd.DataFrame,
                 cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Return df with all feature columns appended."""
    if cfg is None:
        cfg = DEFAULT_CONFIG.feature

    d = df.copy()
    c = d["Close"].values

    _add_rsi(d, c, cfg)
    _add_ema(d, c, cfg)
    _add_returns_and_vol(d, c, cfg)
    _add_volume(d, cfg)
    _add_atr(d, cfg)
    _add_momentum(d, c, cfg)
    _add_bollinger(d, c, cfg)

    return d.ffill().fillna(0)


# ── Private builders ──────────────────────────────────────────────────────────

def _rsi(prices: np.ndarray, period: int) -> np.ndarray:
    delta = np.diff(prices, prepend=prices[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    alpha = 1.0 / period
    ag    = pd.Series(gain).ewm(alpha=alpha, adjust=False).mean().values
    al    = pd.Series(loss).ewm(alpha=alpha, adjust=False).mean().values
    rs    = np.where(al == 0, 100.0, ag / (al + 1e-12))
    return 100 - 100 / (1 + rs)


def _add_rsi(d: pd.DataFrame, c: np.ndarray,
             cfg: FeatureConfig) -> None:
    for p in cfg.rsi_periods:
        d[f"RSI_{p}"] = _rsi(c, p)


def _add_ema(d: pd.DataFrame, c: np.ndarray,
             cfg: FeatureConfig) -> None:
    for span in cfg.ema_spans:
        d[f"EMA_{span}"] = pd.Series(c).ewm(span=span, adjust=False).mean().values
    d["EMA_cross"] = np.sign(d["EMA_9"] - d["EMA_21"])


def _add_returns_and_vol(d: pd.DataFrame, c: np.ndarray,
                         cfg: FeatureConfig) -> None:
    d["LogRet"] = np.log(d["Close"] / d["Close"].shift(1).fillna(d["Close"]))

    lam  = cfg.ewma_lambda
    sq   = d["LogRet"].values ** 2
    ewv  = np.zeros(len(sq))
    ewv[0] = sq[0]
    for i in range(1, len(sq)):
        ewv[i] = lam * ewv[i - 1] + (1 - lam) * sq[i]

    d["EWMA_Vol"]     = np.sqrt(ewv)
    d["EWMA_Vol_Ann"] = d["EWMA_Vol"] * np.sqrt(cfg.annualisation_factor)


def _add_volume(d: pd.DataFrame, cfg: FeatureConfig) -> None:
    w = cfg.vol_ma_window
    d["Vol_MA20"]  = d["Volume"].rolling(w, min_periods=1).mean()
    d["Vol_Ratio"] = d["Volume"] / (d["Vol_MA20"] + 1e-9)
    d["Vol_Z"]     = ((d["Volume"] - d["Vol_MA20"]) /
                      (d["Volume"].rolling(w, min_periods=1).std() + 1e-9))


def _add_atr(d: pd.DataFrame, cfg: FeatureConfig) -> None:
    hl = d["High"] - d["Low"]
    hc = (d["High"] - d["Close"].shift(1).fillna(d["Close"])).abs()
    lc = (d["Low"]  - d["Close"].shift(1).fillna(d["Close"])).abs()
    d["ATR_14"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(
        cfg.atr_period, min_periods=1).mean()


def _add_momentum(d: pd.DataFrame, c: np.ndarray,
                  cfg: FeatureConfig) -> None:
    for p in cfg.mom_periods:
        d[f"Mom_{p}"] = d["Close"] / d["Close"].shift(p).fillna(d["Close"]) - 1


def _add_bollinger(d: pd.DataFrame, c: np.ndarray,
                   cfg: FeatureConfig) -> None:
    w   = cfg.bb_window
    mid = pd.Series(c).rolling(w, min_periods=1).mean()
    std = pd.Series(c).rolling(w, min_periods=1).std().fillna(0)
    d["BB_Upper"] = (mid + cfg.bb_std * std).values
    d["BB_Lower"] = (mid - cfg.bb_std * std).values
    d["BB_Width"] = (d["BB_Upper"] - d["BB_Lower"]) / (mid.values + 1e-12)
    d["BB_Pos"]   = ((d["Close"].values - d["BB_Lower"]) /
                     (d["BB_Upper"] - d["BB_Lower"] + 1e-12))
