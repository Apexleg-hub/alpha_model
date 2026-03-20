"""
risk/engine.py
───────────────
Step 6: Risk engine — Kelly criterion + EWMA volatility scaling.

For each bar, computes:
  1. Rolling win-rate and payoff ratio (lookback_window bars)
  2. Full Kelly fraction
  3. Fractional Kelly (safety multiplier from config)
  4. Volatility scalar (target_vol / realised_EWMA_vol)
  5. Signal-strength scalar
  6. Final position size as % of equity (capped at max_risk_pct)

Output
──────
RiskResult.win_rate   : np.ndarray — rolling win-rate [0,1]
RiskResult.avg_win    : np.ndarray — rolling average winning return
RiskResult.avg_loss   : np.ndarray — rolling average losing return (abs)
RiskResult.kelly_f    : np.ndarray — fractional Kelly fraction
RiskResult.pos_size   : np.ndarray — final position size as fraction of equity
RiskResult.risk_units : np.ndarray — dollar risk per trade
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

from config.config import RiskConfig, DEFAULT_CONFIG


@dataclass
class RiskResult:
    win_rate:   np.ndarray
    avg_win:    np.ndarray
    avg_loss:   np.ndarray
    kelly_f:    np.ndarray
    pos_size:   np.ndarray
    risk_units: np.ndarray


def compute_risk(df: pd.DataFrame,
                 signal: np.ndarray,
                 strength: np.ndarray,
                 cfg: RiskConfig | None = None) -> RiskResult:
    if cfg is None:
        cfg = DEFAULT_CONFIG.risk

    n          = len(df)
    log_ret    = df["LogRet"].values
    ewma_vol   = df["EWMA_Vol"].values

    win_rate   = np.zeros(n)
    avg_win    = np.zeros(n)
    avg_loss   = np.zeros(n)
    kelly_f    = np.zeros(n)
    pos_size   = np.zeros(n)
    risk_units = np.zeros(n)

    # Per-bar realised outcome vs signal (lagged one bar)
    outcomes = log_ret * np.roll(signal, 1)

    w = cfg.lookback_window
    for t in range(w, n):
        sl = outcomes[max(0, t - w): t]

        wins = sl[sl > 0]
        loss = sl[sl < 0]

        p  = len(wins) / (len(wins) + len(loss) + 1e-9)
        aw = wins.mean() if len(wins) else 1e-4
        al = abs(loss.mean()) if len(loss) else 1e-4

        win_rate[t] = p
        avg_win[t]  = aw
        avg_loss[t] = al

        b = aw / (al + 1e-9)
        k = (p * b - (1 - p)) / (b + 1e-9)
        fk = max(0.0, min(k * cfg.kelly_fraction, 0.5))
        kelly_f[t] = fk

        # Volatility scaling
        vol_scale = np.clip(cfg.vol_target / (ewma_vol[t] + 1e-9),
                            cfg.min_vol_scale, 1.0)

        raw_size  = fk * vol_scale * strength[t]
        capped    = min(raw_size, cfg.max_risk_pct)
        pos_size[t]   = capped * abs(signal[t])
        risk_units[t] = cfg.account_equity * capped

    return RiskResult(
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        kelly_f=kelly_f,
        pos_size=pos_size,
        risk_units=risk_units,
    )
