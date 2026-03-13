"""
execution/simulator.py
───────────────────────
Step 7: Execution layer — simulates MT5 order flow and builds the equity curve.

In live mode this module would send orders via mt5.order_send().
In backtest/demo mode it walks through bars, enters on signal, exits on flip.

Output
──────
ExecResult.equity_curve : np.ndarray[float]  — equity at every bar
ExecResult.drawdown     : np.ndarray[float]  — drawdown % at every bar
ExecResult.trades       : pd.DataFrame       — full trade log
ExecResult.stats        : dict               — summary performance statistics
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any

from alpha_model.config import ExecutionConfig, DEFAULT_CONFIG


@dataclass
class ExecResult:
    equity_curve: np.ndarray
    drawdown:     np.ndarray
    trades:       pd.DataFrame
    stats:        Dict[str, Any]


_TRADE_COLS = [
    "entry_date", "exit_date", "direction",
    "entry_price", "exit_price", "pnl_pct", "pnl_abs",
]


def simulate_execution(df: pd.DataFrame,
                       signal: np.ndarray,
                       pos_size: np.ndarray,
                       cfg: ExecutionConfig | None = None,
                       account_equity: float | None = None) -> ExecResult:
    if cfg is None:
        cfg = DEFAULT_CONFIG.execution
    if account_equity is None:
        from alpha_model.config import DEFAULT_CONFIG as DC
        account_equity = DC.risk.account_equity

    equity      = [account_equity]
    trades: list = []
    position    = 0
    entry_price = None
    entry_idx   = None

    closes = df["Close"].values
    dates  = df["Date"].values

    for t in range(1, len(df)):
        sig  = signal[t]
        size = pos_size[t]

        if position == 0 and sig != 0:
            # ── Entry ────────────────────────────────────────────────────────
            slip        = closes[t] * cfg.slippage_pct * np.sign(sig)
            entry_price = closes[t] + slip
            position    = sig
            entry_idx   = t

        elif position != 0 and (sig != position or sig == 0):
            # ── Exit ─────────────────────────────────────────────────────────
            slip     = closes[t] * cfg.slippage_pct * (-np.sign(position))
            ex_price = closes[t] + slip
            ret      = (ex_price - entry_price) / entry_price * position
            pnl_pct  = ret - cfg.commission_pct * 2
            pnl_abs  = equity[-1] * size * pnl_pct * cfg.leverage_proxy

            equity.append(equity[-1] + pnl_abs)
            trades.append({
                "entry_date":  pd.Timestamp(dates[entry_idx]),
                "exit_date":   pd.Timestamp(dates[t]),
                "direction":   "Long" if position == 1 else "Short",
                "entry_price": entry_price,
                "exit_price":  ex_price,
                "pnl_pct":     pnl_pct * 100,
                "pnl_abs":     pnl_abs,
            })
            position    = 0
            entry_price = None
        else:
            equity.append(equity[-1])

    # Pad if open position at end
    while len(equity) < len(df):
        equity.append(equity[-1])

    equity_arr = np.array(equity[: len(df)])
    peak       = np.maximum.accumulate(equity_arr)
    drawdown   = (equity_arr / peak - 1) * 100

    trades_df  = (pd.DataFrame(trades, columns=_TRADE_COLS)
                  if trades else pd.DataFrame(columns=_TRADE_COLS))

    stats = _compute_stats(equity_arr, drawdown, trades_df)

    return ExecResult(
        equity_curve=equity_arr,
        drawdown=drawdown,
        trades=trades_df,
        stats=stats,
    )


# ── Performance statistics ─────────────────────────────────────────────────────

def _compute_stats(equity: np.ndarray,
                   drawdown: np.ndarray,
                   trades: pd.DataFrame) -> Dict[str, Any]:
    rets = np.diff(equity) / equity[:-1]

    sharpe = float(
        (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252)
    ) if len(rets) > 1 else 0.0

    total_ret = float((equity[-1] / equity[0] - 1) * 100)
    max_dd    = float(drawdown.min())

    if len(trades):
        win_mask     = trades["pnl_abs"] > 0
        win_rate     = float(win_mask.mean() * 100)
        avg_win      = float(trades.loc[win_mask,  "pnl_abs"].mean()) if win_mask.any()  else 0.0
        avg_loss     = float(trades.loc[~win_mask, "pnl_abs"].mean()) if (~win_mask).any() else 0.0
        profit_factor = (
            (avg_win * win_rate / 100)
            / (-avg_loss * (1 - win_rate / 100) + 1e-9)
        ) if avg_loss != 0 else 0.0
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0.0

    return {
        "total_return":  total_ret,
        "sharpe":        sharpe,
        "max_drawdown":  max_dd,
        "win_rate":      win_rate,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "profit_factor": profit_factor,
        "n_trades":      len(trades),
        "final_equity":  float(equity[-1]),
    }
