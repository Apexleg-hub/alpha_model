"""
models/regime.py
─────────────────
Step 3 of the pipeline: Hidden Markov Model market regime detection.

Outputs
───────
regime_labels : np.ndarray[str]  — "Trending" | "Ranging" | "Volatile"
regime_states : np.ndarray[int]  — raw HMM state indices
trans_mat     : np.ndarray       — HMM transition probability matrix
model         : GaussianHMM      — fitted model (for inspection)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

from config.config import RegimeConfig, DEFAULT_CONFIG


@dataclass
class RegimeResult:
    labels:    np.ndarray      # string labels per bar
    states:    np.ndarray      # integer state per bar
    trans_mat: np.ndarray      # (n_states × n_states) transition matrix
    model:     object          # fitted GaussianHMM


def detect_regimes(df: pd.DataFrame,
                   cfg: RegimeConfig | None = None) -> RegimeResult:
    """Fit a Gaussian HMM and map states to interpretable regime labels."""
    if cfg is None:
        cfg = DEFAULT_CONFIG.regime

    from hmmlearn.hmm import GaussianHMM

    X = _build_feature_matrix(df)
    X = np.nan_to_num(X)

    # Z-score normalise
    mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xn = (X - mu) / sigma

    model = GaussianHMM(
        n_components=cfg.n_states,
        covariance_type=cfg.covariance_type,
        n_iter=cfg.n_iter,
        random_state=cfg.random_state,
    )
    model.fit(Xn)
    states = model.predict(Xn)

    labels = _assign_labels(df, states, cfg.n_states)

    return RegimeResult(
        labels=labels,
        states=states,
        trans_mat=model.transmat_,
        model=model,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Select the four features used for regime discrimination."""
    return np.column_stack([
        df["LogRet"].values,
        df["EWMA_Vol"].values,
        df["Vol_Z"].values,
        df["Mom_5"].values,
    ])


def _assign_labels(df: pd.DataFrame,
                   states: np.ndarray,
                   n_states: int) -> np.ndarray:
    """
    Assign human-readable labels by ordering states on realised volatility:
      lowest vol  → "Trending"
      middle vol  → "Ranging"
      highest vol → "Volatile"
    """
    vol = df["EWMA_Vol"].values
    regime_vol = [vol[states == s].mean() if (states == s).any() else 0.0
                  for s in range(n_states)]
    order = np.argsort(regime_vol)  # ascending: index 0 = lowest vol

    if n_states == 2:
        label_map = {order[0]: "Trending", order[1]: "Volatile"}
    elif n_states == 3:
        label_map = {order[0]: "Trending", order[1]: "Ranging", order[2]: "Volatile"}
    else:
        # 4+ states: first=Trending, last=Volatile, middle=Ranging
        label_map = {}
        label_map[order[0]]  = "Trending"
        label_map[order[-1]] = "Volatile"
        for i in order[1:-1]:
            label_map[i] = "Ranging"

    return np.array([label_map[s] for s in states])
