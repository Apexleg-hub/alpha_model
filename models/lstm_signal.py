"""
models/lstm_signal.py
──────────────────────
Lightweight stateful LSTM-style signal (no TensorFlow/PyTorch dependency).

Implements a gated recurrent unit using NumPy — three learned weight vectors
simulate forget-gate, input-gate, and output-gate behaviour over a rolling
window. The output is thresholded to +1 / 0 / -1.

For environments where TF/PyTorch is available, swap _gated_rnn_signal() for
a proper Keras/PyTorch LSTM without changing any other module.

Output
──────
LSTMResult.signal     : np.ndarray[float]  — +1, 0, -1 per bar
LSTMResult.confidence : np.ndarray[float]  — [0, 1] signal magnitude
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

from config.config import LSTMConfig, DEFAULT_CONFIG


@dataclass
class LSTMResult:
    signal:     np.ndarray
    confidence: np.ndarray


def run_lstm(df: pd.DataFrame,
             cfg: LSTMConfig | None = None) -> LSTMResult:
    if cfg is None:
        cfg = DEFAULT_CONFIG.lstm

    X = df[cfg.feature_cols].values
    n, f = X.shape

    # Normalise
    mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xn = (X - mu) / sigma

    raw = _gated_rnn_signal(Xn, f, cfg)

    signal = np.sign(raw)
    signal[np.abs(raw) < cfg.flat_threshold] = 0.0

    confidence = np.clip(np.abs(raw), 0.0, 1.0)

    return LSTMResult(signal=signal, confidence=confidence)


# ── Core recurrent computation ────────────────────────────────────────────────

def _gated_rnn_signal(Xn: np.ndarray, f: int,
                      cfg: LSTMConfig) -> np.ndarray:
    """
    Stateful gated RNN over the normalised feature matrix.
    Returns raw output values (unbounded float, ≈ tanh range).
    """
    rng = np.random.default_rng(cfg.random_state)
    Wf  = rng.standard_normal(f) * 0.3   # forget-gate weights  (f,)
    Wi  = rng.standard_normal(f) * 0.3   # input-gate weights   (f,)
    Wo  = rng.standard_normal(f) * 0.3   # output weights       (f,)

    h   = np.zeros(f)   # hidden state  (f,)
    c   = np.zeros(f)   # cell state    (f,)
    out = np.zeros(len(Xn))

    for t in range(cfg.lookback, len(Xn)):
        x  = Xn[t]                                      # (f,)
        fg = _sigmoid(Wf * h + x)                       # element-wise
        ig = _sigmoid(Wi * h + x)
        c  = fg * c + ig * np.tanh(x)
        h  = np.tanh(Wo * c)
        out[t] = float(np.dot(Wo, c))

    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
