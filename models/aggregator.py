"""
models/aggregator.py
─────────────────────
Step 5: Regime-weighted signal aggregator.

Takes the three model signals plus regime labels and combines them into a
single consensus trade direction (+1 / 0 / -1) using regime-dependent
weights. A minimum score magnitude threshold prevents trading on weak consensus.

Output
──────
AggResult.signal    : np.ndarray[float]  — +1, 0, -1
AggResult.raw_score : np.ndarray[float]  — weighted score before thresholding
AggResult.strength  : np.ndarray[float]  — normalised [0,1] signal strength
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from config.config import AggregatorConfig, DEFAULT_CONFIG
from models.svm_signal   import SVMResult
from models.lstm_signal  import LSTMResult
from models.iso_forest   import IsoResult


@dataclass
class AggResult:
    signal:    np.ndarray
    raw_score: np.ndarray
    strength:  np.ndarray


def aggregate(svm: SVMResult,
              lstm: LSTMResult,
              iso: IsoResult,
              regimes: np.ndarray,
              cfg: AggregatorConfig | None = None) -> AggResult:
    if cfg is None:
        cfg = DEFAULT_CONFIG.aggregator

    n         = len(svm.signal)
    raw_score = np.zeros(n)

    # Pre-compute SVM per-bar confidence
    svm_conf = _svm_confidence(svm, n)

    for t in range(n):
        w  = np.array(cfg.regime_weights.get(regimes[t], [1/3, 1/3, 1/3]))
        sc = (w[0] * svm.signal[t]  * svm_conf[t] +
              w[1] * lstm.signal[t] * lstm.confidence[t] +
              w[2] * iso.signal[t]  * iso.score[t])
        raw_score[t] = sc

    # Threshold → discrete signal
    signal = np.where(raw_score >  cfg.signal_threshold,  1.0,
             np.where(raw_score < -cfg.signal_threshold, -1.0, 0.0))

    # Normalise strength to [0, 1]
    abs_score = np.abs(raw_score)
    p95       = np.quantile(abs_score, 0.95) if abs_score.max() > 0 else 1.0
    strength  = np.clip(abs_score / (p95 + 1e-9), 0.0, 1.0)

    return AggResult(signal=signal, raw_score=raw_score, strength=strength)


# ── Helper ────────────────────────────────────────────────────────────────────

def _svm_confidence(svm: SVMResult, n: int) -> np.ndarray:
    conf = np.full(n, 1.0 / max(len(svm.classes), 1))
    for t in range(n):
        s = svm.signal[t]
        if s != 0 and int(s) in svm.classes:
            idx      = svm.classes.index(int(s))
            conf[t]  = svm.proba[t, idx]
    return conf
