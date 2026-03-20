"""
models/iso_forest.py
─────────────────────
Volume anomaly detection via Isolation Forest (Step 4c).

Anomalous volume bars combined with a directional price move produce a
+1 / -1 signal. Normal bars produce 0.

Output
──────
IsoResult.signal      : np.ndarray[float]  — +1, 0, -1
IsoResult.is_anomaly  : np.ndarray[bool]
IsoResult.score       : np.ndarray[float]  — normalised anomaly score [0, 1]
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

from config.config import IsoForestConfig, DEFAULT_CONFIG


@dataclass
class IsoResult:
    signal:     np.ndarray
    is_anomaly: np.ndarray
    score:      np.ndarray


def run_isolation_forest(df: pd.DataFrame,
                         cfg: IsoForestConfig | None = None) -> IsoResult:
    from sklearn.ensemble import IsolationForest

    if cfg is None:
        cfg = DEFAULT_CONFIG.iso_forest

    feat = np.column_stack([df[c].values for c in cfg.feature_cols])
    feat = np.nan_to_num(feat)

    clf    = IsolationForest(n_estimators=cfg.n_estimators,
                              contamination=cfg.contamination,
                              random_state=cfg.random_state)
    labels = clf.fit_predict(feat)          # -1 = anomaly, +1 = normal
    scores = clf.score_samples(feat)        # more negative = more anomalous

    is_anomaly = labels == -1

    # Directional: anomaly + price move → signal
    price_dir  = np.sign(df["LogRet"].values)
    signal     = np.where(is_anomaly, price_dir, 0.0).astype(float)

    # Normalise scores to [0, 1]
    raw_score  = -scores
    ptp        = raw_score.max() - raw_score.min()
    norm_score = (raw_score - raw_score.min()) / (ptp if ptp > 0 else 1.0)

    return IsoResult(signal=signal, is_anomaly=is_anomaly, score=norm_score)
