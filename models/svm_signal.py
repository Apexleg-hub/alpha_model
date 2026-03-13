"""
models/svm_signal.py
─────────────────────
SVM directional classifier.
Predicts next-bar direction: +1 (long), -1 (short), 0 (flat → no clear class).

Output
──────
SVMResult.signal      : np.ndarray[float]  — full-length signal array
SVMResult.proba       : np.ndarray[float]  — (n, n_classes) probability matrix
SVMResult.classes     : list               — ordered class labels
SVMResult.accuracy    : float              — out-of-sample accuracy on test fold
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

from alpha_model.config import SVMConfig, DEFAULT_CONFIG


@dataclass
class SVMResult:
    signal:   np.ndarray
    proba:    np.ndarray
    classes:  list
    accuracy: float


def run_svm(df: pd.DataFrame,
            cfg: SVMConfig | None = None) -> SVMResult:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    if cfg is None:
        cfg = DEFAULT_CONFIG.svm

    X = df[cfg.feature_cols].values
    y = np.sign(df["LogRet"].shift(-1).fillna(0).values).astype(int)

    n         = len(X)
    train_end = int(n * cfg.train_ratio)

    scaler = StandardScaler()
    Xtr    = scaler.fit_transform(X[:train_end])
    Xte    = scaler.transform(X[train_end:])

    clf = SVC(kernel=cfg.kernel, C=cfg.C, gamma=cfg.gamma,
              probability=True, class_weight="balanced",
              random_state=cfg.random_state)
    clf.fit(Xtr, y[:train_end])

    preds  = clf.predict(Xte)
    probas = clf.predict_proba(Xte)

    accuracy = float((preds == y[train_end:]).mean())

    # Pad training period
    full_signal = np.zeros(n)
    full_signal[train_end:] = preds

    full_proba = np.full((n, len(clf.classes_)), 1.0 / len(clf.classes_))
    full_proba[train_end:] = probas

    return SVMResult(
        signal=full_signal,
        proba=full_proba,
        classes=list(clf.classes_),
        accuracy=accuracy,
    )
