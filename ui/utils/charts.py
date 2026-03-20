"""
utils/charts.py
-------------
Shared Plotly layout constants and color helpers.
Import once; every UI module uses the same token set.
"""

from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Design tokens
CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10, 14, 22, 0.85)",
    font=dict(family="Space Grotesk, sans-serif", size=12, color="#cbd5f5"),
    margin=dict(l=50, r=20, t=36, b=30),
    legend=dict(bgcolor="rgba(15, 23, 42, 0.75)", borderwidth=0),
)

REGIME_FILL  = {
    "Trending": "rgba(52, 211, 153, 0.18)",
    "Ranging":  "rgba(99, 179, 237, 0.18)",
    "Volatile": "rgba(248, 113, 113, 0.20)",
}
REGIME_COLOR = {
    "Trending": "#34d399",
    "Ranging": "#63b3ed",
    "Volatile": "#f87171",
}

CANDLE_UP   = "#34d399"
CANDLE_DOWN = "#f87171"
EMA_COLORS  = {9: "#22d3ee", 21: "#60a5fa", 50: "#a78bfa", 200: "#94a3b8"}


# Helper: add regime background bands to any row


def add_regime_bands(fig: go.Figure, dates, regimes: np.ndarray,
                     row: int = 1, col: int = 1) -> None:
    for reg in np.unique(regimes):
        idx = np.where(regimes == reg)[0]
        if not len(idx):
            continue
        groups, start = [], idx[0]
        for k in range(1, len(idx)):
            if idx[k] - idx[k - 1] > 1:
                groups.append((start, idx[k - 1]))
                start = idx[k]
        groups.append((start, idx[-1]))
        for s, e in groups:
            fig.add_vrect(
                x0=dates.iloc[s], x1=dates.iloc[min(e + 1, len(dates) - 1)],
                fillcolor=REGIME_FILL[reg], line_width=0,
                annotation_text="" if (e - s) < 3 else reg[0],
                annotation_font=dict(size=8, color=REGIME_COLOR[reg]),
                annotation_position="top left",
                row=row, col=col,
            )



def apply_base_layout(fig: go.Figure, height: int = 500,
                      title: str = "") -> go.Figure:
    fig.update_layout(**CHART_LAYOUT, height=height, title=title)
    fig.update_xaxes(showgrid=False, rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.2)", gridwidth=0.5)
    return fig
