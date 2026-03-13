"""
config.py
─────────
Single source of truth for every tunable constant in the pipeline.
Swap values here; nothing else needs to change.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# ── Symbols & timeframes ───────────────────────────────────────────────────────
SYMBOLS: List[str] = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
TIMEFRAMES: List[str] = ["D1", "H4", "W1", "MN1"]

# Base prices used for synthetic data generation
BASE_PRICES: Dict[str, float] = {
    "EURUSD": 1.0850, "GBPUSD": 1.2700, "USDJPY": 149.50,
    "AUDUSD": 0.6500, "USDCAD": 1.3600, "USDCHF": 0.8900,
}

# Per-timeframe daily-equivalent volatility
TF_VOL: Dict[str, float] = {
    "H4": 0.0015, "D1": 0.006, "W1": 0.012, "MN1": 0.025,
}

# Pandas freq aliases for synthetic date generation
TF_FREQ: Dict[str, str] = {
    "H4": "4h", "D1": "D", "W1": "W", "MN1": "ME",
}


# ── Feature engineering ────────────────────────────────────────────────────────
@dataclass
class FeatureConfig:
    rsi_periods: List[int]      = field(default_factory=lambda: [7, 14])
    ema_spans: List[int]        = field(default_factory=lambda: [9, 21, 50, 200])
    ewma_lambda: float          = 0.94          # RiskMetrics λ
    annualisation_factor: float = 252.0
    bb_window: int              = 20
    bb_std: float               = 2.0
    atr_period: int             = 14
    vol_ma_window: int          = 20
    mom_periods: List[int]      = field(default_factory=lambda: [5, 20])


# ── Regime detection ────────────────────────────────────────────────────────────
@dataclass
class RegimeConfig:
    n_states: int       = 3
    covariance_type: str = "full"
    n_iter: int         = 200
    random_state: int   = 42


# ── Signal models ──────────────────────────────────────────────────────────────
@dataclass
class SVMConfig:
    kernel: str         = "rbf"
    C: float            = 1.0
    gamma: str          = "scale"
    train_ratio: float  = 0.75
    random_state: int   = 42
    feature_cols: List[str] = field(default_factory=lambda: [
        "RSI_14", "EMA_cross", "Mom_5", "Mom_20",
        "EWMA_Vol", "BB_Pos", "Vol_Ratio", "ATR_14",
    ])


@dataclass
class LSTMConfig:
    lookback: int       = 20
    random_state: int   = 99
    flat_threshold: float = 0.15
    feature_cols: List[str] = field(default_factory=lambda: [
        "LogRet", "RSI_14", "EWMA_Vol", "Mom_5", "BB_Pos",
    ])


@dataclass
class IsoForestConfig:
    n_estimators: int   = 100
    contamination: float = 0.08
    random_state: int   = 42
    feature_cols: List[str] = field(default_factory=lambda: [
        "Volume", "Vol_Ratio", "Vol_Z", "EWMA_Vol", "ATR_14",
    ])


# ── Signal aggregator ──────────────────────────────────────────────────────────
@dataclass
class AggregatorConfig:
    # [SVM weight, LSTM weight, IsoForest weight]
    regime_weights: Dict[str, List[float]] = field(default_factory=lambda: {
        "Trending": [0.45, 0.40, 0.15],
        "Ranging":  [0.35, 0.35, 0.30],
        "Volatile": [0.20, 0.30, 0.50],
    })
    signal_threshold: float = 0.08     # |score| must exceed this to fire a trade


# ── Risk engine ────────────────────────────────────────────────────────────────
@dataclass
class RiskConfig:
    account_equity: float   = 10_000.0
    max_risk_pct: float     = 0.02      # max fraction of equity per trade
    kelly_fraction: float   = 0.25     # fractional Kelly (safety multiplier)
    max_drawdown_pct: float = 0.15
    vol_target: float       = 0.005    # daily vol target for scaling
    min_vol_scale: float    = 0.10     # floor on vol-scaling factor
    lookback_window: int    = 50       # bars used to estimate win/loss stats


# ── Execution ──────────────────────────────────────────────────────────────────
@dataclass
class ExecutionConfig:
    slippage_pct: float    = 0.0002
    commission_pct: float  = 0.00005
    leverage_proxy: float  = 10.0      # used in equity curve simulation


# ── Master config ──────────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    feature:     FeatureConfig    = field(default_factory=FeatureConfig)
    regime:      RegimeConfig     = field(default_factory=RegimeConfig)
    svm:         SVMConfig        = field(default_factory=SVMConfig)
    lstm:        LSTMConfig       = field(default_factory=LSTMConfig)
    iso_forest:  IsoForestConfig  = field(default_factory=IsoForestConfig)
    aggregator:  AggregatorConfig = field(default_factory=AggregatorConfig)
    risk:        RiskConfig       = field(default_factory=RiskConfig)
    execution:   ExecutionConfig  = field(default_factory=ExecutionConfig)


# Default singleton — import this everywhere
DEFAULT_CONFIG = PipelineConfig()
