# Alpha Model v1

Quantitative signal pipeline with plain-English explanations, built on Streamlit.

```
Market Data (MT5)
      ↓
Feature Engineering   ← alpha_model/features/engineering.py
(RSI, EMA, Volume, Volatility)
      ↓
Regime Detection      ← alpha_model/models/regime.py
(HMM)
      ↓
Signal Models         ← alpha_model/models/{svm_signal, lstm_signal, iso_forest}.py
• SVM
• LSTM
• Volume Anomaly (Isolation Forest)
      ↓
Signal Aggregator     ← alpha_model/models/aggregator.py
      ↓
Risk Engine           ← alpha_model/risk/engine.py
(Kelly + volatility sizing)
      ↓
Execution             ← alpha_model/execution/simulator.py
(MT5 API)
```

## Project Structure

```
alpha_model/
├── app.py                    # Streamlit entry point (thin wiring layer)
├── pipeline.py               # End-to-end orchestrator (no Streamlit dep)
├── config.py                 # All tunable constants — one place to change
├── requirements.txt
│
├── data/
│   └── loader.py             # MT5 live data loader (no synthetic fallback)
│
├── features/
│   └── engineering.py        # RSI, EMA, EWMA vol, ATR, Bollinger, volume
│
├── models/
│   ├── regime.py             # HMM regime detection
│   ├── svm_signal.py         # SVM directional classifier
│   ├── lstm_signal.py        # Gated RNN signal (swap for Keras/PyTorch)
│   ├── iso_forest.py         # Isolation Forest volume anomaly
│   └── aggregator.py         # Regime-weighted consensus
│
├── risk/
│   └── engine.py             # Kelly criterion + EWMA volatility sizing
│
├── execution/
│   └── simulator.py          # Trade simulation + equity curve + stats
│
└── ui/
    ├── sidebar.py             # All sidebar widgets → SidebarParams dataclass
    ├── header.py              # Page title + 6 KPI metric cards
    ├── tab_market.py          # Tab 1: price chart + regime bands
    ├── tab_signals.py         # Tab 2: SVM / LSTM / IsoForest signal panels
    ├── tab_aggregator.py      # Tab 3: aggregated signal + weights table
    ├── tab_targets.py         # Tab 4: math-based buy/sell price targets
    ├── tab_risk.py            # Tab 5: equity curve + Kelly + EWMA vol
    ├── tab_trades.py          # Tab 6: trade log + P&L chart
    ├── tab_diagnostics.py     # Tab 7: feature correlations + return dist
    └── tab_validation.py      # Tab 8: prediction vs outcome + OOS validation
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r alpha_model/requirements.txt

# 2. Run the app (from the parent of alpha_model/)
streamlit run alpha_model/app.py
```

## Adding a New Signal Model

1. Create `alpha_model/models/my_model.py` with a `MyResult` dataclass and `run_my_model(df, cfg)` function.
2. Add `MyModelConfig` to `config.py` and include it in `PipelineConfig`.
3. Import and call it in `pipeline.py` between Steps 4 and 5.
4. Add its signal to `aggregator.py` — extend the `regime_weights` dict and the score computation.
5. Add a display column to `ui/tab_signals.py`.

## Integrating Live MT5

Live MT5 data is required. `data/loader.py` calls `mt5.initialize()` and requires a live MT5 connection (no synthetic fallback).

For Windows with MT5 installed:
```bash
pip install MetaTrader5
```
