
alpha_model/
├── app.py              ← Streamlit entry point (~60 lines, pure wiring)
├── pipeline.py         ← Orchestrator (no Streamlit dep, testable standalone)
├── config.py           ← Every tunable constant in one dataclass hierarchy
├── requirements.txt
├── README.md
│
├── data/
│   └── loader.py       ← MT5 live data only, @st.cache_data
│
├── features/
│   └── engineering.py  ← RSI, EMA, EWMA vol, ATR, Bollinger, volume features
│
├── models/
│   ├── regime.py       ← HMM: fits, predicts, labels states by volatility order
│   ├── svm_signal.py   ← SVM: train/test split, probability output
│   ├── lstm_signal.py  ← Gated RNN: swap body for Keras/PyTorch without touching anything else
│   ├── iso_forest.py   ← Isolation Forest: anomaly detection + directional signal
│   └── aggregator.py   ← Regime-weighted consensus, returns AggResult dataclass
│
├── risk/
│   └── engine.py       ← Rolling Kelly + EWMA vol scaling, returns RiskResult
│
├── execution/
│   └── simulator.py    ← Trade sim + equity curve + performance stats
│
└── ui/
    ├── sidebar.py       ← All widgets → SidebarParams dataclass
    ├── header.py        ← Title + 6 KPI cards
    ├── tab_market.py    ← Price chart, regime bands, HMM transition matrix
    ├── tab_signals.py   ← Stacked SVM/LSTM/IsoForest signal panels
    ├── tab_aggregator.py← Score chart, weight table, current signal badge
    ├── tab_risk.py      ← Equity curve, drawdown, Kelly, EWMA vol
    ├── tab_trades.py    ← P&L chart, trade log table
    ├── tab_diagnostics.py← Correlation heatmap, return dist, raw feature table
    ├── tab_validation.py ← Prediction vs outcome + OOS validation dashboard
    ├── tab_targets.py    ← Math-based buy/sell price targets
    └── utils/
        ├── charts.py    ← Shared Plotly tokens, REGIME_COLOR, add_regime_bands
        └── styles.py    ← All CSS in one inject_css() call
