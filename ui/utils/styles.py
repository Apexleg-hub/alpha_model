"""
utils/styles.py
-------------
All custom CSS injected into the Streamlit app.
Call inject_css() once from app.py.
"""

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-0: #0b0f17;
    --bg-1: #0f1524;
    --bg-2: #141c2e;
    --panel: rgba(17, 24, 39, 0.85);
    --panel-border: rgba(148, 163, 184, 0.16);
    --text: #e5e7eb;
    --muted: #94a3b8;
    --accent: #22d3ee;
    --accent-2: #60a5fa;
    --good: #34d399;
    --warn: #f59e0b;
    --bad: #f87171;
    --shadow: 0 10px 30px rgba(2, 6, 23, 0.55);
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text);
}

code, pre, kbd {
    font-family: 'JetBrains Mono', monospace;
}

#MainMenu, footer, header  { visibility: hidden; }

.stApp {
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(34, 211, 238, 0.18), transparent 60%),
        radial-gradient(900px 500px at 110% 10%, rgba(96, 165, 250, 0.16), transparent 55%),
        linear-gradient(180deg, var(--bg-0), var(--bg-1) 45%, var(--bg-2));
    animation: fadeIn 0.5s ease-out;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.4rem;
}

section[data-testid="stSidebar"] {
    background: rgba(8, 12, 20, 0.9);
    border-right: 1px solid var(--panel-border);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.6rem;
}

h1 {
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #e2e8f0, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

h2, h3 {
    letter-spacing: -0.01em;
    color: #e2e8f0;
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.35), transparent);
    margin: 1.2rem 0;
}

.pipeline-kicker {
    font-size: 0.75rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.2rem;
}

.pipeline-sub {
    font-size: 0.95rem;
    color: #cbd5f5;
    margin-bottom: 0.5rem;
}

.explain-box {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.9), rgba(2, 6, 23, 0.9));
    border: 1px solid var(--panel-border);
    border-radius: 14px;
    padding: 12px 16px;
    font-size: 0.9rem;
    color: var(--muted);
    line-height: 1.6;
    margin-top: 0.6rem;
    box-shadow: var(--shadow);
}

.explain-box ul {
    margin: 0.4rem 0 0 1.1rem;
    padding: 0;
}

.explain-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #e2e8f0;
    margin-bottom: 0.2rem;
}

[data-testid="metric-container"] {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(5, 7, 12, 0.92));
    border: 1px solid var(--panel-border);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: var(--shadow);
    animation: riseIn 0.45s ease-out;
}

[data-testid="stMetricLabel"] {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--muted);
}

[data-testid="stMetricValue"] {
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--text);
}

[data-testid="stMetricDelta"] {
    font-size: 0.75rem;
    color: var(--muted);
}

.status-pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    font-weight: 700;
}

.status-live { background: rgba(52, 211, 153, 0.15); color: var(--good); border: 1px solid rgba(52, 211, 153, 0.55); }
.status-demo { background: rgba(245, 158, 11, 0.15); color: var(--warn); border: 1px solid rgba(245, 158, 11, 0.45); }

div[data-testid="stTabs"] {
    margin-top: 0.6rem;
}

div[data-testid="stTabs"] button[role="tab"] {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid var(--panel-border);
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    margin-right: 0.35rem;
    color: var(--muted);
    font-weight: 600;
    transition: all 0.2s ease;
}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--text);
    border-color: rgba(34, 211, 238, 0.6);
    box-shadow: 0 6px 20px rgba(34, 211, 238, 0.2);
}

div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 16px;
    padding: 8px;
    box-shadow: var(--shadow);
}

div[data-testid="stDataFrame"] {
    padding: 6px;
}

.stButton button {
    background: linear-gradient(90deg, rgba(34, 211, 238, 0.8), rgba(96, 165, 250, 0.9));
    color: #0b0f17;
    border: none;
    border-radius: 999px;
    padding: 0.45rem 1.1rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(34, 211, 238, 0.25);
}

a { color: var(--accent); }

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes riseIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
