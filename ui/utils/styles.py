"""
utils/styles.py
───────────────
All custom CSS injected into the Streamlit app.
Call inject_css() once from app.py.
"""

import streamlit as st

CSS = """
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header  { visibility: hidden; }

[data-testid="metric-container"] {
    background: #0f1117;
    border: 1px solid #1e2538;
    border-radius: 10px;
    padding: 14px 18px;
}

.pipeline-header {
    background: linear-gradient(90deg, #1a1f35 0%, #0f1117 100%);
    border-left: 3px solid #4f8ef7;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0 4px 0;
    font-size: 13px;
    font-weight: 600;
    color: #c8d4f0;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.badge-long  { background:#0d3326; color:#34d399; border:1px solid #34d39955;
               padding:4px 14px; border-radius:20px; font-weight:700; font-size:13px; }
.badge-short { background:#3d1212; color:#f87171; border:1px solid #f8717155;
               padding:4px 14px; border-radius:20px; font-weight:700; font-size:13px; }
.badge-flat  { background:#1e2030; color:#94a3b8; border:1px solid #94a3b855;
               padding:4px 14px; border-radius:20px; font-weight:700; font-size:13px; }

.explain-box {
    background: #12161f;
    border: 1px solid #1e2538;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 12.5px;
    color: #94a3b8;
    line-height: 1.6;
    margin-top: 6px;
}

.status-live { color: #34d399; font-size: 11px; font-weight: 600; }
.status-demo { color: #f59e0b; font-size: 11px; font-weight: 600; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
