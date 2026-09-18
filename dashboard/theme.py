"""
Professional visual theme for BD-CeNN + LLM dashboard.
Provides CSS injection, color palette, and layout constants.

Font sizes increased globally for better readability.
"""

import streamlit as st


# ============================================================================
# COLOR PALETTE
# ============================================================================

COLORS = {
    "primary": "#1e3a5f",
    "primary_light": "#2a5285",
    "primary_dark": "#0f2340",
    "accent": "#4a9eff",
    "accent_hover": "#3d8ce6",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "info": "#3b82f6",
    "bg_main": "#f8fafc",
    "bg_card": "#ffffff",
    "bg_sidebar": "#1e293b",
    "text_primary": "#1e293b",
    "text_secondary": "#64748b",
    "border": "#e2e8f0",
    "cci_color": "#3b82f6",
    "cci_aci_color": "#ef4444",
}


# ============================================================================
# GLOBAL CSS INJECTION (LARGER FONTS)
# ============================================================================

CUSTOM_CSS = """
<style>
/* ============ Global typography (increased base size) ============ */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    font-size: 16px;
}

/* Base paragraph size */
.stMarkdown p, .stMarkdown li {
    font-size: 1rem;
    line-height: 1.6;
}

/* Body text */
p, span, div {
    font-size: 1rem;
}

/* Captions and small text */
.stCaption, [data-testid="stCaptionContainer"] {
    font-size: 0.9rem !important;
    color: #64748b;
}

/* ============ Main container ============ */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* ============ App header ============ */
.app-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2a5285 100%);
    padding: 1.4rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(30, 58, 95, 0.15);
    color: white;
}
.app-header h1 {
    color: white !important;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}
.app-header .subtitle {
    color: #cbd5e1;
    font-size: 1.05rem;
    margin-top: 0.4rem;
}

/* ============ Section headers ============ */
.section-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #1e3a5f;
    padding: 0.7rem 0;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 1.2rem;
}

/* ============ Streamlit headers ============ */
h1 { font-size: 2rem !important; }
h2 { font-size: 1.6rem !important; }
h3 { font-size: 1.3rem !important; }
h4 { font-size: 1.15rem !important; }

/* ============ Cards ============ */
.info-card {
    background: white;
    border-radius: 10px;
    padding: 1.3rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    margin-bottom: 1rem;
}
.info-card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.6rem;
}

/* ============ Status badges ============ */
.status-badge {
    display: inline-block;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.badge-success { background: #d1fae5; color: #065f46; }
.badge-warning { background: #fef3c7; color: #92400e; }
.badge-danger  { background: #fee2e2; color: #991b1b; }
.badge-info    { background: #dbeafe; color: #1e40af; }
.badge-neutral { background: #e2e8f0; color: #475569; }

/* ============ Streamlit metrics ============ */
[data-testid="stMetric"] {
    background: white;
    padding: 1.1rem;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}
[data-testid="stMetricLabel"] {
    font-size: 0.9rem !important;
    color: #64748b !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    color: #1e3a5f !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.95rem !important;
}

/* ============ Buttons ============ */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    padding: 0.55rem 1.3rem;
    font-size: 1rem;
    transition: all 0.15s ease;
    border: 1px solid #e2e8f0;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1e3a5f 0%, #2a5285 100%);
    border: none;
    color: white;
}

/* ============ Download buttons ============ */
[data-testid="stDownloadButton"] > button {
    font-size: 0.95rem;
    padding: 0.55rem 1rem;
}

/* ============ Sidebar (compact, no padding at top) ============ */
section[data-testid="stSidebar"] {
    background: #1e293b;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.5rem !important;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #4a9eff !important;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid #334155;
    padding-bottom: 0.4rem;
    margin-top: 1.2rem;
    margin-bottom: 0.6rem;
}
section[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.95rem;
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: #334155;
    color: white !important;
    border: 1px solid #475569;
    font-size: 0.95rem;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #4a9eff;
    border-color: #4a9eff;
}

/* ============ Sidebar system status cards ============ */
.sidebar-status-card {
    background: #334155;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.7rem;
    border-left: 4px solid #64748b;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
}
.sidebar-status-card.status-ok {
    border-left-color: #22c55e;
    background: linear-gradient(135deg, #14532d 0%, #166534 100%);
}
.sidebar-status-card.status-warn {
    border-left-color: #f59e0b;
    background: linear-gradient(135deg, #713f12 0%, #854d0e 100%);
}
.sidebar-status-card.status-err {
    border-left-color: #ef4444;
    background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
}
.sidebar-status-card.status-init {
    border-left-color: #3b82f6;
    background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
}
.sidebar-status-label {
    font-size: 0.85rem;
    color: #cbd5e1 !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.sidebar-status-value {
    font-size: 0.95rem;
    color: white !important;
    font-weight: 600;
    text-align: right;
    max-width: 60%;
    word-wrap: break-word;
}
.sidebar-status-icon {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 0.5rem;
    vertical-align: middle;
}
.sidebar-status-icon.dot-ok { background: #22c55e; box-shadow: 0 0 8px rgba(34, 197, 94, 0.6); }
.sidebar-status-icon.dot-warn { background: #f59e0b; box-shadow: 0 0 8px rgba(245, 158, 11, 0.6); }
.sidebar-status-icon.dot-err { background: #ef4444; box-shadow: 0 0 8px rgba(239, 68, 68, 0.6); }
.sidebar-status-icon.dot-init {
    background: #3b82f6;
    box-shadow: 0 0 8px rgba(59, 130, 246, 0.6);
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* ============ Tabs ============ */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: transparent;
    border-bottom: 2px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    height: 48px;
    padding: 0 1.3rem;
    background: transparent;
    border-radius: 8px 8px 0 0;
    font-weight: 500;
    font-size: 1rem;
    color: #64748b;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #1e3a5f !important;
    border-bottom: 3px solid #4a9eff !important;
    font-weight: 600;
}

/* ============ Expander ============ */
.streamlit-expanderHeader {
    background: #f1f5f9;
    border-radius: 6px;
    font-weight: 500;
    font-size: 1rem;
}

/* ============ Progress bar ============ */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #4a9eff 0%, #22c55e 100%);
}

/* ============ Data frames ============ */
.dataframe {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
    font-size: 0.95rem;
}

/* ============ Alerts ============ */
[data-testid="stAlert"] {
    border-radius: 10px;
    border-left-width: 4px;
    font-size: 0.98rem;
}

/* ============ Radio, checkbox, selectbox ============ */
.stRadio label, .stCheckbox label, .stSelectbox label {
    font-size: 1rem !important;
}
.stRadio div[role="radiogroup"] label {
    font-size: 0.98rem !important;
}

/* ============ Input labels ============ */
label {
    font-size: 1rem !important;
}

/* ============ CCI / CCI+ACI mode tags ============ */
.mode-tag {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 4px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-right: 0.4rem;
}
.mode-tag-cci { background: #dbeafe; color: #1e40af; }
.mode-tag-aci { background: #fee2e2; color: #991b1b; }

/* ============ Experiment card ============ */
.experiment-card {
    background: white;
    border-radius: 10px;
    padding: 1.3rem;
    border: 1px solid #e2e8f0;
    margin-bottom: 0.8rem;
    transition: all 0.15s ease;
}
.experiment-card:hover {
    border-color: #4a9eff;
    box-shadow: 0 2px 8px rgba(74, 158, 255, 0.15);
}
.experiment-card-code {
    display: inline-block;
    background: #1e3a5f;
    color: white;
    padding: 0.25rem 0.7rem;
    border-radius: 4px;
    font-family: monospace;
    font-size: 0.95rem;
    font-weight: 700;
    margin-right: 0.5rem;
}

/* ============ Log console ============ */
.log-console {
    background: #0f172a;
    color: #e2e8f0;
    font-family: "SF Mono", Monaco, Consolas, monospace;
    font-size: 0.9rem;
    padding: 1rem;
    border-radius: 8px;
    max-height: 300px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.5;
}
</style>
"""


def apply_theme():
    """Injects the custom CSS. Must be called once after set_page_config."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_app_header(title: str = "BD-CeNN + LLM",
                      subtitle: str = "Attribution de canaux neuro-symbolique sous contraintes d'interference"):
    """Renders the branded application header."""
    st.markdown(f"""
    <div class="app-header">
        <h1>{title}</h1>
        <div class="subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def section_title(text: str):
    """Renders a styled section title."""
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def status_badge(text: str, kind: str = "info") -> str:
    """Returns HTML for a status badge. kind: success | warning | danger | info | neutral."""
    return f'<span class="status-badge badge-{kind}">{text}</span>'


def mode_tag(mode: str) -> str:
    """Returns HTML for a CCI / CCI+ACI mode tag."""
    if mode in ("cci", "CCI-only"):
        return '<span class="mode-tag mode-tag-cci">CCI-only</span>'
    return '<span class="mode-tag mode-tag-aci">CCI+ACI</span>'


def render_sidebar_status_card(label: str, value: str, kind: str = "info"):
    """
    Renders a professional status card in the sidebar.
    kind: 'ok' | 'warn' | 'err' | 'init'
    """
    class_map = {
        "ok": ("status-ok", "dot-ok"),
        "warn": ("status-warn", "dot-warn"),
        "err": ("status-err", "dot-err"),
        "init": ("status-init", "dot-init"),
    }
    card_class, dot_class = class_map.get(kind, ("", "dot-ok"))

    st.markdown(f"""
    <div class="sidebar-status-card {card_class}">
        <div class="sidebar-status-label">
            <span class="sidebar-status-icon {dot_class}"></span>{label}
        </div>
        <div class="sidebar-status-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)