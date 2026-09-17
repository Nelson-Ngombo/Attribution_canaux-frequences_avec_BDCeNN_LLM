"""
Professional visual theme for BD-CeNN + LLM dashboard.
Provides CSS injection, color palette, and layout constants.
"""

import streamlit as st


# ============================================================================
# COLOR PALETTE (professional dark-accent theme)
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
# GLOBAL CSS INJECTION
# ============================================================================

CUSTOM_CSS = """
<style>
/* ============ Global typography ============ */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
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
    padding: 1.2rem 1.8rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(30, 58, 95, 0.15);
    color: white;
}
.app-header h1 {
    color: white !important;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}
.app-header .subtitle {
    color: #cbd5e1;
    font-size: 0.9rem;
    margin-top: 0.3rem;
}

/* ============ Section headers ============ */
.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #1e3a5f;
    padding: 0.6rem 0;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 1rem;
}

/* ============ Cards ============ */
.info-card {
    background: white;
    border-radius: 10px;
    padding: 1.2rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    margin-bottom: 1rem;
}
.info-card-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

/* ============ Status badges ============ */
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.badge-success { background: #d1fae5; color: #065f46; }
.badge-warning { background: #fef3c7; color: #92400e; }
.badge-danger  { background: #fee2e2; color: #991b1b; }
.badge-info    { background: #dbeafe; color: #1e40af; }
.badge-neutral { background: #e2e8f0; color: #475569; }

/* ============ Streamlit metric override ============ */
[data-testid="stMetric"] {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}
[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    color: #64748b !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
    color: #1e3a5f !important;
    font-weight: 700 !important;
}

/* ============ Buttons ============ */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
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

/* ============ Sidebar ============ */
section[data-testid="stSidebar"] {
    background: #1e293b;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #4a9eff !important;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid #334155;
    padding-bottom: 0.4rem;
    margin-top: 1rem;
}
section[data-testid="stSidebar"] .stButton > button {
    background: #334155;
    color: white !important;
    border: 1px solid #475569;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #4a9eff;
    border-color: #4a9eff;
}

/* ============ Tabs ============ */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: transparent;
    border-bottom: 2px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    height: 44px;
    padding: 0 1.2rem;
    background: transparent;
    border-radius: 8px 8px 0 0;
    font-weight: 500;
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
}

/* ============ Alerts ============ */
[data-testid="stAlert"] {
    border-radius: 10px;
    border-left-width: 4px;
}

/* ============ CCI / CCI+ACI mode tags ============ */
.mode-tag {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.4rem;
}
.mode-tag-cci { background: #dbeafe; color: #1e40af; }
.mode-tag-aci { background: #fee2e2; color: #991b1b; }

/* ============ Experiment card ============ */
.experiment-card {
    background: white;
    border-radius: 10px;
    padding: 1.2rem;
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
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-family: monospace;
    font-size: 0.85rem;
    font-weight: 700;
    margin-right: 0.5rem;
}

/* ============ Log console ============ */
.log-console {
    background: #0f172a;
    color: #e2e8f0;
    font-family: "SF Mono", Monaco, Consolas, monospace;
    font-size: 0.8rem;
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