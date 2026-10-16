"""
Professional visual theme for BD-CeNN + LLM dashboard.
Provides CSS injection, color palette, and layout constants.

Font sizes increased globally for better readability.
"""

from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components


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
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.7rem;
    border-left: 4px solid #64748b;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.6rem;
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
/* Libellé et valeur sur une seule ligne, centrés verticalement */
section[data-testid="stSidebar"] .sidebar-status-label {
    display: flex;
    align-items: center;
    flex-shrink: 0;
    white-space: nowrap;
    font-size: 0.74rem !important;
    color: #cbd5e1 !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    line-height: 1.2;
}
section[data-testid="stSidebar"] .sidebar-status-value {
    flex: 1 1 auto;
    min-width: 0;
    text-align: right;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-family: "SF Mono", Consolas, "Liberation Mono", monospace;
    font-size: 0.8rem !important;
    color: white !important;
    font-weight: 600;
    line-height: 1.2;
}
.sidebar-status-icon {
    display: inline-block;
    flex-shrink: 0;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    margin-right: 0.45rem;
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

/* ============ Sidebar LLM parameters card ============ */
section[data-testid="stSidebar"] .llm-card {
    background: linear-gradient(160deg, #24324a 0%, #1b2638 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 0.85rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25);
}
section[data-testid="stSidebar"] .llm-card-model {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding-bottom: 0.7rem;
    margin-bottom: 0.7rem;
    border-bottom: 1px solid #334155;
}
section[data-testid="stSidebar"] .llm-icon-box {
    width: 34px;
    height: 34px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    background: rgba(74, 158, 255, 0.12);
    border: 1px solid rgba(74, 158, 255, 0.35);
}
section[data-testid="stSidebar"] .llm-icon {
    display: inline-flex;
    color: #4a9eff !important;
}
/* La règle globale de la barre latérale colore aussi les SVG : on rétablit l'héritage */
section[data-testid="stSidebar"] .llm-icon svg,
section[data-testid="stSidebar"] .llm-icon svg * {
    color: inherit !important;
}
section[data-testid="stSidebar"] .llm-card-label {
    font-size: 0.66rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8 !important;
}
section[data-testid="stSidebar"] .llm-card-model-name {
    font-family: "SF Mono", Consolas, "Liberation Mono", monospace;
    font-size: 0.92rem !important;
    font-weight: 600;
    color: #f8fafc !important;
    word-break: break-all;
}
section[data-testid="stSidebar"] .llm-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.45rem;
}
section[data-testid="stSidebar"] .llm-tile {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid #2c3a52;
    border-radius: 8px;
    padding: 0.45rem 0.55rem;
}
section[data-testid="stSidebar"] .llm-tile-head {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.64rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    white-space: nowrap;
    color: #94a3b8 !important;
}
section[data-testid="stSidebar"] .llm-tile-value {
    margin-top: 0.15rem;
    font-family: "SF Mono", Consolas, "Liberation Mono", monospace;
    font-size: 0.95rem !important;
    font-weight: 600;
    color: #f1f5f9 !important;
}
section[data-testid="stSidebar"] .llm-directive {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-top: 0.7rem;
    font-size: 0.78rem !important;
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] .llm-directive .llm-icon {
    color: #22c55e !important;
}
section[data-testid="stSidebar"] .llm-badge {
    margin-left: auto;
    padding: 0.1rem 0.5rem;
    border-radius: 999px;
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.4);
    font-family: "SF Mono", Consolas, "Liberation Mono", monospace;
    font-size: 0.72rem !important;
    font-weight: 600;
    white-space: nowrap;
    color: #86efac !important;
}
section[data-testid="stSidebar"] .llm-note {
    margin-top: 0.55rem;
    font-size: 0.72rem !important;
    color: #94a3b8 !important;
}

/* ============ Sidebar footer (author, copyright, version) ============ */
section[data-testid="stSidebar"] .sidebar-footer {
    margin-top: 0.4rem;
    padding: 0.8rem 0.9rem;
    border-radius: 10px;
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid #334155;
}
section[data-testid="stSidebar"] .sidebar-footer-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
section[data-testid="stSidebar"] .sidebar-footer-author {
    font-size: 0.92rem !important;
    font-weight: 600;
    color: #f1f5f9 !important;
}
section[data-testid="stSidebar"] .sidebar-footer-affiliation {
    font-size: 0.72rem !important;
    color: #94a3b8 !important;
}
section[data-testid="stSidebar"] .sidebar-footer-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 0.65rem;
    padding-top: 0.55rem;
    border-top: 1px solid #334155;
    font-size: 0.72rem !important;
    color: #94a3b8 !important;
}
section[data-testid="stSidebar"] .sidebar-footer-meta span {
    font-size: 0.72rem !important;
}
section[data-testid="stSidebar"] .sidebar-footer-version {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.1rem 0.5rem;
    border-radius: 999px;
    background: rgba(74, 158, 255, 0.12);
    border: 1px solid rgba(74, 158, 255, 0.35);
    font-family: "SF Mono", Consolas, "Liberation Mono", monospace;
    font-size: 0.72rem !important;
    color: #93c5fd !important;
}
section[data-testid="stSidebar"] .sidebar-footer-version .llm-icon {
    color: #93c5fd !important;
}

/* ============ Réexécutions fluides : aucun voile sur les éléments en attente ============ */
[data-stale="true"],
.stale-element,
[data-testid="stElementContainer"][data-stale="true"] {
    opacity: 1 !important;
    filter: none !important;
    transition: none !important;
}

/* Barre de progression indéterminée, visible uniquement pendant l'exécution du script */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 32vw;
    height: 3px;
    z-index: 1000000;
    border-radius: 0 3px 3px 0;
    background: linear-gradient(90deg, rgba(74, 158, 255, 0) 0%, #4a9eff 45%, #22c55e 100%);
    opacity: 0;
    transform: translateX(-100%);
    transition: opacity 0.2s ease;
    pointer-events: none;
}
.stApp:has([data-testid="stStatusWidget"])::before {
    opacity: 1;
    animation: bdcenn-run-progress 1.15s cubic-bezier(0.4, 0, 0.2, 1) infinite;
}
@keyframes bdcenn-run-progress {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(320%); }
}

/* ============ En-tête de barre latérale : bouton de repli en surimpression ============ */
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
    position: absolute;
    top: 0.4rem;
    right: 0.4rem;
    height: auto;
    min-height: 0;
    padding: 0;
    margin: 0;
    z-index: 5;
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 0.9rem !important;
}
section[data-testid="stSidebar"] .sidebar-brand {
    font-size: 1.15rem !important;
    font-weight: 700;
    letter-spacing: 0.01em;
    color: #f8fafc !important;
    padding-right: 2.2rem;
    margin: 0 0 0.2rem 0;
}

/* Déclencheur masqué du remontage du graphe, cliqué par script */
.st-key-graph_tab_refresh {
    display: none !important;
}
</style>
"""


def apply_theme():
    """Injects the custom CSS. Must be called once after set_page_config."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def install_plotly_resize_fix():
    """
    Une figure Plotly tracée dans un onglet masqué garde une largeur nulle.
    Au clic sur l'onglet du graphe, un script injecté dans la page parente
    actionne un bouton masqué qui redessine la figure à sa largeur visible.
    """
    components.html(
        """
        <script>
        const parentWin = window.parent;
        if (!parentWin.__bdcennGraphTabHook) {
            parentWin.__bdcennGraphTabHook = true;
            // Script exécuté dans le contexte parent : survit au remplacement de cet iframe
            const hook = parentWin.document.createElement("script");
            hook.textContent = `
                document.addEventListener("click", (event) => {
                    const tab = event.target.closest('button[role="tab"]');
                    if (!tab || !tab.innerText.trim().startsWith("Graphe")) return;
                    setTimeout(() => {
                        const trigger = document.querySelector(".st-key-graph_tab_refresh button");
                        if (trigger) trigger.click();
                    }, 80);
                }, true);
            `;
            parentWin.document.head.appendChild(hook);
        }
        </script>
        """,
        height=0,
    )


    """
    Une figure Plotly tracée dans un onglet masqué garde une largeur nulle.
    Au clic sur l'onglet du graphe, un script injecté dans la page parente
    actionne un bouton masqué qui redessine la figure à sa largeur visible.
    """
    components.html(
        """
        <script>
        const parentWin = window.parent;
        if (!parentWin.__bdcennGraphTabHook) {
            parentWin.__bdcennGraphTabHook = true;
            // Script exécuté dans le contexte parent : survit au remplacement de cet iframe
            const hook = parentWin.document.createElement("script");
            hook.textContent = `
                document.addEventListener("click", (event) => {
                    const tab = event.target.closest('button[role="tab"]');
                    if (!tab || !tab.innerText.trim().startsWith("Graphe")) return;
                    setTimeout(() => {
                        const trigger = document.querySelector(".st-key-graph_tab_refresh button");
                        if (trigger) trigger.click();
                    }, 80);
                }, true);
            `;
            parentWin.document.head.appendChild(hook);
        }
        </script>
        """,
        height=0,
    )


# Tracés vectoriels au trait (grille 24 x 24), coloriés par currentColor
_ICON_PATHS = {
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
           '<path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>',
    "thermometer": '<path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"/>',
    "sliders": '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
    "layers": '<path d="M12 2 2 7l10 5 10-5-10-5z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
            '<path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/>',
    "hash": '<path d="M4 9h16M4 15h16M10 3 8 21M16 3l-2 18"/>',
    "check-circle": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m22 4-10 10.01-3-3"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/>',
    "user": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "tag": '<path d="M20.59 13.41 13.42 20.58a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/>'
           '<circle cx="7" cy="7" r="1.5"/>',
}


def svg_icon(name: str, size: int = 14) -> str:
    """Icône vectorielle au trait, héritant de la couleur du texte parent."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{_ICON_PATHS[name]}</svg>'
    )


def render_sidebar_llm_params(params: dict):
    """Carte des paramètres d'inférence du LLM (modèle exact et réglages gelés)."""
    tiles = [
        ("thermometer", "Temperature", params["temperature"]),
        ("sliders", "Top-p", params["top_p"]),
        ("layers", "Top-k", params["top_k"]),
        ("file", "Max tokens", params["max_output_tokens"]),
        ("hash", "Seed", params["seed"]),
        ("check-circle", "Audit sem.", params.get("semantic_audit_version", "-")),
    ]
    tiles_html = "".join(
        f'<div class="llm-tile"><div class="llm-tile-head">'
        f'<span class="llm-icon">{svg_icon(icon, 12)}</span>{label}</div>'
        f'<div class="llm-tile-value">{value}</div></div>'
        for icon, label, value in tiles
    )
    campaign_note = (
        f'<div class="llm-note">Modèle unique (dashboard et E10) : '
        f'{params.get("campaign_model", params["model"])}</div>'
        if params.get("fixed_model_for_campaign") else ""
    )
    # HTML sans indentation : quatre espaces en tête de ligne produisent un bloc de code Markdown
    html = (
        '<div class="llm-card">'
        '<div class="llm-card-model">'
        f'<div class="llm-icon-box"><span class="llm-icon">{svg_icon("cpu", 18)}</span></div>'
        '<div><div class="llm-card-label">Modèle actif</div>'
        f'<div class="llm-card-model-name">{params["model"]}</div></div>'
        '</div>'
        f'<div class="llm-grid">{tiles_html}</div>'
        f'<div class="llm-directive"><span class="llm-icon">{svg_icon("shield", 14)}</span>'
        'Directive Zero-Hallucination'
        f'<span class="llm-badge">{params["directive_version"]}</span></div>'
        f'{campaign_note}'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_sidebar_footer(author: str, code_version: str,
                          affiliation: str = "Faculté Polytechnique, Université de Kinshasa"):
    """Pied de barre latérale : auteur, mention de copyright et version du code."""
    html = (
        '<div class="sidebar-footer">'
        '<div class="sidebar-footer-row">'
        f'<div class="llm-icon-box"><span class="llm-icon">{svg_icon("user", 16)}</span></div>'
        f'<div><div class="sidebar-footer-author">{author}</div>'
        f'<div class="sidebar-footer-affiliation">{affiliation}</div></div>'
        '</div>'
        '<div class="sidebar-footer-meta">'
        f'<span>&copy; {datetime.now().year} BD-CeNN + LLM</span>'
        f'<span class="sidebar-footer-version"><span class="llm-icon">{svg_icon("tag", 11)}</span>'
        f'v{code_version}</span>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


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
        <div class="sidebar-status-value" title="{value}">{value}</div>
    </div>
    """, unsafe_allow_html=True)