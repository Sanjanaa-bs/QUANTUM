"""
QRNG Streamlit Frontend — Shared Components and Helpers
Minimalist Pastel Teal Theme + Scientific UI Components
"""

import math
from typing import Any, Dict, List, Optional

import requests

API_BASE = "http://127.0.0.1:8000/api/v1"


# ─────────────────────────────────────────────
# API client helpers
# ─────────────────────────────────────────────

def api_get(path: str, params: Dict = None) -> Any:
    """GET from the backend API. Returns parsed JSON or raises."""
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"_error": "Cannot connect to backend. Is the FastAPI server running on port 8000?"}
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return {"_error": detail}
    except Exception as e:
        return {"_error": str(e)}


def api_post(path: str, body: Dict) -> Any:
    """POST to the backend API. Returns parsed JSON or raises."""
    try:
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=300)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"_error": "Cannot connect to backend. Is the FastAPI server running on port 8000?"}
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return {"_error": detail}
    except Exception as e:
        return {"_error": str(e)}


def api_get_bytes(path: str, params: Dict = None) -> Optional[bytes]:
    """GET raw bytes from the backend API (for file downloads)."""
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=120)
        r.raise_for_status()
        return r.content
    except Exception:
        return None


# ─────────────────────────────────────────────
# Pastel Teal Color Palette
# ─────────────────────────────────────────────

PALETTE = {
    "bg_primary": "#F8F7F2",       # warm ivory
    "bg_card": "#FFFFFF",          # clean white
    "bg_sidebar": "#EEF5F2",       # light pastel teal
    "primary": "#6FAFA5",          # primary teal
    "dark": "#286D67",             # dark teal
    "soft": "#B8DAD5",             # soft pastel teal
    "light_mint": "#E4F1ED",       # light mint
    "text_primary": "#344443",     # deep slate teal text
    "text_secondary": "#71817E",   # muted slate text
    "border": "#DCE7E3",           # soft border
    "success": "#4D9A7A",          # sage green
    "warning": "#D9A441",          # warm amber
    "error": "#C96C70",            # dusty coral
    "info": "#719DB8",             # dusty blue
    # Chart colors
    "chart_zeros": "#6FAFA5",      # teal for 0s
    "chart_ones": "#D8A0B0",       # pastel rose for 1s
    "chart_expected": "#286D67",   # dark teal reference line
    "chart_entropy": "#A6CFC5",    # soft mint
    "chart_noise": "#D9B38C",      # warm sand
    "chart_debiased": "#8EAAA2",   # sage teal
}

# ─────────────────────────────────────────────
# Plotly Chart Default Layout
# ─────────────────────────────────────────────

def get_plotly_layout(height: int = 280, title: str = "") -> Dict[str, Any]:
    """Returns standardized Plotly layout dictionary for pastel teal theme."""
    layout = dict(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, system-ui, sans-serif", color="#344443", size=11),
        margin=dict(l=24, r=24, t=36 if title else 20, b=24),
        height=height,
        xaxis=dict(
            gridcolor="#E7EEEB",
            linecolor="#DCE7E3",
            zerolinecolor="#DCE7E3",
            tickfont=dict(color="#71817E", size=10),
            title_font=dict(color="#344443", size=11),
        ),
        yaxis=dict(
            gridcolor="#E7EEEB",
            linecolor="#DCE7E3",
            zerolinecolor="#DCE7E3",
            tickfont=dict(color="#71817E", size=10),
            title_font=dict(color="#344443", size=11),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#DCE7E3",
            borderwidth=1,
            font=dict(color="#344443", size=10),
        ),
    )
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(family="Inter, system-ui, sans-serif", size=13, color="#286D67"),
            x=0.01,
            xanchor="left",
        )
    return layout


# ─────────────────────────────────────────────
# CSS injection (Minimalist Pastel Teal)
# ─────────────────────────────────────────────

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #F8F7F2;
    --bg-card: #FFFFFF;
    --bg-sidebar: #EEF5F2;
    --primary-teal: #6FAFA5;
    --dark-teal: #286D67;
    --soft-teal: #B8DAD5;
    --light-mint: #E4F1ED;
    --text-primary: #344443;
    --text-secondary: #71817E;
    --border-color: #DCE7E3;
    --success: #4D9A7A;
    --warning: #D9A441;
    --error: #C96C70;
    --info: #719DB8;
}

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    color: var(--text-primary);
}

/* Main background */
.stApp {
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar);
    border-right: 1px solid var(--border-color);
}

section[data-testid="stSidebar"] hr {
    border-color: var(--border-color) !important;
}

/* Sidebar navigation radios */
section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
    gap: 4px;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
    padding: 6px 12px;
    border-radius: 8px;
    transition: all 0.15s ease-in-out;
    color: var(--text-primary);
    font-size: 13px;
    font-weight: 500;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
    background-color: var(--light-mint);
    color: var(--dark-teal);
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {
    background-color: var(--soft-teal);
    color: var(--dark-teal);
    font-weight: 600;
}

/* Section headers */
.section-header {
    font-size: 22px;
    font-weight: 700;
    color: var(--dark-teal);
    border-bottom: 2px solid var(--soft-teal);
    padding-bottom: 8px;
    margin-bottom: 16px;
    letter-spacing: -0.3px;
}

/* Minimalist Card */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    color: var(--text-primary);
}

/* Native Streamlit metrics */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}

[data-testid="stMetricValue"] {
    color: var(--dark-teal) !important;
    font-size: 22px !important;
    font-weight: 700 !important;
}

/* Status badges */
.badge-pass {
    background: var(--light-mint);
    color: var(--dark-teal);
    border: 1px solid var(--success);
    border-radius: 16px;
    padding: 3px 12px;
    font-size: 11px;
    font-weight: 600;
    display: inline-block;
}

.badge-fail {
    background: #FDF0F1;
    color: var(--error);
    border: 1px solid var(--error);
    border-radius: 16px;
    padding: 3px 12px;
    font-size: 11px;
    font-weight: 600;
    display: inline-block;
}

.badge-skip {
    background: #F1F4F3;
    color: var(--text-secondary);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 3px 12px;
    font-size: 11px;
    font-weight: 600;
    display: inline-block;
}

.badge-warn {
    background: #FDF8ED;
    color: #9E7422;
    border: 1px solid var(--warning);
    border-radius: 16px;
    padding: 3px 12px;
    font-size: 11px;
    font-weight: 600;
    display: inline-block;
}

/* Source type labels */
.source-ideal {
    background: var(--light-mint);
    color: var(--dark-teal);
    border-left: 4px solid var(--primary-teal);
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    margin: 6px 0;
}

.source-noisy {
    background: #FDF8ED;
    color: #9E7422;
    border-left: 4px solid var(--warning);
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    margin: 6px 0;
}

.source-hardware {
    background: #FDF0F1;
    color: var(--error);
    border-left: 4px solid var(--error);
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    margin: 6px 0;
}

/* Info and warning boxes */
.info-box {
    background: var(--light-mint);
    border: 1px solid var(--soft-teal);
    border-left: 4px solid var(--primary-teal);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    color: var(--text-primary);
    margin: 8px 0;
    line-height: 1.5;
}

.warning-box {
    background: #FDF8ED;
    border: 1px solid #F3DFB7;
    border-left: 4px solid var(--warning);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    color: #8C651B;
    margin: 8px 0;
    line-height: 1.5;
}

.disclaimer-box {
    background: #FDF0F1;
    border: 1px solid #F6D0D3;
    border-left: 4px solid var(--error);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 11px;
    color: #8C373C;
    margin: 8px 0;
    line-height: 1.5;
}

/* Bitstream box */
.bitstream-box {
    background: #FFFFFF;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--dark-teal);
    word-break: break-all;
    max-height: 120px;
    overflow-y: auto;
    line-height: 1.6;
}

/* Buttons */
.stButton > button {
    background-color: var(--dark-teal) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--dark-teal) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 6px 18px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
}

.stButton > button:hover {
    background-color: #1E534E !important;
    border-color: #1E534E !important;
    transform: translateY(-1px);
    box-shadow: 0 3px 6px rgba(40, 109, 103, 0.15) !important;
}

/* Download buttons */
.stDownloadButton > button {
    background-color: var(--light-mint) !important;
    color: var(--dark-teal) !important;
    border: 1px solid var(--soft-teal) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 12px !important;
}

.stDownloadButton > button:hover {
    background-color: var(--soft-teal) !important;
    border-color: var(--primary-teal) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid var(--border-color);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-secondary);
    font-size: 13px;
    font-weight: 500;
    padding: 8px 14px;
    border-bottom: 2px solid transparent;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--dark-teal);
    border-bottom: 2px solid var(--dark-teal);
    font-weight: 600;
}

/* Tables / DataFrames */
.dataframe {
    background: #FFFFFF !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
}

/* Input fields */
.stSelectbox div[data-baseweb="select"] {
    border-color: var(--border-color);
}

/* Clean Plotly container */
div[data-testid="stPlotlyChart"] {
    background: #FFFFFF;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
}
</style>
"""


def source_badge_html(source_type: str) -> str:
    labels = {
        "ideal_simulator": ("🔵 IDEAL SIMULATOR", "source-ideal"),
        "noisy_simulator": ("🟡 NOISY SIMULATOR", "source-noisy"),
        "hardware": ("🔴 REAL HARDWARE", "source-hardware"),
    }
    label, cls = labels.get(source_type, (source_type, ""))
    return f'<div class="{cls}">{label}</div>'


def outcome_badge_html(outcome: str) -> str:
    badges = {
        "PASS": '<span class="badge-pass">✓ PASS</span>',
        "FAIL": '<span class="badge-fail">✗ FAIL</span>',
        "NOT_RUN": '<span class="badge-skip">— NOT RUN</span>',
        "INSUFFICIENT_DATA": '<span class="badge-warn">⚠ INSUFFICIENT DATA</span>',
        "ERROR": '<span class="badge-fail">⚠ ERROR</span>',
    }
    return badges.get(outcome, f'<span class="badge-skip">{outcome}</span>')


def format_bits_preview(bits: str, max_chars: int = 400) -> str:
    if len(bits) <= max_chars:
        return bits
    half = max_chars // 2
    return bits[:half] + "…" + bits[-half:]
