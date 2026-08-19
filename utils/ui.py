import html
import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st


APP_ROOT = Path(__file__).resolve().parents[1]


def data_refresh_control() -> None:
    """Render a refresh data control in the Data Hub.
    
    Provides a dropdown selector allowing users to manually trigger a cache
    clear for improved data freshness. Uses Streamlit's session state to track
    refresh requests and force rerun when triggered.
    """
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.caption("💾 Data Refresh")
    
    with col2:
        refresh_options = {
            "Manual (default)": "manual",
            "🔄 Refresh now": "refresh",
        }
        selected = st.selectbox(
            "Data refresh",
            list(refresh_options.keys()),
            key="data_refresh_control",
            label_visibility="collapsed",
        )
        
        if refresh_options[selected] == "refresh":
            # Clear Streamlit cache to force re-fetch from Snowflake
            st.cache_data.clear()
            st.success("✅ Data cache cleared. Reloading data on next refresh...")
            st.session_state.data_refresh_control = "Manual (default)"
            st.rerun()
ASSETS_DIR = APP_ROOT / "assets"
BADGE_PATH = ASSETS_DIR / "charlton_badge.png"

CHARLTON_RED = "#c30017"
CHARLTON_DEEP_RED = "#9c0214"
CHARLTON_BLACK = "#111111"
CHARLTON_CHARCOAL = "#18181b"
CHARLTON_WHITE = "#ffffff"
CHARLTON_OFF_WHITE = "#f4f5f7"
CHARLTON_BORDER = "#d8dde6"
CHARLTON_MUTED = "#667085"
AXIS_TEXT = "#111111"
AXIS_LINE = "#98a2b3"
AXIS_FONT_FAMILY = "Inter SemiBold, Arial, sans-serif"


@lru_cache(maxsize=1)
def badge_data_uri() -> str:
    if not BADGE_PATH.exists():
        return ""
    encoded = base64.b64encode(BADGE_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def badge_html(class_name: str, alt: str = "Charlton Athletic crest") -> str:
    uri = badge_data_uri()
    if uri:
        return f'<img class="{esc(class_name)}" src="{uri}" alt="{esc(alt)}">'
    return f'<div class="{esc(class_name)} cafc-badge-fallback">CAFC</div>'


def _bold_axis_title_text(value: object) -> str:
    text = "" if value is None else str(value)
    if not text:
        return text
    stripped = text.strip().lower()
    if stripped.startswith("<b>") and stripped.endswith("</b>"):
        return text
    return f"<b>{text}</b>"


def _apply_plotly_axis_style(fig: object) -> object:
    if not hasattr(fig, "update_xaxes") or not hasattr(fig, "update_yaxes"):
        return fig

    try:
        fig.update_xaxes(
            linecolor=AXIS_LINE,
            tickcolor=AXIS_LINE,
            tickfont=dict(size=12, color=AXIS_TEXT, family=AXIS_FONT_FAMILY),
            title_font=dict(size=14, color=AXIS_TEXT, family=AXIS_FONT_FAMILY),
        )
        fig.update_yaxes(
            linecolor=AXIS_LINE,
            tickcolor=AXIS_LINE,
            tickfont=dict(size=12, color=AXIS_TEXT, family=AXIS_FONT_FAMILY),
            title_font=dict(size=14, color=AXIS_TEXT, family=AXIS_FONT_FAMILY),
        )

        layout = getattr(fig, "layout", None)
        if layout is not None:
            for axis_name in layout:
                if not str(axis_name).startswith(("xaxis", "yaxis")):
                    continue
                axis = getattr(layout, axis_name, None)
                title = getattr(axis, "title", None)
                title_text = getattr(title, "text", None)
                if title_text:
                    title.text = _bold_axis_title_text(title_text)
    except Exception:
        return fig
    return fig


def _patch_plotly_chart_axis_style() -> None:
    if getattr(st.plotly_chart, "_statsearch_axis_patch", False):
        return

    original_plotly_chart = st.plotly_chart

    def patched_plotly_chart(*args, **kwargs):
        if args:
            _apply_plotly_axis_style(args[0])
        else:
            for key in ("figure_or_data", "figure", "fig"):
                if key in kwargs:
                    _apply_plotly_axis_style(kwargs[key])
                    break
        return original_plotly_chart(*args, **kwargs)

    patched_plotly_chart._statsearch_axis_patch = True
    patched_plotly_chart._statsearch_original = original_plotly_chart
    st.plotly_chart = patched_plotly_chart


def apply_statsearch_theme() -> None:
    """Shared visual treatment for the Charlton internal analytics platform."""
    _patch_plotly_chart_axis_style()
    st.markdown(
        """
        <style>
        :root {
            --ss-bg: #f4f5f7;
            --ss-ink: #151515;
            --ss-muted: #667085;
            --ss-border: #d8dde6;
            --ss-panel: #ffffff;
            --ss-panel-soft: #f7f8fa;
            --ss-dark: #111111;
            --ss-accent: #c30017;
            --ss-accent-dark: #9c0214;
            --ss-accent-soft: #fff1f3;
            --ss-shadow: 0 14px 34px rgba(16, 24, 40, 0.10);
        }

        .stApp {
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.65) 0%, rgba(244, 245, 247, 1) 220px),
                var(--ss-bg);
            color: var(--ss-ink);
        }

        [data-testid="stHeader"] {
            background: rgba(244, 245, 247, 0.94);
            border-bottom: 1px solid rgba(216, 221, 230, 0.76);
            backdrop-filter: blur(10px);
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, #111111 0%, #1c1113 54%, #6f0712 128%);
            border-right: 1px solid rgba(255, 255, 255, 0.12);
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] small {
            color: rgba(255, 255, 255, 0.76) !important;
        }

        [data-testid="stSidebarNav"] {
            padding-top: 8px;
        }

        [data-testid="stSidebarNav"] a {
            border-radius: 8px;
            margin: 2px 8px;
            border-left: 3px solid transparent;
        }

        [data-testid="stSidebarNav"] a:hover,
        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: rgba(195, 0, 23, 0.22);
            border-left-color: #ffffff;
        }

        [data-testid="stSidebarHeader"],
        [data-testid="stSidebar"] [data-testid="stLogo"] {
            align-items: center;
            justify-content: center;
            padding-top: 18px;
            padding-bottom: 12px;
        }

        [data-testid="stLogo"] img,
        [data-testid="stSidebarHeader"] img,
        [data-testid="stSidebar"] img[alt="Logo"] {
            width: 112px !important;
            height: 112px !important;
            max-height: 112px !important;
            object-fit: contain;
        }

        .cafc-sidebar-brand {
            display: flex;
            justify-content: center;
            padding: 8px 0 14px;
            margin: 0 8px 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.14);
        }

        .cafc-sidebar-badge {
            display: block;
            width: 104px;
            height: 104px;
            object-fit: contain;
            filter: drop-shadow(0 12px 18px rgba(0, 0, 0, 0.34));
        }

        .main .block-container {
            max-width: 1280px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: var(--ss-ink);
            letter-spacing: 0;
        }

        h2, h3 {
            font-weight: 800;
        }

        div[data-testid="stMetric"] {
            background: var(--ss-panel);
            border: 1px solid var(--ss-border);
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            border-top: 3px solid var(--ss-accent);
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--ss-muted);
            font-weight: 650;
        }

        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] * {
            color: var(--ss-ink);
            font-size: clamp(1.35rem, 1.75vw, 1.75rem) !important;
            font-weight: 400;
            letter-spacing: -0.03em;
            line-height: 1.1;
        }

        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] > div {
            max-width: 100%;
            overflow: visible !important;
            overflow-wrap: anywhere;
            text-overflow: clip !important;
            white-space: normal !important;
        }

        div[data-testid="stButton"] > button {
            width: 100%;
            border-radius: 8px;
            border: 1px solid var(--ss-accent);
            background: var(--ss-accent);
            color: #ffffff;
            font-weight: 700;
        }

        div[data-testid="stButton"] > button:hover {
            border-color: var(--ss-accent-dark);
            background: var(--ss-accent-dark);
            color: #ffffff;
        }

        .stTextInput input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            border-radius: 8px;
            border-color: var(--ss-border);
        }

        .stDataFrame,
        div[data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
        }

        .ss-brandbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 16px;
            color: var(--ss-muted);
            font-size: 0.82rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .ss-brand {
            display: inline-flex;
            align-items: center;
            gap: 16px;
            color: var(--ss-ink);
        }

        .ss-logo {
            display: inline-grid;
            place-items: center;
            width: 72px;
            height: 72px;
            border-radius: 50%;
            background: #ffffff;
            color: #ffffff;
            font-weight: 900;
            letter-spacing: 0;
            border: 1px solid rgba(17, 17, 17, 0.12);
            box-shadow: 0 5px 12px rgba(17, 17, 17, 0.12);
            object-fit: contain;
        }

        .ss-brand-title {
            color: var(--ss-ink);
            font-size: clamp(1.35rem, 2.1vw, 1.9rem);
            font-weight: 900;
            letter-spacing: 0.02em;
            line-height: 1;
        }

        .ss-logo-mark {
            background: var(--ss-accent);
            color: #ffffff;
            border-radius: 8px;
        }

        .ss-hero {
            border-radius: 8px;
            background:
                radial-gradient(circle at 88% 18%, rgba(255, 255, 255, 0.16), transparent 19%),
                linear-gradient(135deg, rgba(17, 17, 17, 0.99) 0%, rgba(64, 13, 18, 0.98) 56%, rgba(156, 2, 20, 0.98) 145%);
            color: #ffffff;
            padding: 34px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: var(--ss-shadow);
            position: relative;
            overflow: hidden;
        }

        .ss-hero::before {
            content: "";
            position: absolute;
            inset: 0;
            border-top: 5px solid var(--ss-accent);
            pointer-events: none;
        }

        .ss-hero-inner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 28px;
            position: relative;
            z-index: 1;
        }

        .ss-hero-copy {
            min-width: 0;
        }

        .ss-hero-badge {
            width: clamp(86px, 12vw, 138px);
            height: clamp(86px, 12vw, 138px);
            object-fit: contain;
            flex: 0 0 auto;
            filter: drop-shadow(0 14px 20px rgba(0, 0, 0, 0.32));
        }

        .ss-eyebrow {
            color: #ffffff;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .ss-title {
            font-size: clamp(2rem, 3.5vw, 3rem);
            line-height: 1.04;
            font-weight: 850;
            max-width: 820px;
            margin: 0;
            letter-spacing: 0;
            color: #ffffff;
        }

        .ss-subtitle {
            color: rgba(255, 255, 255, 0.78);
            max-width: 760px;
            margin-top: 12px;
            font-size: 1rem;
            line-height: 1.55;
        }

        .ss-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 22px;
        }

        .ss-pill {
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 999px;
            padding: 7px 12px;
            color: rgba(255, 255, 255, 0.84);
            background: rgba(255, 255, 255, 0.06);
            font-size: 0.84rem;
            font-weight: 650;
        }

        .ss-section-label {
            margin: 30px 0 10px;
            color: var(--ss-muted);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .ss-panel {
            background: var(--ss-panel);
            border: 1px solid var(--ss-border);
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            border-top: 3px solid var(--ss-accent);
        }

        .ss-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 12px;
        }

        .ss-player-card {
            background: var(--ss-panel);
            border: 1px solid var(--ss-border);
            border-radius: 8px;
            padding: 16px;
            min-height: 150px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            border-top: 3px solid transparent;
        }

        .ss-player-card:hover {
            border-top-color: var(--ss-accent);
            box-shadow: 0 9px 20px rgba(16, 24, 40, 0.09);
        }

        .ss-player-name {
            color: var(--ss-ink);
            font-size: 1.05rem;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 6px;
        }

        .ss-muted {
            color: var(--ss-muted);
            font-size: 0.9rem;
            line-height: 1.45;
        }

        .ss-card-stat {
            margin-top: 14px;
            color: var(--ss-accent);
            font-weight: 850;
            font-size: 1.08rem;
        }

        .ss-profile-hero {
            border-radius: 8px;
            background:
                radial-gradient(circle at right, rgba(255, 255, 255, 0.14), transparent 34%),
                linear-gradient(135deg, rgba(17, 17, 17, 0.99), rgba(156, 2, 20, 0.94));
            color: #ffffff;
            padding: 28px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .ss-profile-hero .ss-title,
        .ss-profile-hero h1 {
            color: #ffffff;
        }

        .ss-table-note {
            color: var(--ss-muted);
            font-size: 0.86rem;
            margin-top: -4px;
            margin-bottom: 12px;
        }

        .ss-visualisation-note {
            color: var(--ss-muted);
            font-size: 0.84rem;
            line-height: 1.35;
            margin: 4px 0 10px;
        }

        .ss-visualisation-note strong {
            color: var(--ss-ink);
            font-weight: 750;
        }

        .cafc-badge-fallback {
            display: inline-grid;
            place-items: center;
            background: var(--ss-accent);
            color: #ffffff;
            font-weight: 900;
            letter-spacing: 0.02em;
        }

        @media (max-width: 760px) {
            .ss-hero-inner {
                align-items: flex-start;
                flex-direction: column-reverse;
                gap: 18px;
            }

            .ss-hero {
                padding: 26px 22px;
            }

            .ss-hero-badge {
                width: 78px;
                height: 78px;
            }

            .ss-logo {
                width: 56px;
                height: 56px;
            }

            .ss-brand-title {
                font-size: 1.18rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def clean_position(position: object) -> str:
    text = "" if position is None else str(position)
    if not text or text.lower() == "nan":
        return "Unknown position"
    parts = [part.strip().replace("_", " ").title() for part in text.split(",")]
    return ", ".join(part for part in parts if part) or "Unknown position"


def format_number(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if number.is_integer():
        return f"{number:,.0f}"
    return f"{number:,.{digits}f}"


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join([c * 2 for c in hex_str])
    return int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)


def get_team_color(team_name: object) -> str:
    """Return the primary home kit color for a given team name."""
    text = str(team_name).strip().lower()
    text = text.replace(".", "").replace("&", "and")
    text = " ".join(text.split())

    for prefix in ["fc ", "afc "]:
        if text.startswith(prefix):
            text = text[len(prefix):]

    for suffix in [" fc", " afc", " football club"]:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    
    key = text.strip()

    # Mapping based on common club home colors
    colors = {
        "barnsley": "#D71920",
        "birmingham": "#0000FF",
        "birmingham city": "#0000FF",
        "blackburn": "#0055A3",
        "blackburn rovers": "#0055A3",
        "blackpool": "#F68712",
        "bolton": "#FFFFFF",
        "bolton wanderers": "#FFFFFF",
        "bristol city": "#BC161C",
        "bristol rovers": "#0000FF",
        "bromley": "#FFFFFF",
        "burnley": "#6C1D45",
        "burton": "#FFE600",
        "burton albion": "#FFE600",
        "cambridge": "#FFD200",
        "cambridge united": "#FFD200",
        "cardiff": "#0000FF",
        "cardiff city": "#0000FF",
        "charlton": CHARLTON_RED,
        "charlton athletic": CHARLTON_RED,
        "chelsea": "#034694",
        "coventry": "#87CEEB",
        "coventry city": "#87CEEB",
        "crawley": "#FF0000",
        "crawley town": "#FF0000",
        "derby": "#FFFFFF",
        "derby county": "#FFFFFF",
        "exeter": "#FF0000",
        "exeter city": "#FF0000",
        "huddersfield": "#0072CE",
        "huddersfield town": "#0072CE",
        "hull": "#FFA500",
        "hull city": "#FFA500",
        "ipswich": "#0033FF",
        "ipswich town": "#0033FF",
        "leeds": "#FFFFFF",
        "leeds united": "#FFFFFF",
        "leicester": "#003090",
        "leicester city": "#003090",
        "leyton orient": "#FF0000",
        "lincoln": "#D71920",
        "lincoln city": "#D71920",
        "luton": "#F78F1E",
        "luton town": "#F78F1E",
        "mansfield": "#FFD200",
        "mansfield town": "#FFD200",
        "middlesbrough": "#D71920",
        "millwall": "#004e92",
        "northampton": "#7E2432",
        "northampton town": "#7E2432",
        "norwich": "#FFF200",
        "norwich city": "#FFF200",
        "oxford": "#F7E919",
        "oxford united": "#F7E919",
        "peterborough": "#005CAB",
        "peterborough united": "#005CAB",
        "plymouth": "#00563F",
        "plymouth argyle": "#00563F",
        "portsmouth": "#001BFF",
        "preston": "#FFFFFF",
        "preston north end": "#FFFFFF",
        "qpr": "#0000FF",
        "queens park rangers": "#0000FF",
        "reading": "#0000FF",
        "rotherham": "#D71920",
        "rotherham united": "#D71920",
        "sheffield united": "#EE272C",
        "sheffield wednesday": "#0000FF",
        "shrewsbury": "#0000FF",
        "shrewsbury town": "#0000FF",
        "southampton": "#D71920",
        "stevenage": "#FF0000",
        "stockport": "#0000FF",
        "stockport county": "#0000FF",
        "stoke": "#E03A3E",
        "stoke city": "#E03A3E",
        "swansea": "#FFFFFF",
        "swansea city": "#FFFFFF",
        "watford": "#FBEE23",
        "watford fc": "#FBEE23",
        "west brom": "#122F67",
        "west bromwich albion": "#122F67",
        "west ham": "#7A263A",
        "west ham united": "#7A263A",
        "wigan": "#0000FF",
        "wigan athletic": "#0000FF",
        "wrexham": "#CF0C30",
        "wycombe": "#0000FF",
        "wycombe wanderers": "#0000FF",
    }
    
    return colors.get(key, CHARLTON_RED)


VISUALISATION_FULLSCREEN_NOTE = (
    "For best readability, open charts in full-screen mode using the expand icon "
    "in the top-right corner of each visualisation."
)


def visualisation_fullscreen_note() -> None:
    st.markdown(
        f'<div class="ss-visualisation-note"><strong>Viewing tip:</strong> {esc(VISUALISATION_FULLSCREEN_NOTE)}</div>',
        unsafe_allow_html=True,
    )
