import html
import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st


APP_ROOT = Path(__file__).resolve().parents[1]


def data_refresh_control() -> None:
    """A "Refresh data" button for the Data Hub that clears every cached query."""
    note, button = st.columns([4, 1], vertical_alignment="center")
    note.caption("Queries are cached for up to an hour. Refresh to read CAFC_DB again now.")
    if button.button("Refresh data", icon=":material/refresh:", key="data_refresh_control", width="stretch"):
        st.cache_data.clear()
        st.toast("Cache cleared; reloading from CAFC_DB.")
        st.rerun()
ASSETS_DIR = APP_ROOT / "assets"
BADGE_PATH = ASSETS_DIR / "charlton_badge.png"

# Palette shared with the Charlton Academy Graduates dashboard.
CHARLTON_RED = "#D4021D"
CHARLTON_DEEP_RED = "#A00016"
CHARLTON_BLACK = "#111111"
CHARLTON_CHARCOAL = "#16181D"
CHARLTON_WHITE = "#ffffff"
CHARLTON_OFF_WHITE = "#F5F3F0"
CHARLTON_SURFACE = "#FFFDFB"
CHARLTON_BORDER = "#E6E1DB"
CHARLTON_GRID = "#ECE8E3"
CHARLTON_MUTED = "#747A85"
CHARLTON_INK_2 = "#4A4F59"
AXIS_TEXT = CHARLTON_INK_2
AXIS_LINE = "#D6D0C8"
AXIS_FONT_FAMILY = "Inter, Segoe UI, Arial, sans-serif"
DISPLAY_FONT_FAMILY = "Archivo Narrow, Archivo, Arial Narrow, sans-serif"


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


THEME_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Archivo+Narrow:wght@600;700&family=Archivo:wdth,wght@62..100,600..800&family=Inter:wght@400;500;600;700&display=swap");

:root {
    --ground: #F5F3F0;
    --surface: #FFFDFB;
    --surface-2: #F8F5F2;
    --ink: #16181D;
    --ink-2: #4A4F59;
    --muted: #747A85;
    --rule: #E6E1DB;
    --grid: #ECE8E3;
    --red: #D4021D;
    --red-deep: #A00016;
    --black: #111111;
    --shadow: 0 1px 2px rgb(20 20 30 / 6%), 0 6px 24px rgb(20 20 30 / 6%);
    --display: "Archivo", "Arial Narrow", system-ui, sans-serif;
    --body: "Inter", "Segoe UI", system-ui, sans-serif;

    /* Older page styles reference these names; point them at the palette. */
    --ss-bg: var(--ground);
    --ss-ink: var(--ink);
    --ss-muted: var(--muted);
    --ss-border: var(--rule);
    --ss-panel: var(--surface);
    --ss-panel-soft: var(--surface-2);
    --ss-dark: var(--black);
    --ss-accent: var(--red);
    --ss-accent-dark: var(--red-deep);
    --ss-accent-soft: #FCEDEE;
    --ss-shadow: var(--shadow);
}

.stApp { background: var(--ground); color: var(--ink); }
.num { font-variant-numeric: tabular-nums; }

/* ---- chrome --------------------------------------------------------------- */
[data-testid="stHeader"] {
    background: color-mix(in srgb, var(--ground) 88%, transparent);
    backdrop-filter: blur(8px);
}
[data-testid="stMainBlockContainer"], .main .block-container {
    max-width: 1440px;
    padding-top: 2.4rem;
    padding-bottom: 4rem;
}
[data-testid="stSidebar"] { border-right: 0; }
/* Crest centred in the sidebar header; the collapse button sits on top at the right. */
[data-testid="stSidebarHeader"] {
    position: relative; justify-content: center; padding-top: 22px; padding-bottom: 10px;
}
[data-testid="stLogoSpacer"] { display: none; }
[data-testid="stSidebarCollapseButton"] { position: absolute; right: 10px; top: 14px; }
[data-testid="stSidebarHeader"] img[data-testid="stSidebarLogo"], [data-testid="stSidebarHeader"] img {
    height: 72px !important; max-height: 72px !important; width: auto;
    filter: drop-shadow(0 4px 10px rgb(0 0 0 / 45%));
}
[data-testid="stSidebarNav"] a { border-radius: 8px; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: var(--red); }
[data-testid="stSidebarNav"] a[aria-current="page"] span { color: #fff !important; font-weight: 600; }
[data-testid="stNavSectionHeader"] {
    font-size: 11px !important; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #8A8F98 !important;
}

/* ---- masthead ------------------------------------------------------------- */
.cafc-mast {
    position: relative;
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
    gap: 18px 36px;
    margin: 0 0 18px;
    padding: 24px 28px;
    border-radius: 16px;
    border-bottom: 6px solid var(--black);
    background: var(--red);
    background-image:
        repeating-linear-gradient(115deg, rgb(255 255 255 / 6%) 0 2px, transparent 2px 16px),
        linear-gradient(115deg, transparent 64%, var(--red-deep) 64%);
    color: #fff;
    box-shadow: var(--shadow);
}
.cafc-brand { display: flex; align-items: center; gap: 20px; min-width: 0; }
.cafc-badge {
    width: 84px; height: 84px; flex: none; object-fit: contain;
    filter: drop-shadow(0 6px 14px rgb(0 0 0 / 35%));
}
.cafc-eyebrow {
    margin: 0 0 6px; font: 600 12px/1 var(--body);
    letter-spacing: 0.14em; text-transform: uppercase; opacity: 0.85;
}
.stApp .cafc-mast h1 {
    margin: 0; padding: 0; color: #fff;
    font: 800 clamp(32px, 4.4vw, 54px)/0.92 var(--display); font-stretch: 70%;
    text-transform: uppercase; letter-spacing: 0.005em; text-wrap: balance;
}
.cafc-mast .cafc-sub { margin: 10px 0 0; max-width: 64ch; opacity: 0.9; font-size: 15px; line-height: 1.45; }
.cafc-facts { display: flex; flex-wrap: wrap; gap: 8px; margin: 0; }
.cafc-facts div { padding: 8px 12px; border-radius: 10px; background: rgb(0 0 0 / 18%); min-width: 92px; }
.cafc-facts dt { font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.8; }
.cafc-facts dd { margin: 2px 0 0; font: 700 18px/1.1 var(--display); font-stretch: 85%; }

/* "About this data" sits under the masthead instead of filling it with prose. */
.cafc-about {
    margin: -6px 0 18px; border: 1px solid var(--rule); border-radius: 12px;
    background: var(--surface); color: var(--ink-2); font-size: 13.5px;
}
.cafc-about summary {
    cursor: pointer; list-style: none; padding: 10px 14px;
    font-weight: 600; color: var(--ink-2); display: flex; align-items: center; gap: 8px;
}
.cafc-about summary::-webkit-details-marker { display: none; }
.cafc-about summary::before { content: "i"; display: inline-grid; place-items: center; width: 18px; height: 18px;
    border-radius: 50%; border: 1px solid var(--muted); font: 700 11px/1 var(--body); color: var(--muted); }
.cafc-about[open] summary { border-bottom: 1px solid var(--rule); }
.cafc-about .cafc-about-body { padding: 10px 14px 12px; display: grid; gap: 8px; line-height: 1.5; }
.cafc-about b { color: var(--ink); }
.cafc-about .cafc-warn { color: #8A5A00; }

/* ---- section headings ------------------------------------------------------ */
.cafc-section {
    display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 4px 16px;
    margin: 26px 0 10px;
}
.cafc-section h2, .stApp .cafc-section h2 {
    margin: 0; padding: 0; color: var(--ink);
    font: 700 21px/1.1 var(--display); font-stretch: 82%;
    text-transform: uppercase; letter-spacing: 0.02em;
}
.cafc-section p { margin: 0; color: var(--muted); font-size: 13px; }
/* Section labels other modules still render. */
.stApp .ma-section, .stApp .pa-section, .stApp .ta-section, .stApp .ss-section-label {
    margin: 26px 0 10px; padding: 0; border: 0; background: none; color: var(--ink);
    font: 700 21px/1.1 var(--display); font-stretch: 82%;
    text-transform: uppercase; letter-spacing: 0.02em;
}
.stApp .ma-section::before, .stApp .pa-section::before, .stApp .ta-section::before,
.stApp .ma-section::after, .stApp .pa-section::after, .stApp .ta-section::after { display: none; }

/* st.header / st.subheader take the dashboard's card-title style. */
.stApp [data-testid="stHeadingWithActionElements"] h2,
.stApp [data-testid="stHeadingWithActionElements"] h3 {
    font-family: var(--display); font-stretch: 82%; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.02em; color: var(--ink);
}
.stApp [data-testid="stHeadingWithActionElements"] h2 { font-size: 23px; }
.stApp [data-testid="stHeadingWithActionElements"] h3 { font-size: 19px; }

/* ---- key figures ------------------------------------------------------------ */
.cafc-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 0 0 6px; }
.cafc-tile {
    padding: 14px 16px; background: var(--surface); border: 1px solid var(--rule);
    border-radius: 14px; box-shadow: var(--shadow); min-width: 0;
}
.cafc-tile .label { font-size: 12.5px; color: var(--ink-2); font-weight: 500; }
.cafc-tile .value {
    margin-top: 4px; font: 700 28px/1.1 var(--body); letter-spacing: -0.01em;
    font-variant-numeric: tabular-nums; overflow-wrap: anywhere;
}
.cafc-tile .note { margin-top: 4px; font-size: 12.5px; color: var(--muted); }
[data-testid="stColumn"] .cafc-tile { height: 100%; margin-bottom: 12px; }
.cafc-tile .value.text { font-size: 17px; line-height: 1.25; letter-spacing: 0; }
.cafc-tile.dark { background: var(--black); color: #fff; border-color: var(--black); }
.cafc-tile.dark .label, .cafc-tile.dark .note { color: inherit; opacity: 0.8; }
.cafc-tile.hero { background: var(--black); color: #fff; border-color: var(--black); border-top: 4px solid var(--red); }
.cafc-tile.hero .label { color: inherit; opacity: 0.8; }
.cafc-tile.hero .value { font-size: 44px; }
.cafc-tile.hero .note { color: inherit; opacity: 0.75; }

/* st.metric renders as the same tile. */
div[data-testid="stMetric"] {
    padding: 14px 16px; background: var(--surface); border: 1px solid var(--rule);
    border-radius: 14px; box-shadow: var(--shadow);
}
div[data-testid="stMetricLabel"] p { font-size: 12.5px; color: var(--ink-2); font-weight: 500; }
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {
    font: 700 clamp(22px, 2vw, 28px)/1.1 var(--body) !important;
    letter-spacing: -0.01em; color: var(--ink); font-variant-numeric: tabular-nums;
}
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] > div {
    overflow: visible !important; text-overflow: clip !important; white-space: normal !important;
    overflow-wrap: anywhere;
}

/* ---- panels ------------------------------------------------------------------ */
.cafc-card, .stApp .ss-panel {
    background: var(--surface); border: 1px solid var(--rule); border-radius: 14px; box-shadow: var(--shadow);
}
/* ui.card() containers; Streamlit gives bordered and plain containers the same markup. */
div[data-testid="stVerticalBlock"][class*="st-key-cafc_card_"] {
    background: var(--surface); border: 1px solid var(--rule); border-radius: 14px;
    box-shadow: var(--shadow); padding: 16px 18px;
}
div[data-testid="stPlotlyChart"] {
    background: var(--surface); border: 1px solid var(--rule); border-radius: 14px;
    box-shadow: var(--shadow); padding: 6px 8px 4px; overflow: hidden;
}
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
    border-radius: 12px; overflow: hidden; box-shadow: var(--shadow);
}
div[data-testid="stExpander"] details {
    background: var(--surface); border: 1px solid var(--rule) !important; border-radius: 12px !important;
}
div[data-testid="stExpander"] summary p { font-weight: 600; }

/* ---- controls ------------------------------------------------------------------ */
[data-testid="stWidgetLabel"] p {
    font-size: 11px !important; font-weight: 600 !important; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--muted) !important;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #9A9EA6 !important; }
.stTextInput input, .stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div,
.stDateInput input { background: var(--surface); }
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div { background: #1C1C1F; }

div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button,
div[data-testid="stFormSubmitButton"] > button, a[data-testid="stPageLink-NavLink"] {
    min-height: 36px; border-radius: 8px; font-weight: 600;
}
div[data-testid="stButton"] > button[kind="secondary"], div[data-testid="stDownloadButton"] > button[kind="secondary"] {
    background: var(--surface); border: 1px solid var(--rule); color: var(--ink);
}
div[data-testid="stButton"] > button[kind="secondary"]:hover { border-color: var(--red); color: var(--red); }
[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"] {
    background: transparent; border-color: #2A2A2E; color: #C3C7CE;
}

/* Tabs as the dashboard's pill switcher. */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 2px; padding: 3px; width: fit-content; max-width: 100%; overflow-x: auto;
    border-radius: 999px; background: var(--surface); border: 1px solid var(--rule);
}
div[data-testid="stTabs"] button[role="tab"] {
    height: 34px; padding: 0 16px; border-radius: 999px; border: 0;
}
div[data-testid="stTabs"] button[role="tab"] p { font-weight: 600; color: var(--ink-2); font-size: 14px; }
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { background: var(--red); }
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p { color: #fff; }
div[data-testid="stTabs"] [data-baseweb="tab-highlight"], div[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }

/* ---- module card grids (match / player / team analysis) -------------------------- */
.stApp .ma-card, .stApp .pa-card, .stApp .ta-card, .stApp .ss-player-card {
    background: var(--surface); border: 1px solid var(--rule); border-top: 1px solid var(--rule);
    border-radius: 14px; box-shadow: var(--shadow);
}
.stApp .ma-card-icon, .stApp .pa-card-icon, .stApp .ta-card-icon { color: var(--red); background: none; }
.stApp .ma-card-title, .stApp .pa-card-title, .stApp .ta-card-title {
    font: 700 17px/1.15 var(--display); font-stretch: 85%; text-transform: uppercase; letter-spacing: 0.02em;
}

/* ---- sticky filter bar: wrap controls in st.container(key="cafc_filter_bar") ---- */
/* Sticky goes on Streamlit's wrapper: the keyed block itself is only as tall as its parent. */
div[data-testid="stLayoutWrapper"]:has(> .st-key-cafc_filter_bar) { position: sticky; top: 3.75rem; z-index: 20; }
.st-key-cafc_filter_bar, [class*="st-key-cafc_controls_"] {
    margin: 0 0 4px; padding: 12px 16px 6px;
    background: color-mix(in srgb, var(--ground) 97%, transparent);
    backdrop-filter: blur(8px);
    border: 1px solid var(--rule); border-radius: 14px;
}

/* ---- leaderboard (horizontal bars) ---------------------------------------------- */
.cafc-lb { display: grid; gap: 9px; padding: 4px 2px 6px; }
.cafc-lb-row { display: grid; grid-template-columns: 26px minmax(120px, 210px) 1fr; gap: 12px; align-items: center; }
.cafc-lb-rank { font: 700 13px/1 var(--body); color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }
.cafc-lb-who { display: grid; line-height: 1.2; min-width: 0; }
.cafc-lb-who b { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.cafc-lb-who small { color: var(--muted); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.cafc-lb-track { display: flex; align-items: center; gap: 8px; min-width: 0; }
.cafc-lb-bar { display: block; flex: none; height: 18px; min-width: 2px; border-radius: 0 4px 4px 0; background: var(--red); }
.cafc-lb-row.muted .cafc-lb-bar { background: #C9C3BB; }
.cafc-lb-val { font-size: 12.5px; color: var(--ink-2); white-space: nowrap; font-variant-numeric: tabular-nums; }

/* ---- link cards -------------------------------------------------------------- */
.cafc-link-title {
    margin: 2px 0 4px; font: 700 19px/1.1 var(--display); font-stretch: 85%;
    text-transform: uppercase; letter-spacing: 0.02em; color: var(--ink);
}
.cafc-link-body { color: var(--muted); font-size: 13.5px; line-height: 1.45; min-height: 3.9em; }

/* st.segmented_control as the dashboard's .seg switcher */
div[data-testid="stButtonGroup"] button { min-height: 32px; }
div[data-testid="stButtonGroup"] button[kind$="Active"], div[data-testid="stButtonGroup"] button[aria-checked="true"] {
    background: var(--ink); color: var(--surface); border-color: var(--ink);
}

.cafc-muted, .ss-muted { color: var(--muted); font-size: 13.5px; line-height: 1.45; }
.cafc-summary { margin: 0 0 4px; color: var(--ink-2); max-width: 95ch; }
.cafc-summary strong { color: var(--ink); }
.cafc-badge-fallback {
    display: inline-grid; place-items: center; background: var(--red); color: #fff;
    font-weight: 900; letter-spacing: 0.02em;
}

@media (max-width: 760px) {
    .cafc-mast { padding: 18px; border-radius: 12px; }
    .cafc-badge { width: 60px; height: 60px; }
    .cafc-brand { gap: 14px; }
    .cafc-tile.hero .value { font-size: 36px; }
}
</style>
"""


def apply_statsearch_theme() -> None:
    """Shared visual treatment, modelled on the Charlton Academy Graduates dashboard."""
    _patch_plotly_chart_axis_style()
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def page_header(
    title: str,
    subtitle: str = "",
    *,
    eyebrow: str = "Charlton Athletic",
    facts: list[tuple[str, object]] | None = None,
    basis: str | None = None,
    limitation: str | None = None,
) -> None:
    """Red crest masthead with optional stat chips and a collapsed data note."""
    facts_html = "".join(
        f"<div><dt>{esc(label)}</dt><dd class='num'>{esc(value)}</dd></div>" for label, value in facts or []
    )
    st.markdown(
        '<header class="cafc-mast">'
        '<div class="cafc-brand">'
        f'{badge_html("cafc-badge")}'
        "<div>"
        f'<p class="cafc-eyebrow">{esc(eyebrow)}</p>'
        f"<h1>{esc(title)}</h1>"
        + (f'<p class="cafc-sub">{esc(subtitle)}</p>' if subtitle else "")
        + "</div></div>"
        + (f'<dl class="cafc-facts">{facts_html}</dl>' if facts_html else "")
        + "</header>",
        unsafe_allow_html=True,
    )
    about_data(basis, limitation)


def about_data(basis: str | None, limitation: str | None = None) -> None:
    """Methodology and caveats, collapsed so the page leads with the analysis."""
    if not basis and not limitation:
        return
    parts = []
    if basis:
        parts.append(f"<div><b>Data basis.</b> {esc(basis)}</div>")
    if limitation:
        parts.append(f'<div class="cafc-warn"><b>Limitation.</b> {esc(limitation)}</div>')
    parts.append("<div>Charts open full screen from the expand icon at their top right.</div>")
    st.markdown(
        '<details class="cafc-about"><summary>About this data</summary>'
        f'<div class="cafc-about-body">{"".join(parts)}</div></details>',
        unsafe_allow_html=True,
    )


def section_heading(label: str, note: str | None = None) -> None:
    st.markdown(
        f'<div class="cafc-section"><h2>{esc(label)}</h2>'
        + (f"<p>{esc(note)}</p>" if note else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def card(name: str):
    """A white panel container; ``name`` must be unique on the page."""
    return st.container(border=True, key=f"cafc_card_{name}")


def filter_bar():
    """Container for a page's main controls; it sticks under the header on scroll."""
    return st.container(key="cafc_filter_bar")


def controls_panel(name: str = "main"):
    """The filter bar's look without sticking, for control sets several rows tall."""
    return st.container(key=f"cafc_controls_{name}")


def leaderboard(rows: list[dict[str, object]], highlight: str | None = None) -> None:
    """Ranked horizontal bars, as in the academy dashboard's "Minutes by graduate".

    Each row is ``{"name", "sub", "value" (number), "text" (formatted)}``. Bars
    scale to the largest value; rows not matching ``highlight`` (a substring of
    ``sub``) are drawn muted when a highlight is given.
    """
    values = [float(row["value"]) for row in rows]
    top = max(values, default=0.0) or 1.0
    if highlight is not None and not any(highlight.casefold() in str(row.get("sub", "")).casefold() for row in rows):
        highlight = None
    html_rows = []
    for rank, row in enumerate(rows, start=1):
        share = max(float(row["value"]), 0.0) / top
        muted = highlight is not None and highlight.casefold() not in str(row.get("sub", "")).casefold()
        html_rows.append(
            f'<div class="cafc-lb-row{" muted" if muted else ""}">'
            f'<span class="cafc-lb-rank">{rank}</span>'
            f'<span class="cafc-lb-who"><b>{esc(row["name"])}</b><small>{esc(row.get("sub", ""))}</small></span>'
            # Leave room beside the longest bar for its value label.
            f'<span class="cafc-lb-track"><i class="cafc-lb-bar" style="width:calc((100% - 76px) * {share:.4f})"></i>'
            f'<span class="cafc-lb-val">{esc(row["text"])}</span></span>'
            "</div>"
        )
    st.markdown(f'<div class="cafc-lb">{"".join(html_rows)}</div>', unsafe_allow_html=True)


def figure_tile_html(
    label: object,
    value: object,
    note: object | None = None,
    *,
    text_value: bool = False,
    dark: bool = False,
) -> str:
    """One figure tile's markup; ``text_value`` sizes a word-valued tile (a name, a team) down."""
    klass = "cafc-tile dark" if dark else "cafc-tile"
    value_class = "value text" if text_value else "value"
    note_html = f'<div class="note">{esc(note)}</div>' if note not in (None, "") else ""
    return (
        f'<div class="{klass}"><div class="label">{esc(label)}</div>'
        f'<div class="{value_class}">{esc(value)}</div>{note_html}</div>'
    )


def figure_tile(
    label: object,
    value: object,
    note: object | None = None,
    *,
    text_value: bool = False,
    dark: bool = False,
) -> None:
    st.markdown(figure_tile_html(label, value, note, text_value=text_value, dark=dark), unsafe_allow_html=True)


def key_figures(tiles: list[dict[str, object]], hero: bool = True) -> None:
    """A row of figure tiles; the first is the dark hero tile when ``hero``.

    Each tile is ``{"label": ..., "value": ..., "note": ...}`` (note optional).
    """
    html_tiles = []
    for index, tile in enumerate(tiles):
        klass = "cafc-tile hero" if hero and index == 0 else "cafc-tile"
        note = tile.get("note")
        html_tiles.append(
            f'<div class="{klass}"><div class="label">{esc(tile.get("label"))}</div>'
            f'<div class="value">{esc(tile.get("value"))}</div>'
            + (f'<div class="note">{esc(note)}</div>' if note else "")
            + "</div>"
        )
    st.markdown(f'<div class="cafc-tiles">{"".join(html_tiles)}</div>', unsafe_allow_html=True)


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
