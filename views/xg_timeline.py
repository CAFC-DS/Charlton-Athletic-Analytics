# =============================================================================
# XG TIMELINE - real Impect shot xG timeline
# =============================================================================
import streamlit as st

from utils import data
from utils import match_analysis as ma
from utils import pitch
from utils import ui


def _xg_timeline_css() -> None:
    st.markdown(
        """
        <style>
        .xgt-fixture-metric {
            background: var(--ss-panel);
            border: 1px solid var(--ss-border);
            border-radius: 8px;
            border-top: 3px solid var(--ss-accent);
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-height: 102px;
            padding: 14px 16px;
        }

        .xgt-fixture-metric-label {
            color: var(--ss-muted);
            font-size: 0.875rem;
            font-weight: 650;
            line-height: 1.25;
            margin-bottom: 18px;
        }

        .xgt-fixture-metric-value {
            color: var(--ss-ink);
            display: block;
            font-size: clamp(0.68rem, 0.78vw, 0.84rem);
            font-weight: 500;
            letter-spacing: -0.01em;
            line-height: 1.2;
            overflow: hidden !important;
            text-overflow: clip !important;
            white-space: nowrap !important;
        }

        .xgt-standard-metric {
            background: var(--ss-panel);
            border: 1px solid var(--ss-border);
            border-radius: 8px;
            border-top: 3px solid var(--ss-accent);
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-height: 102px;
            padding: 14px 16px;
        }

        .xgt-standard-metric-label {
            color: var(--ss-muted);
            font-size: 0.875rem;
            font-weight: 650;
            line-height: 1.25;
            margin-bottom: 14px;
        }

        .xgt-standard-metric-value {
            color: var(--ss-ink);
            display: block;
            font-size: clamp(1.75rem, 2.2vw, 2.2rem);
            font-weight: 400;
            letter-spacing: -0.03em;
            line-height: 1.1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: object) -> None:
    st.markdown(
        f"""
        <div class="xgt-standard-metric">
            <div class="xgt-standard-metric-label">{ui.esc(label)}</div>
            <div class="xgt-standard-metric-value">{ui.esc(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


ma.page_header(
    "xG Timeline",
    "Track cumulative shot xG by minute for both teams in a selected fixture.",
    "CAFC_DB Impect provider events supply shot timestamps, teams, players, outcomes and xG for event-covered seasons.",
)
_xg_timeline_css()

season = ma.select_match_season(key="xg_timeline_season")
matches = ma.load_matches(season)
if matches.empty:
    st.warning("No match data is available for this season.")
    st.stop()

match_row = ma.match_selector(matches, key="xg_timeline_match")
events = data.load_match_events(
    season=season,
    match_id=match_row.get("MatchId"),
    limit=20000,
)
shots = events.dropna(subset=["Shot xG"]).copy()

ma.section_heading("Selected fixture")
metric_cols = st.columns(5)
with metric_cols[0]:
    st.markdown(
        f"""
        <div class="xgt-fixture-metric">
            <div class="xgt-fixture-metric-label">Fixture</div>
            <div class="xgt-fixture-metric-value">{ui.esc(match_row.get("Match", "Unknown"))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with metric_cols[1]:
    _metric_card("Score", f"{match_row['Home Goals']:.0f} - {match_row['Away Goals']:.0f}")
with metric_cols[2]:
    _metric_card("Shots", len(shots))
with metric_cols[3]:
    _metric_card("Total xG", ma.metric_value(shots["Shot xG"].sum() if not shots.empty else 0, "Shot xG"))
with metric_cols[4]:
    _metric_card("Shot teams", shots["Team"].nunique() if not shots.empty else 0)

ma.section_heading("Cumulative xG timeline")
if shots.empty:
    st.info("No shot xG rows are available for this selected match in the connected Impect event feed.")
    st.plotly_chart(ma.scoreline_chart(match_row), width="stretch")
else:
    st.plotly_chart(pitch.xg_timeline(shots, "Cumulative xG by minute", end_minute=events["Minute"].max()), width="stretch")

ma.section_heading("Shot xG table")
display_cols = ma.available_columns(
    shots,
    ["Minute", "Team", "Player", "Action", "Body Part", "Result", "Shot xG", "Post-Shot xG", "Shot Distance", "Shot Angle"],
)
if shots.empty:
    st.caption("No shot rows are available for this fixture.")
else:
    st.dataframe(shots[display_cols].sort_values(["Minute", "Team"]), width="stretch", hide_index=True)
