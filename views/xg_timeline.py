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
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: object) -> None:
    ui.figure_tile(label, value)


ma.page_header(
    "xG Timeline",
    "Track cumulative shot xG by minute for both teams in a selected fixture.",
    "CAFC_DB Impect provider events supply shot timestamps, teams, players, outcomes and xG for event-covered seasons.",
)
_xg_timeline_css()

season, matches, match_row, _ = ma.fixture_bar("xg_timeline_season", "xg_timeline_match")
events = data.load_match_events(
    season=season,
    match_id=match_row.get("MatchId"),
    limit=20000,
)
shots = events.dropna(subset=["Shot xG"]).copy()

ma.section_heading("Selected fixture")
metric_cols = st.columns(5)
with metric_cols[0]:
    ui.figure_tile("Fixture", match_row.get("Match", "Unknown"), text_value=True)
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
