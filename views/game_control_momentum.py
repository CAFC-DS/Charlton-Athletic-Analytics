# =============================================================================
# GAME CONTROL / MOMENTUM - event threat plus result momentum
# =============================================================================
import streamlit as st

from utils import data
from utils import match_analysis as ma
from utils import pitch


ma.page_header(
    "Game Control / Momentum",
    "Track selected-match cumulative threat from Impect event values, with season result momentum below.",
    "CAFC_DB Impect provider events supply pass PXT, shot PXT, shot xG and team xT values where available.",
)

season = ma.select_match_season(key="momentum_match_season")
matches = ma.load_matches(season)
if matches.empty:
    st.warning("No match data is available for this season.")
    st.stop()

match_row = ma.match_selector(matches, key="momentum_match")
team_name = ma.team_selector_for_match(match_row, key="momentum_team")
events = data.load_match_events(season=season, match_id=match_row.get("MatchId"), limit=9000)
team_matches = ma.team_match_rows(matches, team_name)

selected = ma.team_match_summary(match_row, team_name)
ma.section_heading("Selected fixture context")
metric_cols = st.columns(5)
metric_cols[0].metric("Result", selected["Result"])
metric_cols[1].metric("Points", ma.metric_value(selected["Points"]))
metric_cols[2].metric("Goal difference", ma.metric_value(selected["Goal Difference"]))
metric_cols[3].metric("Event rows", len(events))
metric_cols[4].metric("Opponent", str(selected["Opponent"]))

ma.section_heading("Selected-Match Event Threat")
if events.empty:
    st.info("No event-level threat values are available for this selected match. The result momentum view is still shown below where fixtures are available.")
else:
    st.plotly_chart(pitch.threat_timeline(events, "Cumulative Event Threat by Minute"), width="stretch")
    st.caption(
        "Shows each team's running event-threat total across the match. Threat is taken from available Impect event values "
        "(PXT pass, PXT shot, shot xG and team xT), so steeper rises indicate periods where a team created more dangerous actions."
    )

if team_matches.empty:
    st.info("No fixture rows are available for the selected team.")
    st.stop()

ma.section_heading(f"{team_name} Season Result Momentum")
st.plotly_chart(ma.momentum_chart(team_matches, f"{team_name}: Result Momentum"), width="stretch")
st.caption(
    "Bars show each match's goal difference. The line shows rolling points per match over a 5-match window "
    "(using the available fixtures until five matches have been played), giving a smoother view of form."
)

ma.section_heading("Match-by-Match Record")
display_cols = ma.available_columns(
    team_matches,
    [
        "Date",
        "Opponent",
        "Goals For",
        "Goals Against",
        "Goal Difference",
        "Team Result",
        "Points",
        "Rolling Points",
        "Cumulative Points",
    ],
)
st.dataframe(team_matches[display_cols], width="stretch", hide_index=True)
