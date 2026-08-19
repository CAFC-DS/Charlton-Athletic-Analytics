# =============================================================================
# SHOT PROFILE - event shot counts and goal-output proxy
# =============================================================================
import streamlit as st

from utils import team_analysis as ta


ta.page_header(
    "Shot Profile",
    "Review shot-related event counts where available, alongside goal-output rankings.",
    f"{ta.ACTION_SOURCE} Goal output uses {ta.TEAM_SOURCE}",
    "The current app does not have shot locations, shot body part or xG. This page is a volume/output view, not a shot-map model.",
)

team_season = ta.select_season("teams", key="shot_team_season")
teams = ta.load_team_data(team_season)
if teams.empty:
    st.warning("No team data is available for team selection.")
    st.stop()
team_name = ta.team_selector(teams, key="shot_team")

match_season = ta.select_season("matches", key="shot_action_season")
actions = ta.load_action_counts(match_season)
shot_summary = ta.action_summary(actions, ["SHOT", "GOAL", "PENALTY"])
breakdown = ta.team_action_breakdown(actions, team_name, ["SHOT", "GOAL", "PENALTY"])

ta.section_heading("Shot volume and goal output")
c1, c2 = st.columns([1.05, 1])
with c1:
    st.plotly_chart(
        ta.ranked_bar(teams, "Goals /90", selected=team_name, title="Goal-output ranking"),
        width="stretch",
    )
with c2:
    if shot_summary.empty:
        st.info("No shot-related action labels were found for this event season.")
    else:
        st.plotly_chart(ta.action_bar(shot_summary, team_name, "Shot-related event counts"), width="stretch")

ta.section_heading(f"{team_name} shot-action breakdown")
if breakdown.empty:
    st.caption("No matching shot, goal or penalty action labels are available for this team.")
else:
    st.dataframe(breakdown, width="stretch", hide_index=True)
