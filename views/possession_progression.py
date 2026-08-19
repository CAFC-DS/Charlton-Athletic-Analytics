# =============================================================================
# POSSESSION & PROGRESSION - pass security against forward impact
# =============================================================================
import streamlit as st

from utils import team_analysis as ta


ta.page_header(
    "Possession & Progression",
    "Show whether teams combine secure passing with actions that move opponents out of the game.",
    ta.TEAM_SOURCE,
    "Pass % is a ball-security proxy, not possession share. Progression is based on bypassed opponents and final-third pass rates.",
)

season = ta.select_season("teams", key="possession_season")
teams = ta.load_team_data(season)
if teams.empty:
    st.warning("No team data is available for this season.")
    st.stop()

team_name = ta.team_selector(teams, key="possession_team")
style = ta.selected_team_style(teams, team_name)

c1, c2, c3 = st.columns(3)
c1.metric("Ball Security", f"{style['Ball Security']:.0f}th percentile")
c2.metric("Progression", f"{style['Progression']:.0f}th percentile")
c3.metric("Control Proxy", f"{style['Control Proxy']:.0f}th percentile")

ta.section_heading("Ball security vs progression")
st.plotly_chart(
    ta.metric_scatter(
        teams,
        x="Pass %",
        y="Bypassed Opponents /90",
        selected=team_name,
        size="Passes to Final 3rd /90",
        title="Ball security vs progression",
    ),
    width="stretch",
)

ta.section_heading("Control proxy table")
ranked = ta.style_scores(teams).sort_values("Control Proxy", ascending=False)
st.dataframe(ranked, width="stretch", hide_index=True)
