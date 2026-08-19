# =============================================================================
# FIELD TILT / TERRITORY - final-third and bypassing territory proxy
# =============================================================================
import streamlit as st

from utils import team_analysis as ta


ta.page_header(
    "Field Tilt / Territory",
    "Compare teams by how often they reach the final third and bypass opponents.",
    ta.TEAM_SOURCE,
    "The app does not currently have field-zone possession share, so territory is shown through final-third passing and bypassed-opponent proxies.",
)

season = ta.select_season("teams", key="territory_season")
teams = ta.load_team_data(season)
if teams.empty:
    st.warning("No team data is available for this season.")
    st.stop()

team_name = ta.team_selector(teams, key="territory_team")
scores = ta.style_scores(teams)
territory_row = scores[scores["Team"] == team_name].iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Territory Proxy", f"{territory_row['Territory']:.0f}th percentile")
c2.metric("Passes to Final 3rd /90", f"{teams.loc[teams['Team'] == team_name, 'Passes to Final 3rd /90'].iloc[0]:.2f}")
c3.metric("Bypassed Opponents /90", f"{teams.loc[teams['Team'] == team_name, 'Bypassed Opponents /90'].iloc[0]:.2f}")

ta.section_heading("Territory proxy map")
st.plotly_chart(
    ta.metric_scatter(
        teams,
        x="Passes to Final 3rd /90",
        y="Bypassed Opponents /90",
        selected=team_name,
        size="Pass %",
        title="Territory proxy: final-third access vs opponent bypassing",
    ),
    width="stretch",
)

ta.section_heading("Territory proxy table")
territory_table = scores[["Team", "Territory", "Progression", "Control Proxy"]].sort_values("Territory", ascending=False)
st.dataframe(territory_table, width="stretch", hide_index=True)
