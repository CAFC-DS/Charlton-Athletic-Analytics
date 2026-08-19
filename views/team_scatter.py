# =============================================================================
# TEAM SCATTER GRAPHS - compare any two available team metrics
# =============================================================================
import streamlit as st

from utils import team_analysis as ta


ta.page_header(
    "Team Scatter Graphs",
    "Compare every team on two selected metrics, with the selected team highlighted.",
    ta.TEAM_STYLE_SOURCE,
)

season = ta.select_season("players", key="team_scatter_season")
teams = ta.load_team_style_data(season)
if teams.empty:
    st.warning("No team data is available for this season.")
    st.stop()

team_name = ta.team_selector(teams, key="team_scatter_team")
metrics = ta.metric_columns(teams)

ta.section_heading("Scatter controls")
st.caption("Use the dropdowns to choose the two team metrics and optional bubble-size metric.")
c1, c2, c3 = st.columns(3)
x_metric = c1.selectbox("X axis", metrics, index=0, key="team_scatter_x")
y_metric = c2.selectbox("Y axis", metrics, index=1 if len(metrics) > 1 else 0, key="team_scatter_y")
size_metric = c3.selectbox("Bubble size", ["None"] + metrics, index=0, key="team_scatter_size")

ta.section_heading("Metric comparison")
st.plotly_chart(
    ta.metric_scatter(
        teams,
        x=x_metric,
        y=y_metric,
        selected=team_name,
        size=None if size_metric == "None" else size_metric,
        title=f"{x_metric} vs {y_metric}",
    ),
    width="stretch",
)

st.caption(
    "Hover a team badge for the exact values. If a bubble-size metric is selected, badge size reflects that metric. "
    "Teams with identical x/y values are slightly separated visually so each badge can be hovered."
)
