# =============================================================================
# TEAM DATA TABLE - filterable team metrics and ranks
# =============================================================================
import streamlit as st

from utils import team_analysis as ta


ta.page_header(
    "Team Data Table",
    "Inspect, sort and export the team metrics used across the Team Analysis pages.",
    ta.TEAM_SOURCE,
    visualisation_note=False,
)

season = ta.select_season("teams", key="team_table_season")
teams = ta.add_metric_ranks(ta.load_team_data(season))
if teams.empty:
    st.warning("No team data is available for this season.")
    st.stop()

metrics = ta.metric_columns(teams)
ta.section_heading("Table controls")
st.caption("Use the dropdowns to filter teams, choose a sorting metric and control which columns are visible.")
team_filter = st.multiselect("Teams", teams["Team"].tolist(), default=teams["Team"].tolist())
sort_metric = st.selectbox("Sort by", metrics, key="team_table_sort")
columns = st.multiselect("Columns", teams.columns.tolist(), default=["Team"] + metrics)

filtered = teams[teams["Team"].isin(team_filter)].sort_values(sort_metric, ascending=False)
display = filtered[columns] if columns else filtered

ta.section_heading("Filtered team table")
st.dataframe(display, width="stretch", hide_index=True)
st.caption(f"{len(display)} of {len(teams)} teams shown.")

st.download_button(
    "Download team_table.csv",
    display.to_csv(index=False),
    file_name="team_table.csv",
    mime="text/csv",
)
