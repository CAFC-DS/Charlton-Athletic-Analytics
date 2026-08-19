# =============================================================================
# PLAYER DATA TABLE - filterable player metrics and ranks
# =============================================================================
import streamlit as st

from utils import player_analysis as pa


pa.page_header(
    "Player Data Table",
    "Inspect, filter, sort and export the player metrics used across the Player Analysis pages.",
    visualisation_note=False,
)

season = pa.select_season(key="player_table_season")
players = pa.add_metric_ranks(pa.load_player_data(season))
if players.empty:
    st.warning("No players are available for the selected season.")
    st.stop()

metrics = pa.metric_columns(players)
team_options = sorted(players["Team"].dropna().astype(str).unique()) if "Team" in players else []
position_options = sorted(players["_Position Display"].dropna().astype(str).unique())

pa.section_heading("Table controls")
st.caption("Use these controls to filter players, choose a sorting metric and control which columns are visible.")
c1, c2, c3 = st.columns(3)
teams = c1.multiselect("Teams", team_options, default=team_options)
positions = c2.multiselect("Positions", position_options, default=position_options)
sort_metric = c3.selectbox("Sort by", metrics, key="player_table_sort")
columns = st.multiselect("Columns", players.columns.tolist(), default=[col for col in ["Player", "Team", "Position", "Minutes"] + metrics if col in players.columns])

filtered = players.copy()
if teams and "Team" in filtered:
    filtered = filtered[filtered["Team"].astype(str).isin(teams)]
if positions:
    filtered = filtered[filtered["_Position Display"].astype(str).isin(positions)]
filtered = filtered.sort_values(sort_metric, ascending=False)
display = filtered[columns] if columns else filtered

pa.section_heading("Filtered player table")
st.dataframe(display, width="stretch", hide_index=True)
st.caption(f"{len(display)} of {len(players)} players shown.")

st.download_button(
    "Download player_table.csv",
    display.to_csv(index=False),
    file_name="player_table.csv",
    mime="text/csv",
)
