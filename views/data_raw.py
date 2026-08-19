# =============================================================================
# DATA HUB · LOADER OUTPUT — inspect and export app-ready datasets
# =============================================================================
import streamlit as st
from utils import data, ui

st.title("🗃️ Loader Output")
st.write("Check the data feeding the app, and download it as CSV.")
st.caption(
    "App-ready output from the CAFC_DB loaders; fields may be joined, pivoted, normalised or derived — "
    "for polished, presentation-ready tables see **Cleaned Tables** instead."
)

ui.data_refresh_control()
st.divider()

seasons = data.list_seasons()

tab_players, tab_teams, tab_matches = st.tabs(["Players", "Teams", "Matches"])

with tab_players:
    season = st.selectbox("Season", seasons["players"], index=len(seasons["players"]) - 1, key="players_season")
    players = data.load_players(season=season)
    st.dataframe(players, use_container_width=True, hide_index=True)
    st.download_button(
        "Download players.csv",
        players.to_csv(index=False),
        file_name="players.csv",
        mime="text/csv",
    )

with tab_teams:
    season = st.selectbox("Season", seasons["teams"], index=len(seasons["teams"]) - 1, key="teams_season")
    teams = data.load_teams(season=season)
    st.dataframe(teams, use_container_width=True, hide_index=True)
    st.download_button(
        "Download teams.csv",
        teams.to_csv(index=False),
        file_name="teams.csv",
        mime="text/csv",
    )

with tab_matches:
    season = st.selectbox("Season", seasons["matches"], index=len(seasons["matches"]) - 1, key="matches_season")
    matches = data.load_matches(season=season)
    st.dataframe(matches, use_container_width=True, hide_index=True)
    st.download_button(
        "Download matches.csv",
        matches.to_csv(index=False),
        file_name="matches.csv",
        mime="text/csv",
    )
