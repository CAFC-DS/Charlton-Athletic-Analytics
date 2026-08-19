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
    preferred_players_season = data.preferred_season(seasons["players"])
    season = st.selectbox(
        "Season", seasons["players"],
        index=seasons["players"].index(preferred_players_season),
        key="players_season",
    )
    players = data.load_players(season=season)
    st.dataframe(players, width="stretch", hide_index=True)
    st.download_button(
        "Download players.csv",
        players.to_csv(index=False),
        file_name="players.csv",
        mime="text/csv",
    )

with tab_teams:
    preferred_teams_season = data.preferred_season(seasons["teams"])
    season = st.selectbox(
        "Season", seasons["teams"],
        index=seasons["teams"].index(preferred_teams_season),
        key="teams_season",
    )
    teams = data.load_teams(season=season)
    st.dataframe(teams, width="stretch", hide_index=True)
    st.download_button(
        "Download teams.csv",
        teams.to_csv(index=False),
        file_name="teams.csv",
        mime="text/csv",
    )

with tab_matches:
    preferred_matches_season = data.preferred_season(seasons["matches"])
    season = st.selectbox(
        "Season", seasons["matches"],
        index=seasons["matches"].index(preferred_matches_season),
        key="matches_season",
    )
    matches = data.load_matches(season=season)
    st.dataframe(matches, width="stretch", hide_index=True)
    st.download_button(
        "Download matches.csv",
        matches.to_csv(index=False),
        file_name="matches.csv",
        mime="text/csv",
    )
