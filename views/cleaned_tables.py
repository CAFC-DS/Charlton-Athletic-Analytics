# =============================================================================
# DATA HUB · CLEANED TABLES — presentation-ready versions of the loaders
# =============================================================================
# Raw Data shows exactly what the loaders return. This page applies display
# polish on top: readable position labels, rounded numbers, sensible sort
# order and no internal-only columns (e.g. MatchId) — the version you'd
# actually want to read or screenshot, not just audit.
# =============================================================================
import streamlit as st
from utils import data, ui

st.title("🧹 Cleaned Tables")
st.caption("Presentation-ready versions of Players, Teams and Matches — sorted, rounded and relabelled for reading rather than auditing.")

ui.data_refresh_control()
st.divider()


def _clean_position(position: str) -> str:
    """'CENTRAL_MIDFIELD, RIGHT_WINGER' -> 'Central Midfield, Right Winger'."""
    parts = [p.strip().replace("_", " ").title() for p in position.split(",")]
    return ", ".join(parts)


seasons = data.list_seasons()

tab_players, tab_teams, tab_matches = st.tabs(["Players", "Teams", "Matches"])

with tab_players:
    preferred_players_season = data.preferred_season(seasons["players"])
    season = st.selectbox(
        "Season", seasons["players"],
        index=seasons["players"].index(preferred_players_season),
        key="clean_players_season",
    )
    players = data.load_players(season=season).copy()

    players["Position"] = players["Position"].apply(_clean_position)
    players["Minutes"] = players["Minutes"].round(0).astype(int)
    for col in ["Goals /90", "Assists /90", "Bypassed Opponents /90", "Passes to Final 3rd /90"]:
        players[col] = players[col].round(2)

    players = players.sort_values("Minutes", ascending=False).reset_index(drop=True)
    st.dataframe(players, width="stretch", hide_index=True)
    st.caption(f"{len(players)} players, sorted by minutes played (most-featured first).")

with tab_teams:
    preferred_teams_season = data.preferred_season(seasons["teams"])
    season = st.selectbox(
        "Season", seasons["teams"],
        index=seasons["teams"].index(preferred_teams_season),
        key="clean_teams_season",
    )
    teams = data.load_teams(season=season).copy()

    for col in ["Goals /90", "Assists /90", "Bypassed Opponents /90", "Passes to Final 3rd /90"]:
        teams[col] = teams[col].round(2)

    teams = teams.sort_values("Goals /90", ascending=False).reset_index(drop=True)
    st.dataframe(teams, width="stretch", hide_index=True)
    st.caption(f"{len(teams)} teams, sorted by Goals /90 (highest first).")

with tab_matches:
    preferred_matches_season = data.preferred_season(seasons["matches"])
    season = st.selectbox(
        "Season", seasons["matches"],
        index=seasons["matches"].index(preferred_matches_season),
        key="clean_matches_season",
    )
    matches = data.load_matches(season=season).copy()

    matches["Home Goals"] = matches["Home Goals"].astype(int)
    matches["Away Goals"] = matches["Away Goals"].astype(int)
    matches = matches.drop(columns=["MatchId"], errors="ignore")
    matches = matches[["Match", "Date", "Competition", "Season", "Home", "Home Goals",
                        "Away", "Away Goals", "Result", "Venue Verified"]]
    matches = matches.sort_values("Date", ascending=False).reset_index(drop=True)

    st.dataframe(matches, width="stretch", hide_index=True)
    st.caption(f"{len(matches)} matches, most recent first.")
    if not matches["Venue Verified"].all():
        st.caption(
            "⚠️ 'Venue Verified' is False for seasons with no home/away flag in the source data — "
            "Home/Away there is a consistent but arbitrary ordering, not a confirmed hosting fact."
        )
