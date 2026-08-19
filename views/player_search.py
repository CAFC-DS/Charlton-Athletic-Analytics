# =============================================================================
# PLAYER SEARCH - searchable player list with profile handoff
# =============================================================================
import re

import pandas as pd
import streamlit as st

from utils import player_analysis as pa
from utils import ui


def _open_profile(player_name: str) -> None:
    st.session_state["selected_player"] = player_name
    st.switch_page("views/player_profiles.py")


pa.page_header(
    "Player Search",
    "Filter the player table by name, team and position, then open a full profile from the results.",
    visualisation_note=False,
)

season = pa.select_season(key="player_search_season")
players = pa.load_player_data(season)
if players.empty:
    st.warning("No players are available for the selected season.")
    st.stop()

team_options = sorted(players["Team"].dropna().astype(str).unique()) if "Team" in players else []
position_options = sorted(players["_Position Display"].dropna().astype(str).unique())

pa.section_heading("Search filters")
st.caption("Use these dropdowns to narrow the player list. The result table and quick-open cards update immediately.")
f1, f2, f3 = st.columns([1.4, 1, 1])
query = f1.text_input("Player name", placeholder="Search by name...")
team = f2.selectbox("Team", ["All teams"] + team_options)
position = f3.selectbox("Position", ["All positions"] + position_options)

mask = pd.Series(True, index=players.index)
if query:
    mask &= players["Player"].astype(str).str.contains(re.escape(query), case=False, na=False)
if team != "All teams" and "Team" in players:
    mask &= players["Team"].astype(str) == team
if position != "All positions":
    mask &= players["_Position Display"].astype(str) == position

filtered = players[mask].copy()

pa.section_heading("Results")
if filtered.empty:
    st.info("No players match those filters.")
    st.stop()

open_col, count_col = st.columns([2, 1])
with open_col:
    lookup = filtered.drop_duplicates("Player").set_index("Player")

    def _label(player_name: str) -> str:
        row = lookup.loc[player_name]
        return f"{player_name} - {row.get('_Position Display', 'Unknown position')} - {row.get('Team', 'Unknown team')}"

    selected_player = st.selectbox("Open profile", lookup.index.tolist(), format_func=_label)
    if st.button("View player profile", key="search_open_profile"):
        _open_profile(selected_player)

with count_col:
    st.metric("Matching players", len(filtered))


sort_col = "Minutes" if "Minutes" in filtered.columns else "Player"


pa.section_heading("Quick open")
quick = filtered.sort_values(sort_col, ascending=(sort_col == "Player")).head(9)
for start in range(0, len(quick), 3):
    cols = st.columns(3)
    for col, (_, row) in zip(cols, quick.iloc[start:start + 3].iterrows()):
        player_name = str(row["Player"])
        with col:
            st.markdown(
                f"""
                <div class="pa-card">
                    <div class="pa-card-icon">Profile</div>
                    <div class="pa-card-title">{ui.esc(player_name)}</div>
                    <div class="pa-card-body">
                        {ui.esc(row.get("_Position Display", "Unknown position"))}<br>
                        {ui.esc(row.get("Team", "Unknown team"))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Open profile", key=f"search_card_{pa.safe_key(player_name)}"):
                _open_profile(player_name)
