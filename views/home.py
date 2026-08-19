# =============================================================================
# HOME PAGE - Charlton Athletic player discovery landing screen
# =============================================================================
import re

import pandas as pd
import streamlit as st

from utils import data, ui


ui.apply_statsearch_theme()


def _open_profile(player_name: str) -> None:
    st.session_state["selected_player"] = player_name
    st.switch_page("views/player_profiles.py")


def _metric_value(row: pd.Series, metric: str) -> str:
    return ui.format_number(row.get(metric), digits=2)


def _safe_key(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_")


def _load_home_data() -> tuple[dict[str, list[str]], pd.DataFrame, pd.DataFrame]:
    try:
        seasons = data.list_seasons()
        player_seasons = seasons.get("players", [])
        default_season = data.preferred_season(player_seasons)
        season = st.session_state.get("home_season", default_season)
        players = data.load_players(season=season).copy()
        teams = data.load_teams(season=season).copy()
    except Exception as exc:
        st.error("Could not load the football data needed for the homepage.")
        st.exception(exc)
        st.stop()
    return seasons, players, teams


seasons, players, teams = _load_home_data()
if players.empty:
    st.warning("No players are available for the selected data source.")
    st.stop()

players["_Position Display"] = players["Position"].apply(ui.clean_position) if "Position" in players else "Unknown position"
team_options = sorted(players["Team"].dropna().astype(str).unique()) if "Team" in players else []
position_options = sorted(players["_Position Display"].dropna().astype(str).unique())
season_options = seasons.get("players", [])
preferred_home_season = data.preferred_season(season_options)

try:
    summary = data.dataset_summary()
except Exception:
    summary = {
        "season": preferred_home_season if preferred_home_season else "Available",
        "last_refreshed": "Snowflake cache",
        "n_players": len(players),
        "n_teams": len(teams),
        "n_matches": "N/A",
    }

brand_badge = ui.badge_html("ss-logo", "Charlton Athletic crest")
hero_badge = ui.badge_html("ss-hero-badge", "Charlton Athletic crest")

st.markdown(
    f"""
    <div class="ss-brandbar">
        <div class="ss-brand">{brand_badge}<span class="ss-brand-title">Charlton Athletic</span></div>
        <div>Performance analytics platform</div>
    </div>
    <div class="ss-hero">
        <div class="ss-hero-inner">
            <div class="ss-hero-copy">
                <div class="ss-eyebrow">Advanced performance analytics</div>
                <h1 class="ss-title">Unlock tactical intelligence from player and team data</h1>
                <div class="ss-subtitle">
                    Analyse individual player performance, team tactical patterns, and opposition intelligence.
                    Explore positioning data, possession sequences, transitional moments, and phase-specific performance.
                    Powered by advanced event-level analytics and predictive metrics from Impect data.
                </div>
                <div class="ss-pill-row">
                    <span class="ss-pill">Player intelligence</span>
                    <span class="ss-pill">Team tactics</span>
                    <span class="ss-pill">Match analysis</span>
                    <span class="ss-pill">Positional data</span>
                </div>
            </div>
            {hero_badge}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="ss-section-label">Platform overview</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Season", summary.get("season", preferred_home_season if preferred_home_season else "Available"))
c2.metric("Players", summary.get("n_players", len(players)))
c3.metric("Teams", summary.get("n_teams", len(teams)))
c4.metric("Data refreshed", summary.get("last_refreshed", "Snowflake cache"))

st.markdown('<div class="ss-section-label">Scout a player</div>', unsafe_allow_html=True)
filter_col, profile_col = st.columns([1.45, 1])

with filter_col:
    st.markdown('<div class="ss-panel">', unsafe_allow_html=True)
    search = st.text_input("Search player", placeholder="Type a player name...")

    s1, s2, s3 = st.columns(3)
    if season_options:
        selected_season = s1.selectbox(
            "Season", season_options,
            index=season_options.index(preferred_home_season) if preferred_home_season in season_options else len(season_options) - 1,
            key="home_season",
        )
    else:
        selected_season = None
        s1.caption("No season selector available")
    selected_team = s2.selectbox("Team", ["All teams"] + team_options)
    selected_position = s3.selectbox("Position", ["All positions"] + position_options)

    if selected_season and selected_season != preferred_home_season:
        players = data.load_players(season=selected_season).copy()
        players["_Position Display"] = players["Position"].apply(ui.clean_position) if "Position" in players else "Unknown position"

    mask = pd.Series(True, index=players.index)
    if search:
        mask &= players["Player"].astype(str).str.contains(re.escape(search), case=False, na=False)
    if selected_team != "All teams" and "Team" in players:
        mask &= players["Team"].astype(str) == selected_team
    if selected_position != "All positions":
        mask &= players["_Position Display"].astype(str) == selected_position

    filtered = players[mask].copy()
    st.caption(f"{len(filtered)} player(s) match the current filters.")

    display_cols = [col for col in ["Player", "Team", "Position", "Minutes"] + data.PLAYER_METRICS if col in filtered.columns]
    if filtered.empty:
        st.info("No players match those filters. Try clearing the search or widening the filters.")
    else:
        sort_col = "Minutes" if "Minutes" in filtered.columns else "Player"
        ascending = sort_col == "Player"

    st.markdown("</div>", unsafe_allow_html=True)

with profile_col:
    st.markdown('<div class="ss-panel">', unsafe_allow_html=True)
    st.subheader("Open profile")
    if filtered.empty:
        st.caption("A profile link appears here when the filters return players.")
    else:
        lookup = filtered.drop_duplicates("Player").set_index("Player")

        def _label(player_name: str) -> str:
            row = lookup.loc[player_name]
            team = row.get("Team", "Unknown team")
            position = row.get("_Position Display", "Unknown position")
            return f"{player_name} - {position} - {team}"

        selected_player = st.selectbox(
            "Player profile",
            lookup.index.tolist(),
            format_func=_label,
            label_visibility="collapsed",
        )
        if st.button("View player profile", key="home_open_profile"):
            _open_profile(selected_player)

        st.caption("The selected player is carried into the Player Profiles page.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="ss-section-label">Top performers</div>', unsafe_allow_html=True)
primary_metric = "Bypassed Opponents /90" if "Bypassed Opponents /90" in players.columns else data.PLAYER_METRICS[0]
cards = filtered.copy() if not filtered.empty else players.copy()
sort_col = "Minutes" if "Minutes" in cards.columns else "Player"
cards = cards.sort_values(sort_col, ascending=(sort_col == "Player")).head(6)

for start in range(0, len(cards), 3):
    cols = st.columns(3)
    for col, (_, row) in zip(cols, cards.iloc[start:start + 3].iterrows()):
        player_name = str(row["Player"])
        team_name = row.get("Team", "Unknown team")
        position = row.get("_Position Display", ui.clean_position(row.get("Position")))
        with col:
            st.markdown(
                f"""
                <div class="ss-player-card">
                    <div class="ss-player-name">{ui.esc(player_name)}</div>
                    <div class="ss-muted">{ui.esc(position)}<br>{ui.esc(team_name)}</div>
                    <div class="ss-card-stat">{_metric_value(row, primary_metric)}
                        <span class="ss-muted">{ui.esc(primary_metric)}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Open profile", key=f"home_card_{_safe_key(player_name)}"):
                _open_profile(player_name)

st.markdown('<div class="ss-section-label">Core analytics</div>', unsafe_allow_html=True)
a1, a2, a3 = st.columns(3)
with a1:
    st.markdown(
        """
        <div class="ss-player-card">
            <div class="ss-player-name">Player Analysis</div>
            <div class="ss-muted">Detailed player profiles with comparative metrics, positional data, and radar diagnostics for tactical fit assessment.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("views/player_profiles.py", label="Go to Player Analysis")
with a2:
    st.markdown(
        """
        <div class="ss-player-card">
            <div class="ss-player-name">Team Tactics</div>
            <div class="ss-muted">Team-level tactical intelligence including style profile, league comparisons, and squad positioning analysis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("views/team_overview.py", label="Go to Team Tactics")
with a3:
    st.markdown(
        """
        <div class="ss-player-card">
            <div class="ss-player-name">Match Intelligence</div>
            <div class="ss-muted">Per-match tactical review with possession sequences, transition analysis, and performance phase breakdown.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("views/match_overview.py", label="Go to Match Intelligence")

if data.USE_MOCK_DATA:
    st.info("This app is currently running on mock data for development.")
