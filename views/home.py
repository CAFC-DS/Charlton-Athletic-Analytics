# =============================================================================
# HOME PAGE - Charlton Athletic player discovery landing screen
# =============================================================================
import re

import pandas as pd
import streamlit as st

from utils import data, ui


LEADERBOARD_METRICS = [
    "Bypassed Opponents /90",
    "Packing xG /90",
    "xG /90",
    "Key Passes /90",
    "Ball Wins /90",
    "Successful Passes /90",
]
LEADERBOARD_SIZE = 10


def _open_profile(player_name: str) -> None:
    st.session_state["selected_player"] = player_name
    st.switch_page("views/player_profiles.py")


def _named_players(frame: pd.DataFrame) -> pd.DataFrame:
    """Return profile-safe rows with a real, normalized player name."""
    if "Player" not in frame:
        return frame.iloc[0:0].copy()

    names = frame["Player"].astype("string").str.strip()
    invalid_names = {"", "nan", "none", "null", "<na>", "nat"}
    valid = names.notna() & ~names.str.casefold().isin(invalid_names)
    result = frame.loc[valid].copy()
    result["Player"] = names.loc[valid]
    return result


def _refreshed_text(value: object) -> str:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    return timestamp.strftime("%d %b %Y") if pd.notna(timestamp) else "Cached"


def _top_performer_pool(frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Players with enough minutes for a per-90 rate to mean something.

    Ranking a per-90 metric over every player surfaces cameo outliers, so the
    pool keeps players with at least 40% of the season's highest minutes
    (capped at 900, i.e. ten full matches).
    """
    if "Minutes" not in frame:
        return frame, 0.0
    minutes = pd.to_numeric(frame["Minutes"], errors="coerce")
    if minutes.notna().sum() == 0:
        return frame, 0.0
    threshold = min(900.0, float(minutes.max()) * 0.4)
    return frame[minutes.ge(threshold)], threshold


def _link_card(title: str, body: str, page: str, label: str) -> None:
    with ui.card(f"home_link_{title.lower().replace(' ', '_')}"):
        st.markdown(
            f'<div class="cafc-link-title">{ui.esc(title)}</div><div class="cafc-link-body">{ui.esc(body)}</div>',
            unsafe_allow_html=True,
        )
        st.page_link(page, label=label, icon=":material/arrow_forward:")


try:
    seasons = data.list_seasons()
except Exception as exc:
    st.error("Could not load the football data needed for the homepage.")
    st.exception(exc)
    st.stop()

season_options = seasons.get("players", [])
preferred_home_season = data.preferred_season(season_options)
if st.session_state.get("home_season") not in season_options:
    st.session_state.pop("home_season", None)
season = st.session_state.get("home_season", preferred_home_season)

try:
    players = data.load_players(season=season).copy()
    teams = data.load_teams(season=season)
except Exception as exc:
    st.error("Could not load the football data needed for the homepage.")
    st.exception(exc)
    st.stop()

if players.empty:
    st.warning("No players are available for the selected data source.")
    st.stop()

players["_Position Display"] = players["Position"].apply(ui.clean_position) if "Position" in players else "Unknown position"
# Provider data can contain aggregate/placeholder rows without a player name.
# They cannot be profiled and would otherwise all receive the same `nan` widget key.
players = _named_players(players)
if players.empty:
    st.warning("No named players are available for the selected data source.")
    st.stop()
team_options = sorted(players["Team"].dropna().astype(str).unique()) if "Team" in players else []
position_options = sorted(players["_Position Display"].dropna().astype(str).unique())

try:
    last_refreshed = _refreshed_text(data.last_refreshed())
except Exception:
    last_refreshed = "Cached"
team_count = teams["Team"].nunique() if "Team" in teams else players["Team"].nunique()

ui.page_header(
    "CAFC Analytics",
    "Player, team and match analysis for Charlton Athletic and the league around them, "
    "built on Impect event data and Opta feeds.",
    eyebrow="Charlton Athletic · Performance analytics",
    facts=[
        ("Season", season or "—"),
        ("Players", f"{len(players):,}"),
        ("Teams", team_count),
        ("Updated", last_refreshed),
    ],
)

with ui.filter_bar():
    f_season, f_team, f_position, f_search = st.columns([1, 1.4, 1.4, 1.6])
    if season_options:
        f_season.selectbox(
            "Season",
            season_options,
            index=season_options.index(season) if season in season_options else len(season_options) - 1,
            key="home_season",
        )
    selected_team = f_team.selectbox("Team", ["All teams"] + team_options, key="home_team")
    selected_position = f_position.selectbox("Position", ["All positions"] + position_options, key="home_position")
    search = f_search.text_input("Player", placeholder="Search player", key="home_search")

mask = pd.Series(True, index=players.index)
if search:
    mask &= players["Player"].astype(str).str.contains(re.escape(search), case=False, na=False)
if selected_team != "All teams" and "Team" in players:
    mask &= players["Team"].astype(str) == selected_team
if selected_position != "All positions":
    mask &= players["_Position Display"].astype(str) == selected_position
filtered = players[mask].copy()

scope = selected_team if selected_team != "All teams" else f"{team_count} teams"
st.markdown(
    f'<p class="cafc-summary"><strong>{len(filtered):,}</strong> of {len(players):,} players match, '
    f"from {ui.esc(scope)} in <strong>{ui.esc(season)}</strong>"
    + (f", {ui.esc(selected_position.lower())}s only" if selected_position != "All positions" else "")
    + ".</p>",
    unsafe_allow_html=True,
)

board_col, profile_col = st.columns([1.55, 1], gap="medium")

with board_col, ui.card("home_board"):
    available_metrics = [metric for metric in LEADERBOARD_METRICS if metric in players.columns] or data.PLAYER_METRICS
    st.markdown('<div class="cafc-link-title">Top performers</div>', unsafe_allow_html=True)
    metric = st.segmented_control(
        "Metric",
        available_metrics,
        default=available_metrics[0],
        key="home_metric",
        label_visibility="collapsed",
    ) or available_metrics[0]

    card_pool, minutes_threshold = _top_performer_pool(filtered if not filtered.empty else players)
    ranked = (
        card_pool.assign(_rank_value=pd.to_numeric(card_pool[metric], errors="coerce"))
        .dropna(subset=["_rank_value"])
        .sort_values("_rank_value", ascending=False)
        .head(LEADERBOARD_SIZE)
    )
    if ranked.empty:
        st.caption("No players in the current selection have enough minutes to rank.")
    else:
        ui.leaderboard(
            [
                {
                    "name": row["Player"],
                    "sub": f'{row.get("_Position Display", "")} · {row.get("Team", "")}',
                    "value": row["_rank_value"],
                    "text": ui.format_number(row["_rank_value"], digits=2),
                }
                for _, row in ranked.iterrows()
            ],
            highlight="Charlton" if selected_team == "All teams" else None,
        )
        charlton_listed = ranked["Team"].astype(str).str.contains("Charlton", case=False).any()
        st.caption(
            f"{metric} among players with at least {minutes_threshold:,.0f} minutes"
            + (". Charlton players in red." if selected_team == "All teams" and charlton_listed else ".")
        )

with profile_col, ui.card("home_profile"):
    st.markdown('<div class="cafc-link-title">Open a profile</div>', unsafe_allow_html=True)
    if filtered.empty:
        st.caption("No players match these filters. Clear the search or widen the filters.")
    else:
        lookup = filtered.drop_duplicates("Player").set_index("Player")
        top_names = [name for name in ranked["Player"].tolist() if name in lookup.index] if not ranked.empty else []
        options = top_names + [name for name in lookup.index.tolist() if name not in set(top_names)]

        def _label(player_name: str) -> str:
            row = lookup.loc[player_name]
            return f'{player_name} · {row.get("_Position Display", "Unknown position")} · {row.get("Team", "Unknown team")}'

        selected_player = st.selectbox("Player", options, format_func=_label, key="home_profile_player")
        selected_row = lookup.loc[selected_player]
        minutes = pd.to_numeric(pd.Series([selected_row.get("Minutes")]), errors="coerce").iloc[0]
        ui.key_figures(
            [
                {"label": "Minutes", "value": f"{minutes:,.0f}′" if pd.notna(minutes) else "—"},
                {"label": metric, "value": ui.format_number(selected_row.get(metric), digits=2)},
            ],
            hero=False,
        )
        if st.button("View player profile", type="primary", key="home_open_profile", width="stretch"):
            _open_profile(selected_player)
        st.caption("Opens Player Profiles with this player selected.")

ui.section_heading("Explore")
explore = st.columns(4, gap="small")
with explore[0]:
    _link_card(
        "Team analysis",
        "Style profile, league rankings, defending, set pieces and the final third.",
        "views/team_overview.py",
        "Team Overview",
    )
with explore[1]:
    _link_card(
        "Player analysis",
        "Profiles, radars, percentile rankings, comparisons and similar-player search.",
        "views/player_profiles.py",
        "Player Profiles",
    )
with explore[2]:
    _link_card(
        "Match analysis",
        "Fixture reviews: xG and threat timelines, pass maps, transitions and ratings.",
        "views/match_overview.py",
        "Match Overview",
    )
with explore[3]:
    _link_card(
        "Data hub",
        "Loader output, data-quality checks, the metric glossary and exports.",
        "views/metric_glossary.py",
        "Metric Glossary",
    )

if data.USE_MOCK_DATA:
    st.info("This app is currently running on mock data for development.")
