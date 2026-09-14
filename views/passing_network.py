"""Passing-network match view and player-scope controls."""

# =============================================================================
# PASSING NETWORK - real Impect passer-to-receiver map
# =============================================================================
import pandas as pd
import streamlit as st

from utils import data
from utils import match_analysis as ma
from utils import pass_network as pn
from utils import pitch
from utils import ui


def _passing_network_css() -> None:
    st.markdown(
        """
        <style>
        .pn-summary-card {
            background: var(--ss-panel);
            border: 1px solid var(--ss-border);
            border-radius: 8px;
            border-top: 3px solid var(--ss-accent);
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-height: 102px;
            padding: 14px 16px;
        }

        .pn-summary-label {
            color: var(--ss-muted);
            font-size: 0.875rem;
            font-weight: 650;
            line-height: 1.25;
            margin-bottom: 14px;
        }

        .pn-summary-value {
            color: var(--ss-ink);
            display: block;
            font-size: clamp(1.5rem, 1.9vw, 1.9rem);
            font-weight: 400;
            letter-spacing: -0.03em;
            line-height: 1.1;
        }

        .pn-summary-value-text {
            font-size: clamp(0.78rem, 0.92vw, 0.98rem);
            letter-spacing: -0.01em;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _summary_card(label: str, value: object, *, text_value: bool = False) -> None:
    value_class = "pn-summary-value pn-summary-value-text" if text_value else "pn-summary-value"
    st.markdown(
        f"""
        <div class="pn-summary-card">
            <div class="pn-summary-label">{ui.esc(label)}</div>
            <div class="{value_class}">{ui.esc(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _match_player_scopes(
    match_row: pd.Series,
    team_name: str,
    network: pd.DataFrame,
    team_passes: pd.DataFrame,
) -> tuple[dict[str, set[str]], str]:
    """Build full-match, starting-XI and top-passer network scopes."""
    network_names = pn.network_player_names(network)
    starting_names: set[str] = set()
    source_note = "The starting XI uses the paired Opta lineup."

    try:
        fixture_id = data.opta_fixture_id_for_match(match_row)
        lineups = data.load_opta_lineups(fixture_id)
    except Exception:
        fixture_id = None
        lineups = pd.DataFrame()

    team_lineups = lineups[
        lineups.get("Team", pd.Series(index=lineups.index, dtype=object)).map(
            lambda value: pn.same_team_label(value, team_name)
        )
    ].copy() if not lineups.empty else lineups

    if not team_lineups.empty and "Player" in team_lineups:
        lineup_starters = team_lineups.loc[
            team_lineups["Lineup Status"].astype(str).str.casefold().eq("start"),
            "Player",
        ]
        starting_names = pn.resolve_network_name_keys(lineup_starters, network)
    else:
        starting_names = pn.fallback_starting_names(team_passes)
        source_note = "Opta lineup data was unavailable; starting XI is inferred from earliest match involvement."

    if not starting_names:
        starting_names = pn.fallback_starting_names(team_passes)
        source_note = "Starting XI is inferred from earliest match involvement because no Opta starting lineup was available."

    return {
        "Entire match": network_names,
        "Starting XI": starting_names,
        "Top 11 passers": pn.top_network_passer_names(network),
    }, source_note


ma.page_header(
    "Passing Network",
    "Map passer-to-receiver links from completed CAFC_DB Impect pass events for a selected fixture and team.",
    "The app derives link counts and average passer/receiver coordinates from the underlying provider event rows.",
)
_passing_network_css()

season = ma.select_match_season(key="passing_network_match_season")
matches = ma.load_matches(season)
if matches.empty:
    st.warning("No match data is available for this season.")
    st.stop()

match_row = ma.match_selector(matches, key="passing_network_match")
team_name = ma.team_selector_for_match(match_row, key="passing_network_team")
network = data.load_pass_network(
    season=season,
    match_id=match_row.get("MatchId"),
    team=team_name,
)
team_passes = data.load_match_events(
    season=season,
    match_id=match_row.get("MatchId"),
    team=team_name,
    action_types=["PASS"],
    limit=2500,
)
crosses = team_passes[data.is_cross(team_passes)].copy() if not team_passes.empty else team_passes
crosses_completed = int(crosses["Result"].astype(str).str.upper().eq("SUCCESS").sum()) if not crosses.empty else 0

ma.section_heading("Network controls")
control_cols = st.columns(2)
max_count = int(pd.to_numeric(network["Pass Count"], errors="coerce").max()) if not network.empty else 2
default_min_passes = min(2, max(max_count, 1))
min_passes = control_cols[0].slider(
    "Minimum link passes",
    min_value=1,
    max_value=max(max_count, 2),
    value=default_min_passes,
)
scope_options = ["Entire match", "Starting XI", "Top 11 passers"]
player_scope = control_cols[1].selectbox("Players shown", scope_options)
player_scopes, scope_note = _match_player_scopes(match_row, team_name, network, team_passes)
selected_names = player_scopes.get(player_scope, set())
visible_network = (
    network.copy()
    if player_scope == "Entire match"
    else pn.filter_network(network, selected_names)
)
represented_players = pn.network_player_count(visible_network)
scope_detail = {
    "Entire match": "All players with a completed pass connection are included.",
    "Top 11 passers": "Players are ranked by completed passes made.",
}.get(player_scope, scope_note)

ma.section_heading("Selected fixture summary")
metric_cols = st.columns(5)
with metric_cols[0]:
    _summary_card("Fixture", str(match_row.get("Match", "Unknown")), text_value=True)
with metric_cols[1]:
    _summary_card("Team", team_name, text_value=True)
with metric_cols[2]:
    _summary_card("Network links", len(visible_network))
with metric_cols[3]:
    _summary_card("Network passes", ma.metric_value(visible_network["Pass Count"].sum() if not visible_network.empty else 0, "Actions"))
with metric_cols[4]:
    _summary_card("Crosses", f"{crosses_completed}/{len(crosses)} completed", text_value=True)

st.caption(
    f"{player_scope}: {represented_players} players represented. "
    "Only links where both the passer and receiver are in this group are shown. "
    f"{scope_detail}"
)
label = f"{team_name} pass network — {player_scope}"

ma.section_heading("Passer-to-receiver network")
if network.empty:
    st.info("No pass-network rows are available for this selected match and team. This table currently covers the Impect event seasons only.")
elif visible_network.empty:
    st.info(f"No pass-network links are available for the selected {player_scope.lower()} filter.")
else:
    st.plotly_chart(pitch.passing_network(visible_network, team_name, label, min_passes=min_passes), width="stretch")

ma.section_heading("Pass network table")
if network.empty or visible_network.empty:
    st.caption("No pass-network links are available for the current selection.")
else:
    display_cols = ["Player", "Receiver", "Pass Count", "Passer X", "Passer Y", "Receiver X", "Receiver Y"]
    st.dataframe(visible_network[display_cols].sort_values("Pass Count", ascending=False), width="stretch", hide_index=True)

ma.section_heading("Crossing links")
st.caption("Crosses are a delivery into the box rather than a repeated passer-receiver relationship, so they sit here as a separate list rather than folded into the network graph above.")
if crosses.empty:
    st.caption("No crosses were attempted by this team in the selected fixture.")
else:
    cross_display_cols = ma.available_columns(
        crosses,
        ["Minute", "Player", "Receiver", "Action", "Result", "Pass Distance", "PXT Pass", "Start X", "Start Y", "End X", "End Y"],
    )
    st.dataframe(crosses[cross_display_cols].sort_values("Minute"), width="stretch", hide_index=True)
