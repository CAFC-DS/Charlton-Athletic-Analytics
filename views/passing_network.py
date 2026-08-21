"""Passing-network match view and player-scope controls."""

# =============================================================================
# PASSING NETWORK - real Impect passer-to-receiver map
# =============================================================================
import unicodedata

import pandas as pd
import streamlit as st

from utils import data
from utils import match_analysis as ma
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


def _player_key(value: object) -> str:
    """Return a comparison key that is stable across Opta/Impect name variants."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    return "".join(character for character in text if character.isalnum())


def _same_team_label(candidate: object, target: object) -> bool:
    candidate_key = _player_key(candidate)
    target_key = _player_key(target)
    if not candidate_key or not target_key:
        return False
    return candidate_key == target_key or candidate_key in target_key or target_key in candidate_key


def _network_player_names(network: pd.DataFrame) -> set[str]:
    """Return normalized names represented by either endpoint of a network link."""
    names: set[str] = set()
    for column in ["Player", "Receiver"]:
        if column in network:
            names.update(key for key in network[column].map(_player_key) if key)
    return names


def _fallback_starting_names(team_passes: pd.DataFrame) -> set[str]:
    """Infer an XI when an Opta F7 lineup is unavailable.

    The fallback mirrors the match-overview convention: players involved earliest
    in the event feed are preferred, with action volume breaking ties. It is only
    used for older fixtures without a paired Opta lineup.
    """
    if team_passes.empty or "Player" not in team_passes:
        return set()
    candidates = team_passes.dropna(subset=["Player"]).copy()
    if candidates.empty:
        return set()
    candidates["_Second"] = pd.to_numeric(candidates.get("Second"), errors="coerce")
    grouped = candidates.groupby("Player", as_index=False).agg(
        **{
            "First Second": ("_Second", "min"),
            "Passes": ("Player", "size"),
        }
    )
    grouped["First Second"] = grouped["First Second"].fillna(999999)
    grouped = grouped.sort_values(
        ["First Second", "Passes", "Player"],
        ascending=[True, False, True],
    ).head(11)
    return {_player_key(name) for name in grouped["Player"] if _player_key(name)}


def _top_network_passer_names(network: pd.DataFrame) -> set[str]:
    """Return the eleven highest-volume passers in the visible Impect network."""
    if network.empty or "Player" not in network or "Pass Count" not in network:
        return set()
    volume = network[["Player", "Pass Count"]].copy()
    volume["Pass Count"] = pd.to_numeric(volume["Pass Count"], errors="coerce").fillna(0)
    volume["Player Key"] = volume["Player"].map(_player_key)
    volume = (
        volume[volume["Player Key"].ne("")]
        .groupby(["Player Key", "Player"], as_index=False)["Pass Count"]
        .sum()
        .sort_values(["Pass Count", "Player"], ascending=[False, True])
        .head(11)
    )
    return set(volume["Player Key"])


def _match_player_scopes(
    match_row: pd.Series,
    team_name: str,
    network: pd.DataFrame,
    team_passes: pd.DataFrame,
) -> tuple[dict[str, set[str]], str]:
    """Build starting-XI, played-player and top-passer name sets.

    Opta F7/F24 is the authoritative source for the first two scopes. The
    network's Impect participants are always included in the played scope because
    a player with a pass in the selected team feed necessarily appeared. This
    also keeps the filter useful when a provider lineup is partial.
    """
    network_names = _network_player_names(network)
    starting_names: set[str] = set()
    played_names: set[str] = set(network_names)
    source_note = "Player scopes use the paired Opta lineup and substitution feed."

    try:
        fixture_id = data.opta_fixture_id_for_match(match_row)
        lineups = data.load_opta_lineups(fixture_id)
    except Exception:
        fixture_id = None
        lineups = pd.DataFrame()

    team_lineups = lineups[
        lineups.get("Team", pd.Series(index=lineups.index, dtype=object)).map(
            lambda value: _same_team_label(value, team_name)
        )
    ].copy() if not lineups.empty else lineups

    if not team_lineups.empty and "Player" in team_lineups:
        starting_names = {
            _player_key(name)
            for name in team_lineups.loc[
                team_lineups["Lineup Status"].astype(str).str.casefold().eq("start"),
                "Player",
            ]
            if _player_key(name)
        }
        # Start players have definitely appeared. Do not add every F7
        # substitute here: that table also contains unused bench players.
        played_names.update(starting_names)

        # F7 lists unused substitutes as well, so only add players explicitly
        # recorded as entering by the F24 event feed.
        try:
            opta_events = data.load_opta_events(fixture_id, limit=50000)
        except Exception:
            opta_events = pd.DataFrame()
        if not opta_events.empty and "TypeId" in opta_events and "Player" in opta_events:
            entered = opta_events[
                opta_events["TypeId"].eq(getattr(data, "OPTA_TYPE_PLAYER_ON", 19))
                & opta_events.get("Team", pd.Series(index=opta_events.index, dtype=object)).map(
                    lambda value: _same_team_label(value, team_name)
                )
            ]
            played_names.update(_player_key(name) for name in entered["Player"] if _player_key(name))
    else:
        starting_names = _fallback_starting_names(team_passes)
        try:
            minutes = data.load_match_player_minutes(
                season=str(match_row.get("Season", "")),
                match_id=match_row.get("MatchId"),
                team=team_name,
            )
        except Exception:
            minutes = pd.DataFrame()
        if not minutes.empty and "Player" in minutes:
            played_names.update(_player_key(name) for name in minutes["Player"] if _player_key(name))
        source_note = "Opta lineup data was unavailable; starting XI is inferred from earliest match involvement."

    if not starting_names:
        starting_names = _fallback_starting_names(team_passes)
        source_note = "Starting XI is inferred from earliest match involvement because no Opta starting lineup was available."
    if not played_names:
        played_names = set(starting_names)

    return {
        "Starting XI": starting_names,
        "All players who played": played_names,
        "Top 11 passers": _top_network_passer_names(network),
    }, source_note


def _filter_network(network: pd.DataFrame, selected_names: set[str]) -> pd.DataFrame:
    """Keep only links where both passer and receiver are in the selected scope."""
    if network.empty or not selected_names:
        return network.iloc[0:0].copy()
    passer_keys = network["Player"].map(_player_key) if "Player" in network else pd.Series(False, index=network.index)
    receiver_keys = network["Receiver"].map(_player_key) if "Receiver" in network else pd.Series(False, index=network.index)
    return network[passer_keys.isin(selected_names) & receiver_keys.isin(selected_names)].copy()


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
scope_options = ["Starting XI", "All players who played", "Top 11 passers"]
player_scope = control_cols[1].selectbox("Players shown", scope_options)
player_scopes, scope_note = _match_player_scopes(match_row, team_name, network, team_passes)
selected_names = player_scopes.get(player_scope, set())
visible_network = _filter_network(network, selected_names)

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
    f"{player_scope}: {len(selected_names)} players selected. "
    "Only links where both the passer and receiver are in this group are shown. "
    f"{scope_note}"
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
