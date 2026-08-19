# =============================================================================
# PHASE AVERAGE POSITIONS - event-derived player average locations by phase
# =============================================================================
# Supports two data sources:
#   - Impect (default): uses the Phase column (IN_POSSESSION, ATTACKING_TRANSITION, etc.)
#   - Opta: uses Period (1st half / 2nd half) as the grouping dimension, with
#     optional F7 formation overlay markers.
# =============================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils import data
from utils import match_analysis as ma
from utils import pitch
from utils import ui


PHASE_ORDER = ["IN_POSSESSION", "ATTACKING_TRANSITION", "SECOND_BALL", "SET_PIECE"]
MAX_PLAYERS = 11

# Opta F24 coordinates are 0-100 (pitch percentage).  Impect uses centred
# -52.5..52.5 (x) and -34..34 (y).  These constants convert Opta -> Impect.
OPTA_X_MIN = 0.0
OPTA_X_MAX = 100.0
OPTA_Y_MIN = 0.0
OPTA_Y_MAX = 100.0
IMPECT_X_MIN = -52.5
IMPECT_X_MAX = 52.5
IMPECT_Y_MIN = -34.0
IMPECT_Y_MAX = 34.0


def _phase_average_css() -> None:
    st.markdown(
        """
        <style>
        .pap-summary-card {
            background: var(--ss-panel);
            border: 1px solid var(--ss-border);
            border-radius: 8px;
            border-top: 3px solid var(--ss-accent);
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-height: 102px;
            padding: 14px 16px;
        }

        .pap-summary-label {
            color: var(--ss-muted);
            font-size: 0.875rem;
            font-weight: 650;
            line-height: 1.25;
            margin-bottom: 14px;
        }

        .pap-summary-value {
            color: var(--ss-ink);
            display: block;
            font-size: clamp(1.35rem, 1.75vw, 1.75rem);
            font-weight: 400;
            letter-spacing: -0.03em;
            line-height: 1.1;
        }

        .pap-summary-value-text {
            font-size: clamp(0.80rem, 0.90vw, 0.96rem);
            letter-spacing: -0.01em;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _summary_card(label: str, value: object, *, text_value: bool = False) -> None:
    value_class = "pap-summary-value pap-summary-value-text" if text_value else "pap-summary-value"
    st.markdown(
        f"""
        <div class="pap-summary-card">
            <div class="pap-summary-label">{ui.esc(label)}</div>
            <div class="{value_class}">{ui.esc(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _phase_label(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return "No Phase Label"
    return text.replace("_", " ").title()


def _mode_text(values: pd.Series) -> str:
    clean = values.dropna().astype(str).str.strip()
    clean = clean[~clean.str.lower().isin(["", "nan", "none", "null"])]
    if clean.empty:
        return ""
    mode = clean.mode()
    return str(mode.iloc[0] if not mode.empty else clean.iloc[0])


def _available_phases(events: pd.DataFrame) -> list[str]:
    if events.empty or "Phase" not in events:
        return []
    phases = events["Phase"].dropna().astype(str).str.strip()
    phases = phases[~phases.str.lower().isin(["", "nan", "none", "null"])]
    observed = phases.drop_duplicates().tolist()
    ordered = [phase for phase in PHASE_ORDER if phase in observed]
    ordered.extend(sorted(phase for phase in observed if phase not in ordered))
    return ordered


def _filter_phase(events: pd.DataFrame, phase_value: str | None) -> pd.DataFrame:
    if not phase_value or phase_value == "ALL":
        return events.copy()
    if events.empty or "Phase" not in events:
        return events.copy()
    return events[events["Phase"].astype(str) == str(phase_value)].copy()


def _top_players_by_minutes(player_minutes: pd.DataFrame) -> pd.DataFrame:
    columns = ["PlayerId", "Player", "Position", "Minutes", "Match Share"]
    if player_minutes.empty or "Player" not in player_minutes:
        return pd.DataFrame(columns=columns)

    values = player_minutes.copy()
    values["Minutes"] = pd.to_numeric(values.get("Minutes"), errors="coerce").fillna(0)
    values["Match Share"] = pd.to_numeric(values.get("Match Share"), errors="coerce").fillna(0)
    values = values[values["Minutes"] > 0].copy()
    if values.empty:
        return pd.DataFrame(columns=columns)
    for col in columns:
        if col not in values:
            values[col] = np.nan
    return (
        values.sort_values(["Minutes", "Match Share", "Player"], ascending=[False, False, True])
        .head(MAX_PLAYERS)
        .reset_index(drop=True)[columns]
    )


def _event_involvement_pool(events: pd.DataFrame) -> pd.DataFrame:
    """Build a player pool from event involvement, with a wider cap to avoid missing
    low-touch players (e.g. CBs, GKs) when match-minute KPI data is unavailable."""
    columns = ["PlayerId", "Player", "Position", "Minutes", "Match Share"]
    if events.empty or "Player" not in events:
        return pd.DataFrame(columns=columns)

    values = events.dropna(subset=["Player"]).copy()
    if values.empty:
        return pd.DataFrame(columns=columns)
    values["_Position"] = values["Position"] if "Position" in values else ""
    grouped = values.groupby(["PlayerId", "Player"], dropna=False, as_index=False).agg(
        Position=("_Position", _mode_text),
        Actions=("Player", "size"),
    )
    grouped["Minutes"] = np.nan
    grouped["Match Share"] = np.nan
    # Use a wider cap (14) so defensive/low-touch players aren't excluded
    return grouped.sort_values(["Actions", "Player"], ascending=[False, True]).head(14)[columns].reset_index(drop=True)


def _string_keys(values: pd.Series) -> pd.Series:
    keys = values.astype(str)
    return keys.where(~keys.str.lower().isin(["", "nan", "none", "null"]), "")


def _attach_match_minutes(events: pd.DataFrame, player_minutes: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["Match Minutes"] = np.nan
    if out.empty or player_minutes.empty:
        return out

    minutes = player_minutes.copy()
    minutes["Minutes"] = pd.to_numeric(minutes.get("Minutes"), errors="coerce")

    if "PlayerId" in out and "PlayerId" in minutes:
        out["_PlayerIdKey"] = _string_keys(out["PlayerId"])
        lookup = minutes[["PlayerId", "Minutes"]].dropna(subset=["PlayerId"]).drop_duplicates("PlayerId")
        lookup["_PlayerIdKey"] = _string_keys(lookup["PlayerId"])
        lookup = lookup[lookup["_PlayerIdKey"] != ""]
        if not lookup.empty:
            out = out.merge(
                lookup[["_PlayerIdKey", "Minutes"]].rename(columns={"Minutes": "_Match Minutes"}),
                on="_PlayerIdKey",
                how="left",
            )
            out["Match Minutes"] = out["_Match Minutes"].combine_first(out["Match Minutes"])
            out = out.drop(columns=["_Match Minutes"], errors="ignore")

    if "Player" in out and "Player" in minutes:
        out["_PlayerNameKey"] = _string_keys(out["Player"])
        lookup = minutes[["Player", "Minutes"]].dropna(subset=["Player"]).drop_duplicates("Player")
        lookup["_PlayerNameKey"] = _string_keys(lookup["Player"])
        lookup = lookup[lookup["_PlayerNameKey"] != ""]
        if not lookup.empty:
            out = out.merge(
                lookup[["_PlayerNameKey", "Minutes"]].rename(columns={"Minutes": "_Match Minutes"}),
                on="_PlayerNameKey",
                how="left",
            )
            out["Match Minutes"] = out["Match Minutes"].combine_first(out["_Match Minutes"])
            out = out.drop(columns=["_Match Minutes"], errors="ignore")

    return out.drop(columns=["_PlayerIdKey", "_PlayerNameKey"], errors="ignore")


def _filter_player_pool(events: pd.DataFrame, player_pool: pd.DataFrame) -> pd.DataFrame:
    if events.empty or player_pool.empty:
        return events.copy()

    mask = pd.Series(False, index=events.index)
    if "PlayerId" in events and "PlayerId" in player_pool:
        player_ids = set(_string_keys(player_pool["PlayerId"]))
        player_ids.discard("")
        if player_ids:
            mask |= _string_keys(events["PlayerId"]).isin(player_ids)

    if "Player" in events and "Player" in player_pool:
        player_names = set(player_pool["Player"].dropna().astype(str))
        if player_names:
            mask |= events["Player"].astype(str).isin(player_names)

    return events[mask].copy()


def _average_position_table(events: pd.DataFrame) -> pd.DataFrame:
    columns = ["Player", "Position", "Minutes", "Actions", "Primary Action", "Average X", "Average Y"]
    if events.empty or "Player" not in events:
        return pd.DataFrame(columns=columns)

    values = events.dropna(subset=["Player", "Start X", "Start Y"]).copy()
    if values.empty:
        return pd.DataFrame(columns=columns)

    values["Start X"] = pd.to_numeric(values["Start X"], errors="coerce")
    values["Start Y"] = pd.to_numeric(values["Start Y"], errors="coerce")
    values["_Position"] = values["Position"] if "Position" in values else ""
    values["_Action"] = (
        values["Action"].fillna(values["Action Type"]) if "Action Type" in values
        else values.get("Action", "")
    )
    values["_Match Minutes"] = pd.to_numeric(values["Match Minutes"], errors="coerce") if "Match Minutes" in values else np.nan
    summary = values.groupby("Player", as_index=False).agg(
        Position=("_Position", _mode_text),
        Minutes=("_Match Minutes", "max"),
        Actions=("Player", "size"),
        **{
            "Primary Action": ("_Action", _mode_text),
            "Average X": ("Start X", "mean"),
            "Average Y": ("Start Y", "mean"),
        },
    )
    if summary.empty:
        return pd.DataFrame(columns=columns)
    summary["Minutes"] = pd.to_numeric(summary["Minutes"], errors="coerce").round(1)
    summary["Average X"] = summary["Average X"].round(1)
    summary["Average Y"] = summary["Average Y"].round(1)
    return summary.sort_values(["Minutes", "Actions"], ascending=[False, False]).reset_index(drop=True)[columns]


# ---- Opta helpers ----------------------------------------------------------------

def _opta_to_impect_x(opta_x: pd.Series) -> pd.Series:
    """Convert Opta 0-100 x coordinate to Impect -52.5..52.5."""
    fraction = (pd.to_numeric(opta_x, errors="coerce") - OPTA_X_MIN) / (OPTA_X_MAX - OPTA_X_MIN)
    return fraction * (IMPECT_X_MAX - IMPECT_X_MIN) + IMPECT_X_MIN


def _opta_to_impect_y(opta_y: pd.Series) -> pd.Series:
    """Convert Opta 0-100 y coordinate to Impect -34..34.

    Opta y=0 is the top of the pitch (from the broadcast perspective),
    while Impect y=-34 is the bottom.  We flip the axis by subtracting
    from 100 before mapping.
    """
    fraction = (pd.to_numeric(opta_y, errors="coerce") - OPTA_Y_MIN) / (OPTA_Y_MAX - OPTA_Y_MIN)
    # Flip the Y axis: Opta y=100 -> Impect y=-34 (bottom), Opta y=0 -> Impect y=34 (top)
    return IMPECT_Y_MAX - fraction * (IMPECT_Y_MAX - IMPECT_Y_MIN)


def _normalise_opta_events(events: pd.DataFrame) -> pd.DataFrame:
    """Convert Opta F24 event coordinates to the Impect pitch coordinate system."""
    out = events.copy()
    if "Start X" in out:
        out["Start X"] = _opta_to_impect_x(out["Start X"])
    if "Start Y" in out:
        out["Start Y"] = _opta_to_impect_y(out["Start Y"])
    if "End X" in out:
        out["End X"] = _opta_to_impect_x(out["End X"])
    if "End Y" in out:
        out["End Y"] = _opta_to_impect_y(out["End Y"])
    return out


def _opta_id(value: object) -> str:
    """Strip Opta's XML-style entity prefix (e.g. "t33" -> "33")."""
    text = "" if value is None else str(value).strip()
    return text[1:] if len(text) > 1 and text[0].isalpha() else text


def _opta_period_label(period: object) -> str:
    try:
        p = int(float(period))
    except (TypeError, ValueError):
        return "Unknown"
    return {1: "First Half", 2: "Second Half", 3: "First Half ET", 4: "Second Half ET", 5: "Penalties"}.get(p, f"Period {p}")


def _available_periods(events: pd.DataFrame) -> list[int]:
    if events.empty or "Period" not in events:
        return []
    periods = pd.to_numeric(events["Period"], errors="coerce").dropna().astype(int).unique().tolist()
    return sorted(p for p in periods if p in {1, 2, 3, 4})


def _filter_period(events: pd.DataFrame, period_value: int | None) -> pd.DataFrame:
    if period_value is None:
        return events.copy()
    if events.empty or "Period" not in events:
        return events.copy()
    return events[pd.to_numeric(events["Period"], errors="coerce").eq(period_value)].copy()


def _opta_minute_ranked_pool(lineups: pd.DataFrame, team_name: str) -> pd.DataFrame:
    """Build a player pool from Opta F7 lineup data for a given team."""
    columns = ["PlayerId", "Player", "Position", "Minutes", "Match Share"]
    if lineups.empty:
        return pd.DataFrame(columns=columns)
    pool = lineups[lineups["Team"].astype(str).eq(str(team_name))].copy()
    if pool.empty:
        return pd.DataFrame(columns=columns)
    pool["Minutes"] = 90.0
    pool["Match Share"] = 1.0
    pool["Position"] = pool.get("Position Group", pool.get("Registered Position", ""))
    out = pool[["PlayerId", "Player", "Position", "Minutes", "Match Share"]].drop_duplicates("PlayerId").copy()
    return out.head(MAX_PLAYERS).reset_index(drop=True)


# ---- Main page -------------------------------------------------------------------

ma.page_header(
    "Phase Average Positions",
    "Compare event-derived player average locations across match phases. Impect provides native phase labels; "
    "Opta F24 events are grouped by half (Period) with optional F7 formation overlay.",
    "CAFC_DB Impect & Opta provider events. This uses event locations, not continuous tracking data.",
)
_phase_average_css()

# ----- Data source selector -------------------------------------------------------
data_source = st.radio(
    "Data source",
    ["Impect (phases)", "Opta (halves)"],
    horizontal=True,
    index=0,
    key="pap_data_source",
)
using_impect = data_source == "Impect (phases)"

if using_impect:
    # ---- IMPECT DATA FLOW (existing) --------------------------------------------
    season = ma.select_match_season(key="phase_avg_positions_season")
    matches = ma.load_matches(season)
    if matches.empty:
        st.warning("No match data is available for this season.")
        st.stop()

    match_row = ma.match_selector(matches, key="phase_avg_positions_match")
    team_name = ma.team_selector_for_match(match_row, key="phase_avg_positions_team")
    events = data.load_match_events(season=season, match_id=match_row.get("MatchId"), limit=20000)
    if events.empty:
        st.info("No event-level rows are available for this selected match.")
        st.stop()

    team_events = events[events["Team"].astype(str) == str(team_name)].copy() if "Team" in events else events.copy()
    if team_events.empty:
        st.info("No event rows are available for the selected team.")
        st.stop()

    player_minutes = data.load_match_player_minutes(season=season, match_id=match_row.get("MatchId"), team=team_name)
    minute_player_pool = _top_players_by_minutes(player_minutes)
    using_minutes = not minute_player_pool.empty
    if not using_minutes:
        minute_player_pool = _event_involvement_pool(team_events)
    team_events = _attach_match_minutes(team_events, player_minutes)
    player_pool_events = _filter_player_pool(team_events, minute_player_pool)

    phases = _available_phases(team_events)
    phase_options = ["ALL"] + phases
    phase_labels = {"ALL": "All Phases"} | {phase: _phase_label(phase) for phase in phases}

    ma.section_heading("Phase Controls")
    phase_value = st.selectbox(
        "Phase",
        phase_options,
        format_func=lambda value: phase_labels.get(value, _phase_label(value)),
        key="phase_avg_positions_phase",
    )
    st.caption(
        "Available phases come directly from the Impect event feed. This source currently provides In Possession, "
        "Attacking Transition, Second Ball and Set Piece phases; it does not provide continuous out-of-possession tracking positions."
    )
    if using_minutes:
        st.caption(
            f"The map is capped at {MAX_PLAYERS} players. The player pool is selected by total time on the pitch from "
            "CAFC_DB Impect match-player KPI facts, then average locations are calculated from the selected phase event coordinates. "
            "If one of those players has no located event in the chosen phase, fewer than 11 markers may appear."
        )
    else:
        st.caption(
            f"Match-minute KPI data is unavailable for this match. The player pool is based on the top 14 event-involvement "
            f"players (wider than the default 11 to include lower-touch positions). "
            f"Players with fewer than 1 located event in the selected phase may not appear on the map."
        )

    filtered = _filter_phase(player_pool_events, phase_value)
    position_table = _average_position_table(filtered)

    group_label = phase_labels.get(phase_value, _phase_label(phase_value))
    map_title = f"{team_name}: {group_label} Average Positions"
    metric_phase_label = group_label
    caption = (
        "Marker location is each player's average event start location in the selected phase. "
        "Marker size reflects phase action volume; player selection is capped at the 11 highest match-minute players."
    )

else:
    # ---- OPTA DATA FLOW ---------------------------------------------------------
    if data.USE_MOCK_DATA:
        st.warning("Opta feeds are disabled in demo mode. Set CHARLTON_DATA_MODE=production.")
        st.stop()

    fixtures = data.load_opta_fixtures()
    if fixtures.empty:
        st.warning("No Opta fixtures are available to the current Snowflake role.")
        st.stop()

    fixtures = fixtures.sort_values(["Date", "FixtureId"], na_position="last").reset_index(drop=True)
    season_options = sorted(fixtures["Season"].dropna().astype(str).unique().tolist())
    team_options = sorted(
        pd.concat([fixtures["Home"], fixtures["Away"]], ignore_index=True)
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    _opta_charlton_mask = (
        fixtures["Home"].astype(str).str.contains("charlton", case=False, na=False)
        | fixtures["Away"].astype(str).str.contains("charlton", case=False, na=False)
    )

    def _opta_charlton_match_count(season_value: str) -> int:
        return int((_opta_charlton_mask & fixtures["Season"].astype(str).eq(str(season_value))).sum())

    opta_controls = st.columns([0.7, 0.7, 0.9])
    with opta_controls[0]:
        preferred_opta_season = data.preferred_season(season_options, match_count=_opta_charlton_match_count)
        opta_season = st.selectbox(
            "Season", season_options,
            index=season_options.index(preferred_opta_season),
            key="pap_opta_season",
        )
    with opta_controls[1]:
        charlton_idx = next(
            (i + 1 for i, t in enumerate(team_options) if "charlton" in t.casefold()), 0
        )
        opta_team_filter = st.selectbox(
            "Team", ["All teams", *team_options], index=charlton_idx, key="pap_opta_team_filter"
        )

    filtered_fixtures = fixtures[fixtures["Season"].astype(str).eq(str(opta_season))].copy()
    if opta_team_filter != "All teams":
        filtered_fixtures = filtered_fixtures[
            filtered_fixtures["Home"].astype(str).eq(opta_team_filter)
            | filtered_fixtures["Away"].astype(str).eq(opta_team_filter)
        ].copy()
    if filtered_fixtures.empty:
        st.info("No Opta fixtures match those filters.")
        st.stop()

    def _fixture_label(index: int) -> str:
        row = filtered_fixtures.loc[index]
        date = pd.to_datetime(row.get("Date"), errors="coerce")
        date_text = date.strftime("%d %b %Y") if pd.notna(date) else "Undated"
        score = ""
        if pd.notna(row.get("Home Goals")) and pd.notna(row.get("Away Goals")):
            score = f" · {row['Home Goals']:.0f}-{row['Away Goals']:.0f}"
        return f"{date_text} · {row.get('Home')} vs {row.get('Away')}{score}"

    fixture_options = filtered_fixtures.index.tolist()
    selected_fixture_idx = st.selectbox(
        "Fixture",
        fixture_options,
        index=len(fixture_options) - 1,
        format_func=_fixture_label,
        key="pap_opta_fixture",
    )
    selected_fixture = filtered_fixtures.loc[selected_fixture_idx]
    fixture_id = selected_fixture["FixtureId"]

    # Load Opta events first so team names can be resolved consistently.
    events = data.load_opta_events(fixture_id, limit=50000)
    if events.empty:
        st.info("No Opta F24 events are available for this fixture.")
        st.stop()
    events = _normalise_opta_events(events)

    # load_opta_fixtures() sources team names from the raw DVMS fixtures table
    # (e.g. "Charlton Athletic"), while load_opta_events()/load_opta_lineups()
    # source them from the F7 teams table (e.g. "Charlton Athletic FC") -- the
    # same club, different strings. TeamId is consistent across both once the
    # "t" entity prefix used in the fixtures table is stripped, so resolve the
    # display name via the events table before using it for any Team-string
    # filtering (a naive fixtures-table name here always fails to match and
    # silently empties every downstream filter).
    home_id = _opta_id(selected_fixture.get("Home Team Id"))
    away_id = _opta_id(selected_fixture.get("Away Team Id"))
    id_to_name = (
        events[["TeamId", "Team"]].dropna().drop_duplicates().assign(TeamId=lambda d: d["TeamId"].astype(str))
        .set_index("TeamId")["Team"].to_dict()
        if "TeamId" in events else {}
    )
    resolved_home = id_to_name.get(home_id, str(selected_fixture.get("Home", "")))
    resolved_away = id_to_name.get(away_id, str(selected_fixture.get("Away", "")))

    # Pick a team from the fixture
    opta_teams_in_fixture = [name for name in [resolved_home, resolved_away] if name and name.lower() != "nan"]
    if not opta_teams_in_fixture:
        st.warning("The selected fixture has no team names.")
        st.stop()
    team_name = st.selectbox("Team", opta_teams_in_fixture, key="pap_opta_team")

    team_events = events[events["Team"].astype(str) == str(team_name)].copy() if "Team" in events else events.copy()
    if team_events.empty:
        st.info("No Opta event rows are available for the selected team.")
        st.stop()

    lineups = data.load_opta_lineups(fixture_id)
    minute_player_pool = _opta_minute_ranked_pool(lineups, team_name)
    using_minutes = not minute_player_pool.empty
    if not using_minutes:
        minute_player_pool = _event_involvement_pool(team_events)
        st.caption(
            "F7 lineup data is unavailable for this fixture. Falling back to event-involvement player pool. "
            "The formation overlay will not be available."
        )

    # Attach match minutes (all 90 for Opta lineup players)
    team_events["Match Minutes"] = 90.0
    if using_minutes:
        lookup = minute_player_pool[["PlayerId", "Player"]].dropna(subset=["PlayerId"]).drop_duplicates("PlayerId")
        player_ids = set(lookup["PlayerId"].astype(str).unique())
        team_events["Match Minutes"] = team_events["PlayerId"].astype(str).isin(player_ids).astype(float) * 90.0

    player_pool_events = _filter_player_pool(team_events, minute_player_pool)

    # Period selection
    available_periods = _available_periods(team_events)
    period_options = [None] + available_periods
    period_labels = {None: "Both Halves", 1: "First Half", 2: "Second Half", 3: "First Half ET", 4: "Second Half ET"}

    ma.section_heading("Half Controls")
    selected_period = st.selectbox(
        "Half",
        period_options,
        format_func=lambda p: period_labels.get(p, _opta_period_label(p)),
        key="pap_opta_period",
    )
    st.caption(
        "Opta F24 events do not carry Impect phase labels. The grouping dimension is Period (match half). "
        "All event types are included in the average position calculation."
    )

    # Formation overlay toggle
    formations = data.load_opta_formations(fixture_id)
    team_formation = None
    if not formations.empty:
        # Determine which side the selected team is on
        side = "Home" if resolved_home == team_name else "Away"
        formation_row = formations[formations["Side"].astype(str).str.upper() == side.upper()]
        if not formation_row.empty:
            team_formation = str(formation_row.iloc[0].get("Formation", ""))
    show_formation = st.checkbox(
        "Show formation overlay",
        value=bool(team_formation),
        key="pap_opta_formation",
        help="Overlay the official F7 starting formation places as reference markers.",
    )

    filtered = _filter_period(player_pool_events, selected_period)
    position_table = _average_position_table(filtered)

    group_label = period_labels.get(selected_period, "Both Halves")
    map_title = f"{team_name}: {group_label} Average Positions"
    metric_phase_label = group_label
    caption = (
        "Marker location is each player's average event start location. "
        "Marker size reflects event volume; player selection is capped at the 11 highest-minute players."
    )

    # ---- Build the map with optional formation overlay ----
    fig = pitch.average_position_map(filtered, team_name, map_title, min_actions=1)
    if show_formation and team_formation and not lineups.empty:
        team_lineup = lineups[lineups["Team"].astype(str).eq(str(team_name))].copy()
        if not team_lineup.empty:
            formation_dots = pitch.formation_overlay_trace(
                team_lineup, team_name, team_formation, marker_color=pitch.BLUE
            )
            if formation_dots is not None:
                fig.add_trace(formation_dots)
                fig.update_layout(showlegend=True)
    st.plotly_chart(fig, width="stretch")
    st.caption(caption)

# ---- Shared metrics & table (both data sources) ----------------------------------
metric_cols = st.columns(5)
with metric_cols[0]:
    _summary_card("Team", team_name, text_value=True)
with metric_cols[1]:
    _summary_card("Phase / Half", metric_phase_label, text_value=True)
with metric_cols[2]:
    _summary_card("Player pool", len(minute_player_pool))
with metric_cols[3]:
    _summary_card("Players plotted", len(position_table))
with metric_cols[4]:
    _summary_card("Event rows", f"{len(filtered):,}")

if using_impect:
    ma.section_heading(f"{team_name}: {group_label} Average Positions")
    st.plotly_chart(
        pitch.average_position_map(
            filtered,
            team_name,
            map_title,
            min_actions=1,
        ),
        width="stretch",
    )
    st.caption(caption)

ma.section_heading("Average Position Table")
if position_table.empty:
    st.caption("No minute-ranked players have located events in the current selection.")
else:
    st.dataframe(position_table, width="stretch", hide_index=True)

with st.expander("Show player pool"):
    pool_cols = [col for col in ["Player", "Position", "Minutes", "Match Share"] if col in minute_player_pool]
    st.dataframe(minute_player_pool[pool_cols], width="stretch", hide_index=True)
