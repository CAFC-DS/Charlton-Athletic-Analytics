# =============================================================================
# SET PIECE ANALYSIS - report-style pre-match and team-window analysis
# =============================================================================
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import streamlit as st

from utils import data
from utils import match_analysis as ma
from utils import set_piece_analysis as spa
from utils import team_analysis as ta


SET_PIECE_SOURCE = (
    "Restarts and outcomes come from CAFC_DB Impect provider events. One set piece is defined by "
    "MatchId + the provider set-piece ID and main-event marker; shots, goals, xG and second phases "
    "are reconstructed within that full sequence. No iteration-average table is used."
)


def _team_options(matches: pd.DataFrame) -> list[str]:
    values = pd.concat(
        [matches.get("Home", pd.Series(dtype=str)), matches.get("Away", pd.Series(dtype=str))],
        ignore_index=True,
    )
    return sorted(values.dropna().astype(str).loc[lambda series: series.str.strip().ne("")].unique().tolist())


def _default_team_index(teams: list[str]) -> int:
    for index, team in enumerate(teams):
        if "charlton" in team.lower():
            return index
    return 0


def _custom_match_ids(fixtures: pd.DataFrame, team_name: str) -> list[str]:
    labels = {
        str(row["MatchId"]): (
            f"{pd.to_datetime(row['Date']).strftime('%d %b %Y')} · "
            f"{row['Venue']} vs {row['Opponent']}"
        )
        for _, row in fixtures.iterrows()
    }
    options = list(labels)
    return st.multiselect(
        "Custom Matches",
        options,
        default=options,
        format_func=lambda match_id: labels.get(str(match_id), str(match_id)),
        key=f"set_piece_custom_{re.sub(r'[^a-z0-9]+', '_', team_name.lower())}",
    )


def _window_match_ids(fixtures: pd.DataFrame, window: str, team_name: str) -> list[str]:
    if window == "Last 5":
        return fixtures.tail(5)["MatchId"].astype(str).tolist()
    if window == "Last 10":
        return fixtures.tail(10)["MatchId"].astype(str).tolist()
    if window == "Custom":
        return _custom_match_ids(fixtures, team_name)
    return fixtures["MatchId"].astype(str).tolist()


def _selection_caption(fixtures: pd.DataFrame, venue: str, window: str) -> str:
    dates = pd.to_datetime(fixtures.get("Date"), errors="coerce").dropna()
    date_text = "Dates unavailable"
    if not dates.empty:
        date_text = f"{dates.min():%d %b %Y} to {dates.max():%d %b %Y}"
    return f"{len(fixtures)} matches · {date_text} · {venue} venue · {window} window"


def _format_metric(metric: str, value: object) -> str:
    suffix = "%" if "%" in metric else ""
    digits = 1 if suffix else 2
    return spa.value_text(value, digits, suffix)


def _metric_cards(
    league_profiles: pd.DataFrame,
    selected_profile: pd.DataFrame,
    team_name: str,
    metric_specs: list[tuple[str, bool]],
    columns_per_row: int = 4,
) -> None:
    benchmarks = spa.benchmark_metrics(league_profiles, selected_profile, team_name, metric_specs)
    if benchmarks.empty:
        return
    for start in range(0, len(benchmarks), columns_per_row):
        columns = st.columns(columns_per_row)
        chunk = benchmarks.iloc[start : start + columns_per_row]
        for column, (_, row) in zip(columns, chunk.iterrows(), strict=False):
            rank = f"Rank {int(row['Rank'])} of {int(row['Teams'])}" if row["Rank"] else "Rank unavailable"
            column.metric(
                str(row["Metric"]),
                _format_metric(str(row["Metric"]), row["Value"]),
                f"{rank} · Avg {_format_metric(str(row['Metric']), row['League Average'])}",
                delta_color="off",
            )


def _profile_chart(
    league_profiles: pd.DataFrame,
    selected_profile: pd.DataFrame,
    team_name: str,
    metric_specs: list[tuple[str, bool]],
    title: str,
) -> None:
    benchmarks = spa.benchmark_metrics(league_profiles, selected_profile, team_name, metric_specs)
    st.plotly_chart(spa.percentile_profile_chart(benchmarks, title), width="stretch")
    st.caption(
        "Scores are league percentiles. Defensive metrics are inverted where lower is better, so a larger bar always "
        "means a stronger league-relative outcome. The selected match window replaces this team's full-season row."
    )


def _table_pair(left_title: str, left: pd.DataFrame, right_title: str, right: pd.DataFrame) -> None:
    columns = st.columns(2)
    with columns[0]:
        st.markdown(f"#### {left_title}")
        if left.empty:
            st.caption("No qualifying players in this match window.")
        else:
            st.dataframe(left, width="stretch", hide_index=True)
    with columns[1]:
        st.markdown(f"#### {right_title}")
        if right.empty:
            st.caption("No qualifying players in this match window.")
        else:
            st.dataframe(right, width="stretch", hide_index=True)


ta.page_header(
    "Set Piece Analysis",
    "Prepare for opponents and review team set-play performance through restart volume, delivery plans, first contacts, second phases and shot value.",
    SET_PIECE_SOURCE,
    (
        "Swing direction, block location, marking scheme and player movement require tracking or coded video and are not "
        "inferred. First-contact player names are supplied for attacking wins; the provider does not name the defender on "
        "every lost contact. Throw retention means the next team touch after the restart, while a long throw is a 20m+ "
        "forward gain in adjusted coordinates."
    ),
)

with st.expander("Set Piece Analysis Controls", expanded=True):
    available_seasons = data.list_seasons().get("matches", [])
    event_seasons = [season for season in available_seasons if str(season).replace("2025", "25") == "25/26"]
    if not event_seasons:
        st.warning("No event-level 2025/26 season is available for set-piece analysis.")
        st.stop()

    control_columns = st.columns([0.9, 1.35, 0.85, 0.85])
    with control_columns[0]:
        season = st.selectbox("Match Season", event_seasons, key="set_piece_analysis_season")

    matches = ma.load_matches(season)
    if matches.empty:
        st.warning("No match data is available for this season.")
        st.stop()

    teams = _team_options(matches)
    with control_columns[1]:
        team_name = st.selectbox(
            "Team",
            teams,
            index=_default_team_index(teams),
            key="set_piece_analysis_team",
        )

    all_fixtures = spa.team_fixture_rows(matches, team_name)
    with control_columns[2]:
        venue = st.selectbox("Venue", ["All", "Home", "Away"], key="set_piece_analysis_venue")
    with control_columns[3]:
        window = st.selectbox(
            "Match Window",
            ["Last 10", "Last 5", "Full Season", "Custom"],
            key="set_piece_analysis_window",
        )

    venue_fixtures = all_fixtures if venue == "All" else all_fixtures[all_fixtures["Venue"].eq(venue)].copy()
    if venue_fixtures.empty:
        st.info("No fixtures match the current team and venue selection.")
        st.stop()

    selected_match_ids = _window_match_ids(venue_fixtures, window, team_name)
    if not selected_match_ids:
        st.info("Select at least one fixture to build the analysis.")
        st.stop()

    selected_id_set = {str(value) for value in selected_match_ids}
    selected_fixtures = venue_fixtures[venue_fixtures["MatchId"].astype(str).isin(selected_id_set)].copy()
    st.caption(_selection_caption(selected_fixtures, venue, window))

with st.spinner("Building set-piece sequences and league benchmarks..."):
    season_sequences = data.load_set_piece_sequences(season)
    detailed_events = data.load_set_piece_events(season=season, match_ids=selected_match_ids, limit=120000)

if season_sequences.empty:
    st.warning("No provider-defined set-piece sequences are available for this event season.")
    st.stop()

selected_sequences = spa.filter_sequences_to_matches(season_sequences, selected_match_ids)
league_profiles = spa.aggregate_team_profiles(season_sequences, matches)
selected_profile = spa.aggregate_team_profiles(selected_sequences, selected_fixtures, teams=[team_name])
own_sequences = selected_sequences[selected_sequences["Team"].astype(str).eq(str(team_name))].copy()
against_sequences = selected_sequences[selected_sequences["Opponent"].astype(str).eq(str(team_name))].copy()

if detailed_events.empty:
    st.warning("Sequence totals are available, but detailed event locations are unavailable for the selected window.")
elif len(detailed_events) >= 120000:
    st.warning("The detailed pull reached 120,000 rows; pitch maps may not contain every selected restart.")

overview_tab, attack_corner_tab, defend_corner_tab, free_kick_tab, throw_tab = st.tabs(
    ["Overview", "Attacking Corners", "Defending Corners", "Free Kicks", "Throw-Ins"]
)

with overview_tab:
    ta.section_heading("Set-Play Snapshot")
    _metric_cards(
        league_profiles,
        selected_profile,
        team_name,
        spa.OVERVIEW_PROFILE_METRICS,
        columns_per_row=3,
    )
    share_columns = st.columns([1, 2])
    share_columns[0].metric(
        "Share of All Goals from Set Pieces",
        _format_metric("Set-Piece Goal Share %", spa.profile_value(selected_profile, "Set-Piece Goal Share %")),
    )
    share_columns[1].caption(
        "Set-play totals include corners, direct and indirect free kicks, and throw-ins. Goal kicks are excluded. "
        "The percentage compares set-piece goals with the selected team's scoreline goals in the same fixtures."
    )

    ta.section_heading("Game-by-Game Set-Play Threat")
    st.plotly_chart(
        spa.match_xg_trend(selected_sequences, selected_fixtures, team_name, f"{team_name}: Set-Play xG For and Against"),
        width="stretch",
    )

    ta.section_heading("League Profile")
    _profile_chart(
        league_profiles,
        selected_profile,
        team_name,
        spa.OVERVIEW_PROFILE_METRICS,
        f"{team_name}: Overall Set-Play Profile",
    )

    ta.section_heading("Set-Play Shot Locations")
    for_tab, against_tab = st.tabs(["Shots For", "Shots Against"])
    with for_tab:
        st.plotly_chart(
            spa.shot_map(detailed_events, selected_sequences, team_name, f"{team_name}: Set-Play Shots For"),
            width="stretch",
        )
    with against_tab:
        st.plotly_chart(
            spa.shot_map(
                detailed_events,
                selected_sequences,
                team_name,
                f"{team_name}: Set-Play Shots Conceded",
                against=True,
            ),
            width="stretch",
        )

    with st.expander("Show Set-Play Goal Log"):
        goal_tabs = st.tabs(["Goals For", "Goals Against"])
        with goal_tabs[0]:
            st.dataframe(
                spa.goal_log(selected_sequences, selected_fixtures, team_name),
                width="stretch",
                hide_index=True,
            )
        with goal_tabs[1]:
            st.dataframe(
                spa.goal_log(selected_sequences, selected_fixtures, team_name, against=True),
                width="stretch",
                hide_index=True,
            )

with attack_corner_tab:
    attacking_corners = own_sequences[own_sequences["Set Piece Type"].eq("Corner")].copy()
    ta.section_heading("Attacking Corner Snapshot")
    _metric_cards(
        league_profiles,
        selected_profile,
        team_name,
        spa.ATTACKING_CORNER_METRICS,
        columns_per_row=4,
    )
    ta.section_heading("Attacking Corner League Profile")
    _profile_chart(
        league_profiles,
        selected_profile,
        team_name,
        spa.ATTACKING_CORNER_METRICS,
        f"{team_name}: Attacking Corner Profile",
    )
    ta.section_heading("Delivery and First-Contact Maps")
    st.caption(
        "Line colour identifies the delivery family; the endpoint identifies whether the attacking or defending team "
        "won first contact. Coordinates are normalised so the set-piece team attacks to the right."
    )
    map_columns = st.columns(2)
    with map_columns[0]:
        st.plotly_chart(
            spa.delivery_map(
                selected_sequences,
                "Left-Side Attacking Corners",
                {"Corner"},
                team_name=team_name,
                side="Left",
            ),
            width="stretch",
        )
    with map_columns[1]:
        st.plotly_chart(
            spa.delivery_map(
                selected_sequences,
                "Right-Side Attacking Corners",
                {"Corner"},
                team_name=team_name,
                side="Right",
            ),
            width="stretch",
        )
    st.markdown("#### Attacking Corner Delivery by Side")
    st.markdown(f"**Key takeaway:** {spa.delivery_mix_takeaway(attacking_corners)}")
    st.caption(
        "Bars show each delivery's share within left- or right-side corners. Labels give the percentage and number of "
        "corners; colour identifies the taking side, not quality or success."
    )
    st.plotly_chart(
        spa.delivery_mix_chart(attacking_corners),
        width="stretch",
    )
    ta.section_heading("Shot Outcome and Personnel")
    st.plotly_chart(
        spa.shot_map(
            detailed_events,
            selected_sequences,
            team_name,
            f"{team_name}: Shots from Attacking Corners",
            set_piece_types={"Corner"},
        ),
        width="stretch",
    )
    st.plotly_chart(
        spa.outcome_funnel_chart(attacking_corners, f"{team_name}: Attacking Corner Outcome Funnel", {"Corner"}),
        width="stretch",
    )
    _table_pair(
        "Corner Takers",
        spa.taker_table(attacking_corners, {"Corner"}),
        "First-Contact Threats",
        spa.first_contact_table(attacking_corners, {"Corner"}),
    )

with defend_corner_tab:
    faced_corners = against_sequences[against_sequences["Set Piece Type"].eq("Corner")].copy()
    ta.section_heading("Defending Corner Snapshot")
    _metric_cards(
        league_profiles,
        selected_profile,
        team_name,
        spa.DEFENDING_CORNER_METRICS,
        columns_per_row=4,
    )
    ta.section_heading("Defending Corner League Profile")
    _profile_chart(
        league_profiles,
        selected_profile,
        team_name,
        spa.DEFENDING_CORNER_METRICS,
        f"{team_name}: Defending Corner Profile",
    )
    ta.section_heading("Corner Delivery Maps Faced")
    st.caption(
        "The attacking team's adjusted direction is shown. Red endpoints mean the opposition won first contact; "
        "green endpoints mean the defending side won it."
    )
    map_columns = st.columns(2)
    with map_columns[0]:
        st.plotly_chart(
            spa.delivery_map(
                selected_sequences,
                "Left-Side Corners Faced",
                {"Corner"},
                team_name=team_name,
                against=True,
                side="Left",
            ),
            width="stretch",
        )
    with map_columns[1]:
        st.plotly_chart(
            spa.delivery_map(
                selected_sequences,
                "Right-Side Corners Faced",
                {"Corner"},
                team_name=team_name,
                against=True,
                side="Right",
            ),
            width="stretch",
        )
    ta.section_heading("Shots Conceded from Corners")
    st.plotly_chart(
        spa.shot_map(
            detailed_events,
            selected_sequences,
            team_name,
            f"{team_name}: Corner Shots Conceded",
            against=True,
            set_piece_types={"Corner"},
        ),
        width="stretch",
    )
    with st.expander("Show Opposition Corner Takers and First Contacts"):
        _table_pair(
            "Opposition Corner Takers",
            spa.taker_table(faced_corners, {"Corner"}),
            "Opposition First-Contact Threats",
            spa.first_contact_table(faced_corners, {"Corner"}),
        )

with free_kick_tab:
    own_free_kicks = own_sequences[own_sequences["Set Piece Type"].isin(spa.FREE_KICK_TYPES)].copy()
    ta.section_heading("Free-Kick Snapshot")
    _metric_cards(
        league_profiles,
        selected_profile,
        team_name,
        spa.FREE_KICK_METRICS,
        columns_per_row=4,
    )
    ta.section_heading("Free-Kick League Profile")
    _profile_chart(
        league_profiles,
        selected_profile,
        team_name,
        spa.FREE_KICK_METRICS,
        f"{team_name}: Free-Kick Profile",
    )
    ta.section_heading("Indirect Free-Kick Delivery Map")
    st.plotly_chart(
        spa.delivery_map(
            selected_sequences,
            f"{team_name}: Indirect Free-Kick Deliveries",
            {"Indirect Free Kick"},
            team_name=team_name,
        ),
        width="stretch",
    )
    ta.section_heading("Free-Kick Shot Locations")
    direct_tab, all_fk_tab = st.tabs(["Direct Free Kicks", "All Free-Kick Shots"])
    with direct_tab:
        st.plotly_chart(
            spa.shot_map(
                detailed_events,
                selected_sequences,
                team_name,
                f"{team_name}: Direct Free-Kick Attempts",
                set_piece_types={"Direct Free Kick"},
            ),
            width="stretch",
        )
    with all_fk_tab:
        st.plotly_chart(
            spa.shot_map(
                detailed_events,
                selected_sequences,
                team_name,
                f"{team_name}: All Free-Kick Shots",
                set_piece_types=spa.FREE_KICK_TYPES,
            ),
            width="stretch",
        )
    st.plotly_chart(
        spa.outcome_funnel_chart(own_free_kicks, f"{team_name}: Free-Kick Outcome Funnel", spa.FREE_KICK_TYPES),
        width="stretch",
    )
    _table_pair(
        "Free-Kick Takers",
        spa.taker_table(own_free_kicks, spa.FREE_KICK_TYPES),
        "First-Contact Threats",
        spa.first_contact_table(own_free_kicks, spa.FREE_KICK_TYPES),
    )

with throw_tab:
    own_throws = own_sequences[own_sequences["Set Piece Type"].eq("Throw-In")].copy()
    ta.section_heading("Throw-In Snapshot")
    _metric_cards(
        league_profiles,
        selected_profile,
        team_name,
        spa.THROW_IN_METRICS,
        columns_per_row=4,
    )
    ta.section_heading("Throw-In League Profile")
    _profile_chart(
        league_profiles,
        selected_profile,
        team_name,
        spa.THROW_IN_METRICS,
        f"{team_name}: Throw-In Profile",
    )
    ta.section_heading("Throw-In Territory and Retention")
    st.caption(
        "Green means the selected team made the next team touch; red means possession moved to the opponent. "
        "Solid thick lines are long throws with at least 20m of forward gain; dotted lines are standard throws."
    )
    st.plotly_chart(
        spa.throw_in_map(selected_sequences, f"{team_name}: Throw-In Map", team_name=team_name),
        width="stretch",
    )
    ta.section_heading("Throw-In Shot Value")
    st.plotly_chart(
        spa.shot_map(
            detailed_events,
            selected_sequences,
            team_name,
            f"{team_name}: Shots Following Throw-Ins",
            set_piece_types={"Throw-In"},
        ),
        width="stretch",
    )
    st.plotly_chart(
        spa.outcome_funnel_chart(own_throws, f"{team_name}: Throw-In Outcome Funnel", {"Throw-In"}),
        width="stretch",
    )
    _table_pair(
        "Throwers",
        spa.taker_table(own_throws, {"Throw-In"}),
        "First-Contact Threats",
        spa.first_contact_table(own_throws, {"Throw-In"}),
    )

ta.section_heading("Terminology and Method Key")
with st.expander("Open Set-Piece Terminology Key"):
    st.markdown(
        """
        - **Set piece:** one provider-defined restart identified by `MatchId` and `setPieceId`; goal kicks are excluded from headline set-play figures.
        - **First contact won:** inferred from the provider receiver identity/type on the restart; it is unknown when no receiver is supplied. A defensive first-contact win is the inverse on corners faced.
        - **Second phase:** a shot or goal after the initial set-piece phase, identified from provider phase/subphase values within the same possession sequence.
        - **Ending in shot:** at least one shot occurred anywhere inside that restart sequence; it is a sequence conversion rate, not shot conversion.
        - **Box free kick:** an indirect free kick whose provider type or end zone indicates a cross/high delivery into the penalty area; short possession restarts are excluded.
        - **Direct free kick:** the provider main event is a shot or the free-kick type contains a shot classification.
        - **Throw-in retention:** the restarting team made the next team touch after the throw. It does not guarantee a completed possession.
        - **Long throw:** adjusted end X minus start X is at least 20 metres. Coordinates are normalised to the restarting team's attacking direction.
        - **League percentile:** the selected window replaces the chosen team's full-season row; every other club retains its full-season profile. Higher bars always represent a stronger outcome after direction is accounted for.
        """
    )

with st.expander("Show Selected Set-Piece Sequence Data"):
    st.dataframe(spa.match_sequence_table(selected_sequences), width="stretch", hide_index=True)
