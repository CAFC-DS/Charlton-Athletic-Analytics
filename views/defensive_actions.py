# =============================================================================
# DEFENSIVE ACTIONS - team defensive identity, outcomes and player contribution
# =============================================================================
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from utils import data
from utils import defensive_analysis as da
from utils import match_analysis as ma
from utils import team_analysis as ta


DEFENSIVE_SOURCE = (
    "Team and player totals come from CAFC_DB Impect match-level KPI facts. Pitch "
    "locations and post-regain sequences come from provider events through the app's "
    "event adapter. No aggregated iteration-average table is used."
)

SNAPSHOT_METRICS = [
    ("Ball Wins / Match", "Ball Wins / Match", True, 1, ""),
    ("Opponents Removed / Win", "Opponents Removed / Ball Win", True, 2, ""),
    ("Presses / Match", "Presses / Match", True, 1, ""),
    ("Second-Ball Win %", "Second-Ball Win %", True, 1, "%"),
    ("Duel Win %", "Duel Win %", True, 1, "%"),
    ("xG Conceded / Match", "xG Conceded / Match", False, 2, ""),
]

TREND_METRICS = [
    "Ball Wins / Match",
    "Presses / Match",
    "Opponents Removed / Ball Win",
    "Second-Ball Win %",
    "Ground Duel Win %",
    "Aerial Duel Win %",
    "xG Conceded / Match",
]


def _team_options(matches: pd.DataFrame) -> list[str]:
    values = pd.concat([matches.get("Home", pd.Series(dtype=str)), matches.get("Away", pd.Series(dtype=str))])
    return sorted(values.dropna().astype(str).loc[lambda series: series.str.strip().ne("")].unique().tolist())


def _default_team_index(teams: list[str]) -> int:
    for index, team in enumerate(teams):
        if "charlton" in team.lower():
            return index
    return 0


def _normalise_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_convert(None)


def _team_fixture_rows(matches: pd.DataFrame, team_name: str) -> pd.DataFrame:
    rows = matches[
        matches["Home"].astype(str).eq(str(team_name)) | matches["Away"].astype(str).eq(str(team_name))
    ].copy()
    rows["Date"] = _normalise_dates(rows["Date"])
    rows["Venue"] = np.where(rows["Home"].astype(str).eq(str(team_name)), "Home", "Away")
    rows["Opponent"] = np.where(
        rows["Home"].astype(str).eq(str(team_name)),
        rows["Away"].astype(str),
        rows["Home"].astype(str),
    )
    return rows.sort_values(["Date", "MatchId"]).reset_index(drop=True)


def _format_value(value: object, digits: int = 1, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{number:,.{digits}f}{suffix}"


def _metric_cards(
    league_summary: pd.DataFrame,
    selected_summary: pd.DataFrame,
    team_name: str,
) -> None:
    for start in range(0, len(SNAPSHOT_METRICS), 3):
        columns = st.columns(3)
        for column, (label, metric, higher_is_better, digits, suffix) in zip(
            columns,
            SNAPSHOT_METRICS[start : start + 3],
            strict=False,
        ):
            benchmark = da.metric_benchmark(
                league_summary,
                selected_summary,
                team_name,
                metric,
                higher_is_better,
            )
            rank_text = (
                f"Rank {benchmark['rank']} of {benchmark['teams']} · "
                f"League avg {_format_value(benchmark['average'], digits, suffix)}"
                if benchmark["rank"]
                else "League comparison unavailable"
            )
            column.metric(
                label,
                _format_value(benchmark["value"], digits, suffix),
                rank_text,
                delta_color="off",
            )


def _selected_match_ids(
    fixture_rows: pd.DataFrame,
    window: str,
    team_name: str,
) -> list[str]:
    if window == "Last 5":
        return fixture_rows.tail(5)["MatchId"].astype(str).tolist()
    if window == "Last 10":
        return fixture_rows.tail(10)["MatchId"].astype(str).tolist()
    if window != "Custom":
        return fixture_rows["MatchId"].astype(str).tolist()

    label_lookup = {
        str(row["MatchId"]): (
            f"{pd.to_datetime(row['Date']).strftime('%d %b %Y')} · "
            f"{row['Venue']} vs {row['Opponent']}"
        )
        for _, row in fixture_rows.iterrows()
    }
    options = list(label_lookup)
    return st.multiselect(
        "Custom Matches",
        options,
        default=options,
        format_func=lambda match_id: label_lookup.get(str(match_id), str(match_id)),
        key=f"defensive_actions_custom_matches_{team_name}",
    )


def _selection_caption(selected_fixtures: pd.DataFrame, venue: str, window: str) -> str:
    if selected_fixtures.empty:
        return "No matches selected."
    dates = _normalise_dates(selected_fixtures["Date"]).dropna()
    date_text = "Dates unavailable"
    if not dates.empty:
        date_text = f"{dates.min():%d %b %Y} to {dates.max():%d %b %Y}"
    return f"{len(selected_fixtures)} matches · {date_text} · {venue} venue · {window} window"


ta.page_header(
    "Defensive Actions",
    "Assess defensive identity, regain territory, pressure context, duel outcomes and player contribution against league benchmarks.",
    DEFENSIVE_SOURCE,
    (
        "The regain map and post-regain funnel use event-based spatial/sequence proxies and may not reconcile "
        "one-for-one with IMPECT's provider-defined Ball Win total. True defensive-line height and compactness "
        "require tracking data and are therefore not inferred here."
    ),
)

ta.section_heading("Defensive Analysis Controls")
control_columns = st.columns([1.0, 1.35, 0.9, 0.9])
with control_columns[0]:
    defensive_seasons = data.list_seasons().get("matches", [])
    if not defensive_seasons:
        st.warning("No match seasons are available.")
        st.stop()
    preferred_season = "25/26" if "25/26" in defensive_seasons else defensive_seasons[-1]
    season = st.selectbox(
        "Match Season",
        defensive_seasons,
        index=defensive_seasons.index(preferred_season),
        key="defensive_actions_season",
    )

matches = ma.load_matches(season)
if matches.empty:
    st.warning("No match data is available for the selected season.")
    st.stop()

teams = _team_options(matches)
if not teams:
    st.warning("No teams are available from the selected match data.")
    st.stop()

with control_columns[1]:
    team_name = st.selectbox(
        "Team",
        teams,
        index=_default_team_index(teams),
        key="defensive_actions_team",
    )

all_team_fixtures = _team_fixture_rows(matches, team_name)
if all_team_fixtures.empty:
    st.warning("No fixtures are available for the selected team.")
    st.stop()

with control_columns[2]:
    venue = st.selectbox("Venue", ["All", "Home", "Away"], key="defensive_actions_venue")
with control_columns[3]:
    window = st.selectbox(
        "Match Window",
        ["Full Season", "Last 5", "Last 10", "Custom"],
        key="defensive_actions_window",
    )

venue_fixtures = all_team_fixtures.copy()
if venue != "All":
    venue_fixtures = venue_fixtures[venue_fixtures["Venue"].eq(venue)].copy()
if venue_fixtures.empty:
    st.info(f"No {venue.lower()} fixtures are available for this team and season.")
    st.stop()

selected_match_ids = _selected_match_ids(venue_fixtures, window, team_name)
if not selected_match_ids:
    st.info("Select at least one match to analyse.")
    st.stop()

selected_match_id_set = {str(match_id) for match_id in selected_match_ids}
selected_fixtures = venue_fixtures[venue_fixtures["MatchId"].astype(str).isin(selected_match_id_set)].copy()
st.caption(_selection_caption(selected_fixtures, venue, window))

with st.spinner("Loading trusted defensive match totals..."):
    squad_rows = data.load_squad_defensive_match_sums(season)
    player_rows = data.load_player_defensive_match_sums(season)

if squad_rows.empty:
    st.warning(
        "No Impect squad defensive match-level KPI facts are available for this season. "
        "Choose a season covered by the CAFC_DB Impect match-level squad KPI facts."
    )
    st.stop()

league_summary = da.aggregate_teams(squad_rows)
selected_team_rows = squad_rows[
    squad_rows["Team"].astype(str).eq(str(team_name))
    & squad_rows["MatchId"].astype(str).isin(selected_match_id_set)
].copy()
if selected_team_rows.empty:
    st.warning("The selected fixtures do not have squad defensive match-level KPI rows.")
    st.stop()

selected_summary = da.aggregate_teams(selected_team_rows)
selected_player_rows = player_rows[
    player_rows["Team"].astype(str).eq(str(team_name))
    & player_rows["MatchId"].astype(str).isin(selected_match_id_set)
].copy()
player_summary = da.aggregate_players(selected_player_rows)
match_rows = da.add_match_context(selected_team_rows, matches, team_name)

ta.section_heading("Defensive Snapshot")
_metric_cards(league_summary, selected_summary, team_name)
st.caption(
    "The selected match window replaces this team's full-season value inside the league ranking. "
    "League averages remain full-season rates. Press volume is descriptive: more presses do not automatically mean better defending."
)

ta.section_heading("Defensive Identity")
identity_rows = da.identity_components(league_summary, selected_summary, team_name)
st.caption(
    "High describes where possession is won, Active describes defensive pressure and second-ball activity, "
    "and Tight describes how effectively progression, goals and xG are restricted. Each score is the mean "
    "of three league percentiles. A higher score means a stronger expression of that identity, not a better "
    "overall defensive grade."
)
st.html(
    """
    <div style="display:flex; flex-wrap:wrap; align-items:center; gap:12px 22px; padding:10px 14px; margin:4px 0 12px; border:1px solid #e6edf5; border-radius:8px; background:#ffffff; font-size:0.88rem;">
        <strong>Profile Colour Key:</strong>
        <span style="display:inline-flex; align-items:center; gap:7px;"><span style="width:12px; height:12px; border-radius:3px; background:#16803c; display:inline-block;"></span><span>Upper third &middot; Score 67+</span></span>
        <span style="display:inline-flex; align-items:center; gap:7px;"><span style="width:12px; height:12px; border-radius:3px; background:#d89216; display:inline-block;"></span><span>Middle third &middot; Score 33&ndash;66.9</span></span>
        <span style="display:inline-flex; align-items:center; gap:7px;"><span style="width:12px; height:12px; border-radius:3px; background:#c30017; display:inline-block;"></span><span>Lower third &middot; Score below 33</span></span>
        <span style="display:inline-flex; align-items:center; gap:7px;"><span style="width:30px; border-top:2px dashed #7a7f87; display:inline-block;"></span><span>League midpoint &middot; Score 50</span></span>
    </div>
    """
)

st.plotly_chart(
    da.identity_chart(identity_rows, f"{team_name}: High, Active and Tight Defensive Profile"),
    width="stretch",
)
with st.expander("Show Defensive Identity Components"):
    identity_table = identity_rows.copy()
    for column in ["Selected Value", "League Average", "Percentile"]:
        identity_table[column] = pd.to_numeric(identity_table[column], errors="coerce").round(2)
    st.dataframe(identity_table, width="stretch", hide_index=True)

ta.section_heading("Regain Territory")
map_view = st.selectbox(
    "Regain Map View",
    [
        "All Regains",
        "Opposition-Half Regains",
        "Final-Third Regains",
        "Second-Ball Regains",
        "Interceptions",
    ],
    key="defensive_actions_regain_view",
)
events = data.load_match_events(
    season=season,
    team=team_name,
    match_ids=selected_match_ids,
    limit=120000,
)
if len(events) >= 120000:
    st.warning("The selected-window event pull reached its 120,000-row cap, so spatial and sequence visuals may be incomplete.")
all_regains = da.filter_regain_events(events, "All Regains")
mapped_regains = da.filter_regain_events(events, map_view)

regain_columns = st.columns(3)
average_height = pd.to_numeric(all_regains.get("Start X", pd.Series(dtype=float)), errors="coerce").mean() + 52.5
opposition_half_share = (
    pd.to_numeric(all_regains.get("Start X", pd.Series(dtype=float)), errors="coerce").ge(0).mean() * 100
    if not all_regains.empty
    else np.nan
)
final_third_share = (
    pd.to_numeric(all_regains.get("Start X", pd.Series(dtype=float)), errors="coerce").ge(17.5).mean() * 100
    if not all_regains.empty
    else np.nan
)
regain_columns[0].metric("Average Regain Height", _format_value(average_height, 1, "m"))
regain_columns[1].metric("Opposition-Half Share", _format_value(opposition_half_share, 1, "%"))
regain_columns[2].metric("Final-Third Share", _format_value(final_third_share, 1, "%"))
st.caption(
    "Coordinates are normalised so the selected team attacks from left to right. The heatmap uses loose-ball regains, "
    "interceptions and goalkeeper catches with valid locations. Exact provider Ball Wins remain in the cards above."
)
st.plotly_chart(
    da.regain_density_map(mapped_regains, f"{team_name}: {map_view}"),
    width="stretch",
)

ta.section_heading("Pressure Context")
st.caption(
    "These are IMPECT's provider-defined pressure contexts per match, not counts inferred from the event Pressure field. "
    "Context categories can overlap with the all-press total."
)
st.plotly_chart(
    da.pressing_context_chart(
        selected_summary,
        league_summary,
        team_name,
        f"{team_name}: Pressure Context vs League Average",
    ),
    width="stretch",
)

ta.section_heading("What Happens After a Regain?")
sequence_summary, sequence_xg = da.regain_sequence_outcomes(events)
sequence_metrics = st.columns(3)
sequence_count = int(sequence_summary["Sequences"].iloc[0]) if not sequence_summary.empty else 0
shot_sequences = (
    int(sequence_summary.loc[sequence_summary["Stage"].eq("Produced a Shot"), "Sequences"].sum())
    if not sequence_summary.empty
    else 0
)
sequence_metrics[0].metric("Regain Sequences", f"{sequence_count:,}")
sequence_metrics[1].metric("Shot-Producing Sequences", f"{shot_sequences:,}")
sequence_metrics[2].metric("xG After Regains", _format_value(sequence_xg, 2))
st.caption(
    "A regain is linked only to later actions with the same MatchId and Sequence Index. This measures attacking "
    "outcome after the event-based regain proxy; it does not claim that every provider Ball Win starts a new sequence."
)
st.plotly_chart(
    da.regain_conversion_chart(sequence_summary, f"{team_name}: Regain-to-Attack Conversion"),
    width="stretch",
)

ta.section_heading("Duels, Second Balls and Ball-Win Outcomes")
if player_summary.empty:
    st.info("No player defensive match-level KPI facts are available for the selected fixtures.")
else:
    ground_tab, aerial_tab, second_ball_tab, win_loss_tab = st.tabs(
        ["Ground Duels", "Aerial Duels", "Second Balls", "Ball Wins and Losses"]
    )
    with ground_tab:
        st.caption("Right of zero shows ground duels won; left of zero shows ground duels lost. Bars use trusted player match-level KPI facts.")
        st.plotly_chart(
            da.player_diverging_chart(
                player_summary,
                "Ground Duels Won",
                "Ground Duels Lost",
                f"{team_name}: Player Ground-Duel Outcomes",
            ),
            width="stretch",
        )
    with aerial_tab:
        st.caption("Right of zero shows aerial duels won; left of zero shows aerial duels lost.")
        st.plotly_chart(
            da.player_diverging_chart(
                player_summary,
                "Aerial Duels Won",
                "Aerial Duels Lost",
                f"{team_name}: Player Aerial-Duel Outcomes",
            ),
            width="stretch",
        )
    with second_ball_tab:
        st.caption("Colour and bar length both use a fixed 0-100% win-rate scale; hover shows won and contested totals.")
        st.plotly_chart(
            da.second_ball_player_chart(player_summary, f"{team_name}: Player Second-Ball Win Rate"),
            width="stretch",
        )
    with win_loss_tab:
        st.caption("Right of zero shows provider Ball Wins; left of zero shows provider Ball Losses.")
        st.plotly_chart(
            da.player_diverging_chart(
                player_summary,
                "Ball Wins",
                "Ball Losses",
                f"{team_name}: Player Ball Wins and Losses",
                won_label="Ball Wins",
                lost_label="Ball Losses",
            ),
            width="stretch",
        )

ta.section_heading("Player Defensive Contribution")
if player_summary.empty:
    st.info("No player contribution data is available for the selected fixtures.")
else:
    max_minutes = max(float(pd.to_numeric(player_summary["Minutes"], errors="coerce").max()), 90.0)
    slider_max = max(90, int(math.ceil(max_minutes / 30) * 30))
    default_minutes = min(slider_max, max(60, min(450, len(selected_match_ids) * 30)))
    minimum_minutes = st.slider(
        "Minimum Minutes",
        min_value=0,
        max_value=slider_max,
        value=int(default_minutes),
        step=30,
        key="defensive_actions_min_minutes",
    )
    st.caption(
        "Further right means more Ball Wins /90; higher means each win removes more opponents. Bubble size shows minutes. "
        "Colour uses a fixed 0-100% combined duel-win scale, and dashed lines are selected-squad averages."
    )
    st.plotly_chart(
        da.player_contribution_scatter(
            player_summary,
            minimum_minutes,
            f"{team_name}: Player Defensive Volume and Effect",
        ),
        width="stretch",
    )

ta.section_heading("Match-by-Match Trend")
trend_metric = st.selectbox("Trend Metric", TREND_METRICS, key="defensive_actions_trend_metric")
st.plotly_chart(
    da.match_trend_chart(match_rows, trend_metric, f"{team_name}: {trend_metric} by Match"),
    width="stretch",
)

with st.expander("Show Match-by-Match Defensive Data"):
    table_columns = [
        "Date",
        "Match Label",
        "Ball Wins",
        "Ball Losses",
        "Opponents Removed",
        "Presses",
        "Counterpresses",
        "Second Balls",
        "Second Balls Won",
        "Ground Duels Won",
        "Ground Duels Lost",
        "Aerial Duels Won",
        "Aerial Duels Lost",
        "Goals Conceded",
        "xG Conceded",
    ]
    match_table = match_rows[[column for column in table_columns if column in match_rows]].copy()
    if "Date" in match_table:
        match_table["Date"] = pd.to_datetime(match_table["Date"], errors="coerce").dt.strftime("%d %b %Y")
    st.dataframe(match_table, width="stretch", hide_index=True)

ta.section_heading("Terminology and Methodology Key")
terminology = pd.DataFrame(
    [
        ("Ball Win", "Impect provider-defined regained possession total from match-level KPI facts."),
        ("Opponents Removed / Win", "Total opponents bypassed by Ball Wins divided by Ball Wins; an effect/packing measure."),
        ("Press", "IMPECT provider-defined pressure action. Context columns describe counterpress, build-up and between-lines situations."),
        ("Second-Ball Win %", "Second Balls Won divided by Second Ball Starts across the selected matches."),
        ("Duel Win %", "Ground and aerial duels won divided by all recorded won and lost ground/aerial duels."),
        ("High", "League-percentile blend of final-third Ball Win volume, final-third share and opponent-box Ball Wins."),
        ("Active", "League-percentile blend of presses, counterpresses and Second-Ball Win %."),
        ("Tight", "League-percentile blend of lower xG conceded, lower goals conceded and fewer suffered bypassed opponents."),
        ("Regain Map", "Event-location proxy using loose-ball regains, interceptions and goalkeeper catches; not the provider Ball Win total."),
        ("Regain Sequence", "Events after a spatial regain that share the same match and possession-sequence identifier."),
        ("xG Conceded", "Impect conceded-shot xG from squad match-level KPI facts."),
    ],
    columns=["Term", "Meaning on This Page"],
)
st.dataframe(terminology, width="stretch", hide_index=True)
st.markdown(
    "The identity structure is informed by the IMPECT-based "
    "[CIES High / Active / Tight team-style framework](https://football-observatory.com/IMG/pdf/team_profil_en.pdf). "
    "The app uses the exact available components listed above rather than claiming unavailable tracking measures. "
    "For transition context, see FIFA's work on "
    "[regaining possession high up the pitch](https://www.fifatrainingcentre.com/en/game/game-analysis/transition-to-defending/regaining-possession-high-up-the-pitch.php) "
    "and StatsBomb's distinction between "
    "[counterpress process and outcome](https://statsbomb.com/articles/soccer/how-statsbomb-data-helps-measure-counter-pressing/)."
)
