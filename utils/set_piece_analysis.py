"""Set-piece calculations and report-style visuals.

The module keeps two grains deliberately separate:

* one row per provider-defined set piece for rates and league comparisons;
* detailed events from selected fixtures for shot and trajectory maps.

No iteration-average or aggregated-average table is used here. A restart is
identified by ``MatchId`` + ``Set Piece ID`` and its provider main-event flag.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils import charting, pitch, ui


RED = ui.CHARLTON_RED
DARK = ui.CHARLTON_BLACK
GREEN = "#16803c"
AMBER = "#d89216"
GREY = "#7a7f87"
BLUE = "#2563eb"
PURPLE = "#7c3aed"
TEAL = "#0f766e"
LIGHT_GREY = "#e6edf5"

ANALYSED_TYPES = {"Corner", "Direct Free Kick", "Indirect Free Kick", "Throw-In"}
CORNER_TYPES = {"Corner"}
FREE_KICK_TYPES = {"Direct Free Kick", "Indirect Free Kick"}
THROW_TYPES = {"Throw-In"}

OVERVIEW_PROFILE_METRICS = [
    ("Set-Play Goals /90", True),
    ("Set-Play xG /90", True),
    ("Set-Play Goals Conceded /90", False),
    ("Set-Play xG Conceded /90", False),
    ("Set-Play Shots /90", True),
    ("Set-Play Shots Conceded /90", False),
]

ATTACKING_CORNER_METRICS = [
    ("Corners /90", True),
    ("Corner xG /90", True),
    ("Corner Goals /90", True),
    ("Corner Shots /90", True),
    ("Corner First Contact Won %", True),
    ("Corners Ending in Shot %", True),
    ("Corner Second-Phase xG /90", True),
]

DEFENDING_CORNER_METRICS = [
    ("Corners Faced /90", False),
    ("Corner xG Conceded /90", False),
    ("Corner Goals Conceded /90", False),
    ("Corner Shots Conceded /90", False),
    ("Defensive Corner First Contact Won %", True),
    ("Opponent Corners Ending in Shot %", False),
    ("Corner Second-Phase xG Conceded /90", False),
]

FREE_KICK_METRICS = [
    ("Box Free Kicks /90", True),
    ("Free Kick xG /90", True),
    ("Free Kick Goals /90", True),
    ("Free Kick Shots /90", True),
    ("Free Kick First Contact Won %", True),
    ("Direct Free Kicks /90", True),
    ("Free Kick xG Conceded /90", False),
]

THROW_IN_METRICS = [
    ("Throw-Ins /90", True),
    ("Long-Throw Share %", True),
    ("Throw-In Retention %", True),
    ("Throw-Ins Ending in Shot %", True),
    ("Throw-In xG /90", True),
    ("Throw-In xG Conceded /90", False),
    ("Opponent Throw-In Retention %", False),
]


def _number(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _sum(rows: pd.DataFrame, column: str) -> float:
    if rows.empty or column not in rows:
        return 0.0
    return float(_number(rows[column]).fillna(0).sum())


def _per_match(value: float, matches: int) -> float:
    return float(value) / matches if matches else np.nan


def _percentage(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) * 100 if denominator else np.nan


def _bool_count(values: pd.Series, expected: bool) -> int:
    if values.empty:
        return 0
    return int(values.astype("boolean").eq(expected).fillna(False).sum())


def _fixture_team_names(matches: pd.DataFrame) -> list[str]:
    if matches.empty:
        return []
    values = pd.concat(
        [
            matches.get("Home", pd.Series(dtype="string")),
            matches.get("Away", pd.Series(dtype="string")),
        ],
        ignore_index=True,
    )
    return sorted(values.dropna().astype(str).loc[lambda series: series.str.strip().ne("")].unique().tolist())


def team_fixture_rows(matches: pd.DataFrame, team_name: str) -> pd.DataFrame:
    """Return one fixture row per match from the selected team's perspective."""
    if matches.empty or not {"Home", "Away"}.issubset(matches.columns):
        return pd.DataFrame()
    rows = matches[
        matches["Home"].astype(str).eq(str(team_name))
        | matches["Away"].astype(str).eq(str(team_name))
    ].copy()
    if rows.empty:
        return rows
    rows["Date"] = pd.to_datetime(rows.get("Date"), errors="coerce", utc=True).dt.tz_convert(None)
    is_home = rows["Home"].astype(str).eq(str(team_name))
    rows["Venue"] = np.where(is_home, "Home", "Away")
    rows["Opponent"] = np.where(is_home, rows["Away"], rows["Home"])
    rows["Goals For"] = np.where(is_home, rows.get("Home Goals", np.nan), rows.get("Away Goals", np.nan))
    rows["Goals Against"] = np.where(is_home, rows.get("Away Goals", np.nan), rows.get("Home Goals", np.nan))
    return rows.sort_values(["Date", "MatchId"], na_position="last").reset_index(drop=True)


def filter_sequences_to_matches(sequences: pd.DataFrame, match_ids: list[object] | set[object]) -> pd.DataFrame:
    if sequences.empty or not match_ids:
        return sequences.iloc[0:0].copy()
    wanted = {str(value) for value in match_ids}
    return sequences[sequences["MatchId"].astype(str).isin(wanted)].copy()


def _box_free_kick_mask(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(False, index=rows.index)
    type_text = rows.get("Free Kick Type", pd.Series("", index=rows.index)).astype("string").str.upper()
    end_zone = rows.get("Free Kick End Zone", pd.Series("", index=rows.index)).astype("string").str.upper()
    indirect = rows["Set Piece Type"].astype(str).eq("Indirect Free Kick")
    into_box = (
        type_text.str.contains("CROSS|HIGH_BALL|BOX", regex=True, na=False)
        | end_zone.str.contains("POST|BOX|CENTRAL_CLOSE|CENTRAL_WIDE|GOAL", regex=True, na=False)
    )
    recycle = type_text.str.contains("POSSESSION|SHORT", regex=True, na=False)
    return indirect & into_box & ~recycle


def _profile_row(
    sequences: pd.DataFrame,
    fixtures: pd.DataFrame,
    team_name: str,
) -> dict[str, object]:
    match_ids = set(fixtures.get("MatchId", pd.Series(dtype=object)).astype(str))
    rows = filter_sequences_to_matches(sequences, match_ids) if match_ids else sequences.iloc[0:0].copy()
    rows = rows[rows["Set Piece Type"].astype(str).isin(ANALYSED_TYPES)].copy()
    own = rows[rows["Team"].astype(str).eq(str(team_name))].copy()
    against = rows[rows["Opponent"].astype(str).eq(str(team_name))].copy()
    matches = int(fixtures["MatchId"].astype(str).nunique()) if "MatchId" in fixtures else 0

    own_corners = own[own["Set Piece Type"].isin(CORNER_TYPES)]
    faced_corners = against[against["Set Piece Type"].isin(CORNER_TYPES)]
    own_free_kicks = own[own["Set Piece Type"].isin(FREE_KICK_TYPES)]
    faced_free_kicks = against[against["Set Piece Type"].isin(FREE_KICK_TYPES)]
    direct_free_kicks = own[own["Set Piece Type"].eq("Direct Free Kick")]
    own_throws = own[own["Set Piece Type"].isin(THROW_TYPES)]
    faced_throws = against[against["Set Piece Type"].isin(THROW_TYPES)]

    corner_contacts = own_corners["First Touch Won"].dropna()
    faced_corner_contacts = faced_corners["First Touch Won"].dropna()
    free_kick_contacts = own_free_kicks["First Touch Won"].dropna()

    goals_for = _number(fixtures.get("Goals For", pd.Series(dtype=float))).fillna(0).sum()
    own_set_play_goals = _sum(own, "Goals")

    return {
        "Team": team_name,
        "Matches": matches,
        "Set-Play Goals /90": _per_match(own_set_play_goals, matches),
        "Set-Play xG /90": _per_match(_sum(own, "xG"), matches),
        "Set-Play Shots /90": _per_match(_sum(own, "Shots"), matches),
        "Set-Play Goals Conceded /90": _per_match(_sum(against, "Goals"), matches),
        "Set-Play xG Conceded /90": _per_match(_sum(against, "xG"), matches),
        "Set-Play Shots Conceded /90": _per_match(_sum(against, "Shots"), matches),
        "Set-Piece Goal Share %": _percentage(own_set_play_goals, float(goals_for)),
        "Corners /90": _per_match(len(own_corners), matches),
        "Corner Goals /90": _per_match(_sum(own_corners, "Goals"), matches),
        "Corner xG /90": _per_match(_sum(own_corners, "xG"), matches),
        "Corner Shots /90": _per_match(_sum(own_corners, "Shots"), matches),
        "Corner First Contact Won %": _percentage(
            _bool_count(corner_contacts, True), len(corner_contacts)
        ),
        "Corners Ending in Shot %": _percentage(
            int(_number(own_corners.get("Shots", pd.Series(dtype=float))).fillna(0).gt(0).sum()),
            len(own_corners),
        ),
        "Corner Second-Phase Shots /90": _per_match(_sum(own_corners, "Second-Phase Shots"), matches),
        "Corner Second-Phase xG /90": _per_match(_sum(own_corners, "Second-Phase xG"), matches),
        "Corners Faced /90": _per_match(len(faced_corners), matches),
        "Corner Goals Conceded /90": _per_match(_sum(faced_corners, "Goals"), matches),
        "Corner xG Conceded /90": _per_match(_sum(faced_corners, "xG"), matches),
        "Corner Shots Conceded /90": _per_match(_sum(faced_corners, "Shots"), matches),
        "Defensive Corner First Contact Won %": _percentage(
            _bool_count(faced_corner_contacts, False), len(faced_corner_contacts)
        ),
        "Opponent Corners Ending in Shot %": _percentage(
            int(_number(faced_corners.get("Shots", pd.Series(dtype=float))).fillna(0).gt(0).sum()),
            len(faced_corners),
        ),
        "Corner Second-Phase Shots Conceded /90": _per_match(
            _sum(faced_corners, "Second-Phase Shots"), matches
        ),
        "Corner Second-Phase xG Conceded /90": _per_match(
            _sum(faced_corners, "Second-Phase xG"), matches
        ),
        "Box Free Kicks /90": _per_match(int(_box_free_kick_mask(own_free_kicks).sum()), matches),
        "Free Kick Goals /90": _per_match(_sum(own_free_kicks, "Goals"), matches),
        "Free Kick xG /90": _per_match(_sum(own_free_kicks, "xG"), matches),
        "Free Kick Shots /90": _per_match(_sum(own_free_kicks, "Shots"), matches),
        "Free Kick First Contact Won %": _percentage(
            _bool_count(free_kick_contacts, True), len(free_kick_contacts)
        ),
        "Direct Free Kicks /90": _per_match(len(direct_free_kicks), matches),
        "Direct Free Kick xG /90": _per_match(_sum(direct_free_kicks, "xG"), matches),
        "Free Kicks Faced /90": _per_match(len(faced_free_kicks), matches),
        "Free Kick Goals Conceded /90": _per_match(_sum(faced_free_kicks, "Goals"), matches),
        "Free Kick xG Conceded /90": _per_match(_sum(faced_free_kicks, "xG"), matches),
        "Free Kick Shots Conceded /90": _per_match(_sum(faced_free_kicks, "Shots"), matches),
        "Throw-Ins /90": _per_match(len(own_throws), matches),
        "Long-Throw Share %": _percentage(int(own_throws["Long Throw"].fillna(False).sum()), len(own_throws)),
        "Throw-In Retention %": _percentage(int(own_throws["Retained"].fillna(False).sum()), len(own_throws)),
        "Throw-Ins Ending in Shot %": _percentage(
            int(_number(own_throws.get("Shots", pd.Series(dtype=float))).fillna(0).gt(0).sum()), len(own_throws)
        ),
        "Throw-In Goals /90": _per_match(_sum(own_throws, "Goals"), matches),
        "Throw-In xG /90": _per_match(_sum(own_throws, "xG"), matches),
        "Throw-In Shots /90": _per_match(_sum(own_throws, "Shots"), matches),
        "Throw-Ins Faced /90": _per_match(len(faced_throws), matches),
        "Long Throws Faced /90": _per_match(int(faced_throws["Long Throw"].fillna(False).sum()), matches),
        "Opponent Throw-In Retention %": _percentage(
            int(faced_throws["Retained"].fillna(False).sum()), len(faced_throws)
        ),
        "Throw-In Goals Conceded /90": _per_match(_sum(faced_throws, "Goals"), matches),
        "Throw-In xG Conceded /90": _per_match(_sum(faced_throws, "xG"), matches),
        "Throw-In Shots Conceded /90": _per_match(_sum(faced_throws, "Shots"), matches),
    }


def aggregate_team_profiles(
    sequences: pd.DataFrame,
    matches: pd.DataFrame,
    teams: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate restart outcomes after summing match-level event counts."""
    if matches.empty:
        return pd.DataFrame()
    team_names = teams or _fixture_team_names(matches)
    rows = []
    for team_name in team_names:
        fixtures = team_fixture_rows(matches, team_name)
        if not fixtures.empty:
            rows.append(_profile_row(sequences, fixtures, team_name))
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def comparison_pool(
    league_profiles: pd.DataFrame,
    selected_profile: pd.DataFrame,
    selected_team: str,
) -> pd.DataFrame:
    if league_profiles.empty or selected_profile.empty:
        return pd.DataFrame()
    selected = selected_profile[selected_profile["Team"].astype(str).eq(str(selected_team))]
    if selected.empty:
        return pd.DataFrame()
    others = league_profiles[~league_profiles["Team"].astype(str).eq(str(selected_team))]
    return pd.concat([others, selected], ignore_index=True, sort=False)


def benchmark_metrics(
    league_profiles: pd.DataFrame,
    selected_profile: pd.DataFrame,
    selected_team: str,
    metric_specs: list[tuple[str, bool]],
) -> pd.DataFrame:
    pool = comparison_pool(league_profiles, selected_profile, selected_team)
    selected = selected_profile[selected_profile["Team"].astype(str).eq(str(selected_team))]
    if pool.empty or selected.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for metric, higher_is_better in metric_specs:
        if metric not in pool or metric not in selected:
            continue
        values = _number(pool[metric])
        valid = pool.loc[values.notna(), ["Team"]].copy()
        valid[metric] = values.loc[values.notna()].to_numpy()
        if valid.empty:
            continue
        valid["Rank"] = valid[metric].rank(ascending=not higher_is_better, method="min")
        valid["Percentile"] = valid[metric].rank(ascending=higher_is_better, pct=True).mul(100)
        selected_row = valid[valid["Team"].astype(str).eq(str(selected_team))]
        selected_value = _number(selected[metric]).iloc[0]
        rows.append(
            {
                "Metric": metric,
                "Value": float(selected_value) if pd.notna(selected_value) else np.nan,
                "League Average": float(_number(league_profiles[metric]).mean()),
                "Rank": int(selected_row["Rank"].iloc[0]) if not selected_row.empty else 0,
                "Teams": int(len(valid)),
                "Percentile": float(selected_row["Percentile"].iloc[0]) if not selected_row.empty else np.nan,
                "Better Direction": "Higher" if higher_is_better else "Lower",
            }
        )
    return pd.DataFrame(rows)


def profile_value(profile: pd.DataFrame, metric: str) -> float:
    if profile.empty or metric not in profile:
        return np.nan
    value = _number(profile[metric]).iloc[0]
    return float(value) if pd.notna(value) else np.nan


def value_text(value: object, digits: int = 2, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{number:,.{digits}f}{suffix}"


def _place_pitch_legend_below(fig: go.Figure) -> go.Figure:
    """Keep set-piece pitch titles and keys from competing for the same space."""
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.06,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=11),
            title=dict(text="Key"),
        ),
        margin=dict(l=28, r=28, t=86, b=92),
    )
    return fig


def percentile_profile_chart(benchmarks: pd.DataFrame, title: str) -> go.Figure:
    if benchmarks.empty:
        return charting.polish_figure(go.Figure(), title, height=420)
    rows = benchmarks.copy()
    rows["Percentile"] = _number(rows["Percentile"]).fillna(0)
    rows["Colour"] = np.select(
        [rows["Percentile"].ge(67), rows["Percentile"].ge(33)],
        [GREEN, AMBER],
        default=RED,
    )
    rows["Label"] = rows["Metric"].map(lambda value: charting.wrap_label(value, width=26, max_lines=2))
    customdata = np.stack(
        [
            rows["Value"].map(lambda value: value_text(value, 2)),
            rows["League Average"].map(lambda value: value_text(value, 2)),
            rows["Rank"].astype(str),
            rows["Teams"].astype(str),
            rows["Better Direction"].astype(str),
        ],
        axis=-1,
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[100] * len(rows),
            y=rows["Label"],
            orientation="h",
            marker_color="#eef2f6",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Bar(
            x=rows["Percentile"],
            y=rows["Label"],
            orientation="h",
            marker=dict(color=rows["Colour"], line=dict(color="#ffffff", width=1)),
            text=[f"{value:.0f}" for value in rows["Percentile"]],
            textposition="inside",
            textfont=dict(color="#ffffff", size=13),
            customdata=customdata,
            hovertemplate=(
                "<b>%{y}</b><br>League percentile: %{x:.0f}<br>"
                "Selected: %{customdata[0]}<br>League average: %{customdata[1]}<br>"
                "Rank: %{customdata[2]} of %{customdata[3]}<br>"
                "Better direction: %{customdata[4]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_vline(x=50, line=dict(color=GREY, width=1.5, dash="dash"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(range=[0, 100], dtick=25, title="League Percentile Score")
    fig.update_yaxes(autorange="reversed", title="")
    return charting.polish_figure(fig, title, height=max(420, len(rows) * 54 + 150))


def match_xg_trend(
    sequences: pd.DataFrame,
    fixtures: pd.DataFrame,
    team_name: str,
    title: str,
) -> go.Figure:
    if fixtures.empty:
        return charting.polish_figure(go.Figure(), title, height=430)
    rows = fixtures.copy()
    rows["MatchId Text"] = rows["MatchId"].astype(str)
    own = sequences[
        sequences["Team"].astype(str).eq(str(team_name))
        & sequences["Set Piece Type"].astype(str).isin(ANALYSED_TYPES)
    ].copy()
    against = sequences[
        sequences["Opponent"].astype(str).eq(str(team_name))
        & sequences["Set Piece Type"].astype(str).isin(ANALYSED_TYPES)
    ].copy()
    own["MatchId Text"] = own["MatchId"].astype(str)
    against["MatchId Text"] = against["MatchId"].astype(str)
    own_xg = own.groupby("MatchId Text")["xG"].sum()
    against_xg = against.groupby("MatchId Text")["xG"].sum()
    own_goals = own.groupby("MatchId Text")["Goals"].sum()
    against_goals = against.groupby("MatchId Text")["Goals"].sum()
    rows["Set-Play xG"] = rows["MatchId Text"].map(own_xg).fillna(0)
    rows["Set-Play xG Against"] = rows["MatchId Text"].map(against_xg).fillna(0)
    rows["Set-Play Goals"] = rows["MatchId Text"].map(own_goals).fillna(0)
    rows["Set-Play Goals Against"] = rows["MatchId Text"].map(against_goals).fillna(0)
    rows["Label"] = rows.apply(
        lambda row: f"{pd.to_datetime(row['Date']).strftime('%d %b')}<br>{row['Opponent']}", axis=1
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=rows["Label"],
            y=rows["Set-Play xG"],
            name="Set-Play xG For",
            marker_color=RED,
            customdata=np.stack([rows["Set-Play Goals"]], axis=-1),
            hovertemplate="<b>%{x}</b><br>xG for: %{y:.2f}<br>Goals: %{customdata[0]:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=rows["Label"],
            y=-rows["Set-Play xG Against"],
            name="Set-Play xG Against",
            marker_color="#344054",
            customdata=np.stack([rows["Set-Play xG Against"], rows["Set-Play Goals Against"]], axis=-1),
            hovertemplate=(
                "<b>%{x}</b><br>xG against: %{customdata[0]:.2f}<br>"
                "Goals against: %{customdata[1]:.0f}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line=dict(color="#98a2b3", width=1))
    fig.update_layout(barmode="relative")
    fig.update_xaxes(title="Fixture")
    fig.update_yaxes(title="Set-Play xG (Against Below Zero)")
    return charting.polish_figure(fig, title, height=470)


def _sequence_key(frame: pd.DataFrame) -> pd.Series:
    match_id = frame.get("MatchId", pd.Series("", index=frame.index)).astype(str)
    set_piece_id = frame.get("Set Piece ID", pd.Series("", index=frame.index)).astype(str)
    return match_id + "::" + set_piece_id.str.replace(r"\.0$", "", regex=True)


def events_with_sequence_context(events: pd.DataFrame, sequences: pd.DataFrame) -> pd.DataFrame:
    if events.empty or sequences.empty:
        return pd.DataFrame()
    detail = events.copy()
    detail["_SetPieceKey"] = _sequence_key(detail)
    lookup_columns = [
        "MatchId",
        "Set Piece ID",
        "Team",
        "Opponent",
        "Set Piece Type",
        "Side",
        "Taker",
        "First Touch Player",
        "First Touch Won",
    ]
    lookup = sequences[[column for column in lookup_columns if column in sequences]].copy()
    lookup["_SetPieceKey"] = _sequence_key(lookup)
    lookup = lookup.drop_duplicates("_SetPieceKey").rename(
        columns={
            "Team": "Set Piece Team",
            "Opponent": "Set Piece Opponent",
            "Taker": "Set Piece Taker",
        }
    )
    drop_join_columns = [column for column in ["MatchId", "Set Piece ID"] if column in lookup]
    lookup = lookup.drop(columns=drop_join_columns)
    return detail.merge(lookup, on="_SetPieceKey", how="inner")


def set_piece_shots(
    events: pd.DataFrame,
    sequences: pd.DataFrame,
    team_name: str,
    against: bool = False,
    set_piece_types: set[str] | None = None,
) -> pd.DataFrame:
    merged = events_with_sequence_context(events, sequences)
    if merged.empty:
        return merged
    owners = ~merged["Set Piece Team"].astype(str).eq(str(team_name)) if against else merged[
        "Set Piece Team"
    ].astype(str).eq(str(team_name))
    rows = merged[
        owners
        & merged["Team"].astype(str).eq(merged["Set Piece Team"].astype(str))
        & merged["Action Type"].astype(str).str.upper().eq("SHOT")
    ].copy()
    if set_piece_types:
        rows = rows[rows["Set Piece Type"].astype(str).isin(set_piece_types)].copy()
    return rows


def shot_map(
    events: pd.DataFrame,
    sequences: pd.DataFrame,
    team_name: str,
    title: str,
    against: bool = False,
    set_piece_types: set[str] | None = None,
) -> go.Figure:
    shots = set_piece_shots(events, sequences, team_name, against, set_piece_types)
    fig = pitch.half_pitch_vertical_figure(title, height=610, legend=True)
    _place_pitch_legend_below(fig)
    if shots.empty:
        fig.add_annotation(
            x=0,
            y=25,
            text="No set-piece shots in the selected match window",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor=LIGHT_GREY,
            borderwidth=1,
            borderpad=7,
        )
        return fig

    start_x = _number(shots["Start X"])
    start_y = _number(shots["Start Y"])
    end_x = _number(shots.get("End X", pd.Series(np.nan, index=shots.index)))
    direction = pd.Series(np.where(end_x.fillna(start_x).lt(0), -1.0, 1.0), index=shots.index)
    shots["_Plot X"] = (start_y * direction).clip(-34, 34)
    shots["_Plot Y"] = (start_x * direction).clip(0, 52.5)
    shots["_xG"] = _number(shots["Shot xG"]).fillna(0)
    shots["_Goal"] = (
        shots["Action"].astype(str).str.upper().eq("GOAL")
        | shots["Result"].astype(str).str.upper().isin(["SUCCESS", "GOAL"])
    )
    shots["_Outcome"] = np.where(shots["_Goal"], "Goal", "Shot")
    shots["_Size"] = 11 + np.sqrt(shots["_xG"].clip(lower=0)) * 28
    shots["_Minute"] = _number(shots.get("Minute", pd.Series(np.nan, index=shots.index)))

    for outcome, colour, symbol in [("Shot", RED, "circle"), ("Goal", GREEN, "star")]:
        subset = shots[shots["_Outcome"].eq(outcome)]
        if subset.empty:
            continue
        customdata = np.stack(
            [
                subset["Player"].fillna("Unknown").astype(str),
                subset["Set Piece Type"].fillna("Set Piece").astype(str),
                subset["_xG"],
                subset["_Minute"],
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=subset["_Plot X"],
                y=subset["_Plot Y"],
                mode="markers",
                name=outcome,
                marker=dict(
                    size=subset["_Size"],
                    color=colour,
                    symbol=symbol,
                    opacity=0.82,
                    line=dict(color="#ffffff", width=1.5),
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                    "Minute: %{customdata[3]:.0f}<br>xG: %{customdata[2]:.3f}<extra></extra>"
                ),
            )
        )
    fig.add_annotation(
        x=30,
        y=8,
        text="ATTACK ↑",
        showarrow=False,
        font=dict(size=11, color=GREY),
    )
    return fig


def _delivery_bucket(row: pd.Series) -> str:
    type_text = " ".join(
        str(row.get(column, ""))
        for column in ["Corner Type", "Corner End Zone", "Free Kick Type", "Free Kick End Zone", "Execution Type"]
    ).upper()
    if "NEAR" in type_text or "FLICK" in type_text:
        return "Near Post"
    if "FAR" in type_text or "BACK_OF_BOX" in type_text:
        return "Far Post"
    if "OPEN_PLAY" in type_text or "SHORT" in type_text or "POSSESSION" in type_text:
        return "Short / Recycled"
    if "CENTRAL" in type_text or "BOX" in type_text or "CROSS" in type_text or "HIGH_BALL" in type_text:
        return "Central"
    if str(row.get("Set Piece Type", "")) == "Direct Free Kick":
        return "Direct Shot"
    return "Other"


DELIVERY_COLOURS = {
    "Near Post": RED,
    "Central": BLUE,
    "Far Post": PURPLE,
    "Short / Recycled": TEAL,
    "Direct Shot": AMBER,
    "Other": GREY,
}


def delivery_map(
    sequences: pd.DataFrame,
    title: str,
    set_piece_types: set[str],
    team_name: str | None = None,
    against: bool = False,
    side: str | None = None,
) -> go.Figure:
    rows = sequences[sequences["Set Piece Type"].astype(str).isin(set_piece_types)].copy()
    if team_name:
        team_mask = rows["Opponent"].astype(str).eq(str(team_name)) if against else rows["Team"].astype(str).eq(str(team_name))
        rows = rows[team_mask].copy()
    if side:
        rows = rows[rows["Side"].astype(str).eq(str(side))].copy()
    rows = rows.dropna(subset=["Start X", "Start Y", "End X", "End Y"])
    fig = pitch.pitch_figure(title, height=540, legend=True)
    _place_pitch_legend_below(fig)
    if rows.empty:
        fig.add_annotation(
            x=18,
            y=0,
            text="No mapped deliveries in this selection",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor=LIGHT_GREY,
            borderwidth=1,
            borderpad=7,
        )
        fig.update_xaxes(range=[8, 55])
        return fig

    x_values = pd.concat([_number(rows["Start X"]), _number(rows["End X"])], ignore_index=True).dropna()
    y_values = pd.concat([_number(rows["Start Y"]), _number(rows["End Y"])], ignore_index=True).dropna()
    x_min = max(pitch.PITCH_X_MIN - 2.0, min(8.0, float(x_values.min()) - 3.0)) if not x_values.empty else 8.0
    x_max = min(pitch.PITCH_X_MAX + 2.0, max(55.0, float(x_values.max()) + 3.0)) if not x_values.empty else 55.0
    y_min = max(pitch.PITCH_Y_MIN - 2.0, min(-36.0, float(y_values.min()) - 3.0)) if not y_values.empty else -36.0
    y_max = min(pitch.PITCH_Y_MAX + 2.0, max(36.0, float(y_values.max()) + 3.0)) if not y_values.empty else 36.0

    rows["Delivery"] = rows.apply(_delivery_bucket, axis=1)
    first_won = rows["First Touch Won"].astype("boolean")
    rows["Contact Outcome"] = np.select(
        [first_won.eq(True).fillna(False), first_won.eq(False).fillna(False)],
        ["Attacking First Contact", "Defensive First Contact"],
        default="Uncontested / Unknown",
    )
    outcome_colours = {
        "Attacking First Contact": GREEN,
        "Defensive First Contact": RED,
        "Uncontested / Unknown": GREY,
    }
    for _, row in rows.iterrows():
        colour = DELIVERY_COLOURS.get(str(row["Delivery"]), GREY)
        hover = (
            f"<b>{row.get('Taker') if pd.notna(row.get('Taker')) else 'Unknown taker'}</b><br>"
            f"{row.get('Set Piece Type')} · {row.get('Delivery')}<br>"
            f"First contact: {row.get('Contact Outcome')}<br>"
            f"Shots: {int(row.get('Shots', 0))} · xG: {float(row.get('xG', 0)):.3f}"
        )
        fig.add_trace(
            go.Scatter(
                x=[row["Start X"], row["End X"]],
                y=[row["Start Y"], row["End Y"]],
                mode="lines+markers",
                line=dict(color=colour, width=2.1),
                marker=dict(
                    size=[5, 8],
                    color=[colour, outcome_colours.get(str(row["Contact Outcome"]), GREY)],
                    line=dict(color="#ffffff", width=1.1),
                ),
                opacity=0.72,
                hovertemplate=hover + "<extra></extra>",
                showlegend=False,
            )
        )
    for outcome, colour in outcome_colours.items():
        subset = rows[rows["Contact Outcome"].eq(outcome)]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["End X"],
                y=subset["End Y"],
                mode="markers",
                name=outcome,
                marker=dict(size=9, color=colour, line=dict(color="#ffffff", width=1.2)),
                hoverinfo="skip",
            )
        )
    for delivery in sorted(rows["Delivery"].unique()):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=str(delivery),
                line=dict(color=DELIVERY_COLOURS.get(str(delivery), GREY), width=3),
                hoverinfo="skip",
            )
        )
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])
    fig.add_annotation(x=x_min + 4, y=y_max - 5, text="ATTACK →", showarrow=False, font=dict(size=11, color=GREY))
    return fig


def _delivery_mix_tables(sequences: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = sequences[sequences["Set Piece Type"].eq("Corner")].copy()
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows["Delivery"] = rows.apply(_delivery_bucket, axis=1)
    side_values = rows["Side"].fillna("Unknown").astype(str).str.strip()
    rows["Side"] = side_values.mask(side_values.eq("") | side_values.str.lower().isin({"nan", "none"}), "Unknown")

    present_sides = rows["Side"].unique().tolist()
    preferred_sides = ["Left", "Right", "Centre", "Unknown"]
    side_order = [side for side in preferred_sides if side in present_sides]
    side_order.extend(sorted(side for side in present_sides if side not in side_order))

    present_deliveries = rows["Delivery"].unique().tolist()
    preferred_deliveries = ["Near Post", "Central", "Far Post", "Short / Recycled", "Other"]
    delivery_order = [delivery for delivery in preferred_deliveries if delivery in present_deliveries]
    delivery_order.extend(sorted(delivery for delivery in present_deliveries if delivery not in delivery_order))
    matrix = pd.crosstab(rows["Side"], rows["Delivery"]).reindex(
        index=side_order, columns=delivery_order, fill_value=0
    )
    percentages = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0).mul(100).fillna(0)
    return matrix, percentages


def delivery_mix_takeaway(sequences: pd.DataFrame) -> str:
    """Describe the leading delivery choice in stakeholder-friendly language."""
    matrix, percentages = _delivery_mix_tables(sequences)
    if matrix.empty:
        return "No attacking corners are available for this selection."

    leaders: dict[str, tuple[str, float, int, int]] = {}
    for side in matrix.index:
        delivery = str(percentages.loc[side].idxmax())
        leaders[str(side)] = (
            delivery,
            float(percentages.loc[side, delivery]),
            int(matrix.loc[side, delivery]),
            int(matrix.loc[side].sum()),
        )

    if "Left" in leaders and "Right" in leaders:
        left_delivery, left_share, left_count, left_total = leaders["Left"]
        right_delivery, right_share, right_count, right_total = leaders["Right"]
        if left_delivery == right_delivery:
            gap = abs(right_share - left_share)
            return (
                f"{left_delivery} deliveries lead from both sides: {right_share:.0f}% of right-side corners "
                f"({right_count} of {right_total}) and {left_share:.0f}% of left-side corners "
                f"({left_count} of {left_total}), a difference of {gap:.0f} percentage points."
            )
        return (
            f"Left-side corners most often use {left_delivery} ({left_share:.0f}%, {left_count} of {left_total}); "
            f"right-side corners most often use {right_delivery} ({right_share:.0f}%, {right_count} of {right_total})."
        )

    summaries = [
        f"{side}: {delivery} leads at {share:.0f}% ({count} of {total})"
        for side, (delivery, share, count, total) in leaders.items()
    ]
    return "; ".join(summaries) + "."


def delivery_mix_chart(sequences: pd.DataFrame) -> go.Figure:
    """Compare delivery shares from a common baseline with direct labels."""
    matrix, percentages = _delivery_mix_tables(sequences)
    if matrix.empty:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No attacking corners in this selection",
            showarrow=False,
            font=dict(size=13, color=GREY),
        )
        return charting.polish_figure(fig, height=320)

    side_colours = {
        "Left": RED,
        "Right": "#344054",
        "Centre": BLUE,
        "Unknown": GREY,
    }
    max_share = float(percentages.to_numpy().max()) if percentages.size else 0.0
    x_max = min(110, max(50, math.ceil((max_share + 18) / 10) * 10))
    fig = go.Figure()
    for side in matrix.index:
        counts = matrix.loc[side].astype(int)
        shares = percentages.loc[side]
        total = int(counts.sum())
        customdata = np.column_stack([counts.to_numpy(), shares.to_numpy()])
        fig.add_trace(
            go.Bar(
                x=shares,
                y=matrix.columns,
                orientation="h",
                name=str(side),
                offsetgroup=str(side),
                marker=dict(
                    color=side_colours.get(str(side), GREY),
                    line=dict(color="#ffffff", width=1),
                ),
                text=[
                    f"{side} · {share:.0f}% · {count} corner{'s' if count != 1 else ''}" if count else ""
                    for share, count in zip(shares, counts)
                ],
                textposition="outside",
                textfont=dict(color=DARK, size=12),
                cliponaxis=False,
                customdata=customdata,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    + f"{side}-side corners: %{{customdata[0]:.0f}} of {total}<br>"
                    + "Share from this side: %{customdata[1]:.1f}%<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        barmode="group",
        bargap=0.28,
        bargroupgap=0.08,
        showlegend=False,
    )
    fig.update_xaxes(range=[0, x_max], dtick=10, ticksuffix="%", title="Share of corners from that side")
    fig.update_yaxes(title="", autorange="reversed")
    fig = charting.polish_figure(fig, height=max(420, len(matrix.columns) * 74 + 150))
    fig.update_layout(margin=dict(l=30, r=190, t=30, b=62))
    return fig


def throw_in_map(
    sequences: pd.DataFrame,
    title: str,
    team_name: str | None = None,
    against: bool = False,
) -> go.Figure:
    rows = sequences[sequences["Set Piece Type"].eq("Throw-In")].copy()
    if team_name:
        mask = rows["Opponent"].astype(str).eq(str(team_name)) if against else rows["Team"].astype(str).eq(str(team_name))
        rows = rows[mask].copy()
    rows = rows.dropna(subset=["Start X", "Start Y", "End X", "End Y"])
    fig = pitch.pitch_figure(title, height=600, legend=True)
    if rows.empty:
        fig.add_annotation(
            x=0,
            y=0,
            text="No mapped throw-ins in this selection",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor=LIGHT_GREY,
            borderwidth=1,
            borderpad=7,
        )
        return fig
    rows["Outcome"] = np.where(rows["Retained"].fillna(False), "Retained", "Lost")
    for _, row in rows.iterrows():
        colour = GREEN if row["Outcome"] == "Retained" else RED
        width = 3.3 if bool(row["Long Throw"]) else 1.7
        dash = "solid" if bool(row["Long Throw"]) else "dot"
        hover = (
            f"<b>{row.get('Taker') if pd.notna(row.get('Taker')) else 'Unknown thrower'}</b><br>"
            f"{row.get('Outcome')} · {'Long' if bool(row.get('Long Throw')) else 'Standard'} throw<br>"
            f"Shots: {int(row.get('Shots', 0))} · xG: {float(row.get('xG', 0)):.3f}"
        )
        fig.add_trace(
            go.Scatter(
                x=[row["Start X"], row["End X"]],
                y=[row["Start Y"], row["End Y"]],
                mode="lines",
                line=dict(color=colour, width=width, dash=dash),
                opacity=0.72,
                showlegend=False,
                hovertemplate=hover + "<extra></extra>",
            )
        )
        fig.add_annotation(
            x=float(row["End X"]),
            y=float(row["End Y"]),
            ax=float(row["Start X"]),
            ay=float(row["Start Y"]),
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="",
            showarrow=True,
            arrowhead=2,
            arrowsize=0.65,
            arrowwidth=1.2,
            arrowcolor=colour,
            opacity=0.72,
        )
    legend_items = [
        ("Retained", GREEN, "solid", 3),
        ("Lost", RED, "solid", 3),
        ("Long Throw (20m+ Gain)", DARK, "solid", 4),
        ("Standard Throw", GREY, "dot", 2),
    ]
    for label, colour, dash, width in legend_items:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=label,
                line=dict(color=colour, dash=dash, width=width),
                hoverinfo="skip",
            )
        )
    fig.add_annotation(x=-46, y=31, text="ATTACK →", showarrow=False, font=dict(size=11, color=GREY))
    return fig


def taker_table(sequences: pd.DataFrame, set_piece_types: set[str]) -> pd.DataFrame:
    rows = sequences[
        sequences["Set Piece Type"].astype(str).isin(set_piece_types) & sequences["Taker"].notna()
    ].copy()
    if rows.empty:
        return pd.DataFrame(columns=["Player", "Restarts", "Shots Created", "Goals", "xG Created"])
    result = rows.groupby("Taker", as_index=False).agg(
        Restarts=("Set Piece ID", "count"),
        **{
            "Shots Created": ("Shots", "sum"),
            "Goals": ("Goals", "sum"),
            "xG Created": ("xG", "sum"),
        },
    )
    result = result.rename(columns={"Taker": "Player"})
    result["xG Created"] = _number(result["xG Created"]).round(3)
    return result.sort_values(["Restarts", "xG Created"], ascending=[False, False]).reset_index(drop=True)


def first_contact_table(sequences: pd.DataFrame, set_piece_types: set[str]) -> pd.DataFrame:
    rows = sequences[
        sequences["Set Piece Type"].astype(str).isin(set_piece_types)
        & sequences["First Touch Player"].notna()
        & sequences["First Touch Won"].astype("boolean").eq(True).fillna(False)
    ].copy()
    if rows.empty:
        return pd.DataFrame(columns=["Player", "First Contacts Won", "Shot Sequences", "Goals", "Sequence xG"])
    result = rows.groupby("First Touch Player", as_index=False).agg(
        **{
            "First Contacts Won": ("Set Piece ID", "count"),
            "Shot Sequences": ("Shots", lambda values: int(_number(values).fillna(0).gt(0).sum())),
            "Goals": ("Goals", "sum"),
            "Sequence xG": ("xG", "sum"),
        }
    )
    result = result.rename(columns={"First Touch Player": "Player"})
    result["Sequence xG"] = _number(result["Sequence xG"]).round(3)
    return result.sort_values(["First Contacts Won", "Sequence xG"], ascending=[False, False]).reset_index(drop=True)


def goal_log(
    sequences: pd.DataFrame,
    fixtures: pd.DataFrame,
    team_name: str,
    against: bool = False,
) -> pd.DataFrame:
    mask = sequences["Opponent"].astype(str).eq(str(team_name)) if against else sequences["Team"].astype(str).eq(str(team_name))
    rows = sequences[
        mask
        & sequences["Set Piece Type"].astype(str).isin(ANALYSED_TYPES)
        & _number(sequences["Goals"]).fillna(0).gt(0)
    ].copy()
    if rows.empty:
        return pd.DataFrame(columns=["Date", "Opponent", "Type", "Taker", "First Contact", "Goals", "xG"])
    context = fixtures[[column for column in ["MatchId", "Date", "Opponent"] if column in fixtures]].copy()
    context["MatchId Text"] = context["MatchId"].astype(str)
    rows["MatchId Text"] = rows["MatchId"].astype(str)
    rows = rows.merge(context.drop(columns=["MatchId"]), on="MatchId Text", how="left", suffixes=("", " Fixture"))
    rows["Date"] = pd.to_datetime(rows.get("Date Fixture", rows.get("Date")), errors="coerce").dt.strftime("%d %b %Y")
    if "Opponent Fixture" in rows:
        rows["Opponent"] = rows["Opponent Fixture"].fillna(rows["Opponent"])
    return rows.rename(
        columns={
            "Set Piece Type": "Type",
            "First Touch Player": "First Contact",
        }
    )[["Date", "Opponent", "Type", "Taker", "First Contact", "Goals", "xG"]].sort_values("Date", ascending=False)


def outcome_funnel_chart(sequences: pd.DataFrame, title: str, set_piece_types: set[str]) -> go.Figure:
    rows = sequences[sequences["Set Piece Type"].astype(str).isin(set_piece_types)].copy()
    if rows.empty:
        return charting.polish_figure(go.Figure(), title, height=390)
    total = len(rows)
    contact_rows = rows["First Touch Won"].dropna()
    values = pd.DataFrame(
        {
            "Stage": ["Restarts", "First Contacts Won", "Shot-Producing", "Goals"],
            "Count": [
                total,
                _bool_count(contact_rows, True),
                int(_number(rows["Shots"]).fillna(0).gt(0).sum()),
                int(_number(rows["Goals"]).fillna(0).sum()),
            ],
        }
    )
    values["Share"] = values["Count"].div(total).mul(100) if total else 0
    fig = go.Figure(
        go.Funnel(
            y=values["Stage"],
            x=values["Count"],
            text=[f"{count} · {share:.0f}%" for count, share in zip(values["Count"], values["Share"], strict=False)],
            textposition="inside",
            marker=dict(color=["#344054", BLUE, AMBER, GREEN]),
            customdata=np.stack([values["Share"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>Count: %{x}<br>Share of restarts: %{customdata[0]:.1f}%<extra></extra>",
        )
    )
    return charting.polish_figure(fig, title, height=390)


def match_vs_season_chart(
    match_sequences: pd.DataFrame,
    season_sequences: pd.DataFrame,
    season_matches: pd.DataFrame,
    teams: list[str],
    set_piece_types: set[str],
    title: str,
) -> go.Figure:
    """Index match output to each team's own full-season per-match baseline."""
    metric_sources = [
        ("Restarts", None),
        ("Shots", "Shots"),
        ("Goals", "Goals"),
        ("xG", "xG"),
    ]
    chart_rows: list[dict[str, object]] = []
    for team_name in teams:
        match_rows = match_sequences[
            match_sequences["Team"].astype(str).eq(str(team_name))
            & match_sequences["Set Piece Type"].astype(str).isin(set_piece_types)
        ]
        team_fixtures = team_fixture_rows(season_matches, team_name)
        season_match_count = max(int(team_fixtures["MatchId"].astype(str).nunique()), 1)
        season_rows = season_sequences[
            season_sequences["Team"].astype(str).eq(str(team_name))
            & season_sequences["Set Piece Type"].astype(str).isin(set_piece_types)
        ]
        for label, source in metric_sources:
            match_value = float(len(match_rows)) if source is None else _sum(match_rows, source)
            season_total = float(len(season_rows)) if source is None else _sum(season_rows, source)
            season_average = season_total / season_match_count
            index = match_value / season_average * 100 if season_average else np.nan
            chart_rows.append(
                {
                    "Team": team_name,
                    "Metric": label,
                    "Match Value": match_value,
                    "Season Average": season_average,
                    "Index": index,
                }
            )
    rows = pd.DataFrame(chart_rows)
    if rows.empty:
        return charting.polish_figure(go.Figure(), title, height=430)
    colours = [RED, "#344054", BLUE, TEAL]
    fig = go.Figure()
    for team_name, colour in zip(teams, colours, strict=False):
        subset = rows[rows["Team"].astype(str).eq(str(team_name))]
        customdata = np.stack([subset["Match Value"], subset["Season Average"]], axis=-1)
        fig.add_trace(
            go.Bar(
                x=subset["Metric"],
                y=subset["Index"],
                name=team_name,
                marker_color=colour,
                customdata=customdata,
                hovertemplate=(
                    "<b>%{fullData.name} · %{x}</b><br>Match: %{customdata[0]:.3f}<br>"
                    "Season average: %{customdata[1]:.3f}<br>Index: %{y:.0f}<extra></extra>"
                ),
            )
        )
    fig.add_hline(y=100, line=dict(color=GREY, width=1.6, dash="dash"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Outcome")
    fig.update_yaxes(title="Match vs Own Season Average (100 = Average)", rangemode="tozero")
    return charting.polish_figure(fig, title, height=470)


def match_first_contact_table(match_sequences: pd.DataFrame, teams: list[str], set_piece_types: set[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for team_name in teams:
        team_rows = match_sequences[
            match_sequences["Team"].astype(str).eq(str(team_name))
            & match_sequences["Set Piece Type"].astype(str).isin(set_piece_types)
        ]
        contacts = team_rows["First Touch Won"].dropna()
        rows.append(
            {
                "Team": team_name,
                "Restarts": len(team_rows),
                "First Contacts Won": _bool_count(contacts, True),
                "First Contacts Lost": _bool_count(contacts, False),
                "Shot-Producing": int(_number(team_rows.get("Shots", pd.Series(dtype=float))).fillna(0).gt(0).sum()),
                "Shots": int(_sum(team_rows, "Shots")),
                "Goals": int(_sum(team_rows, "Goals")),
                "xG": round(_sum(team_rows, "xG"), 3),
            }
        )
    return pd.DataFrame(rows)


def match_sequence_table(sequences: pd.DataFrame) -> pd.DataFrame:
    if sequences.empty:
        return pd.DataFrame()
    rows = sequences.copy()
    rows["Minute"] = np.floor(_number(rows["Game Second"]).fillna(0) % 10000 / 60).astype(int) + 1
    display = [
        "Minute",
        "Team",
        "Set Piece Type",
        "Side",
        "Taker",
        "First Touch Player",
        "First Touch Won",
        "Shots",
        "Goals",
        "xG",
        "Second-Phase Shots",
        "Second-Phase xG",
    ]
    rows["xG"] = _number(rows["xG"]).round(3)
    rows["Second-Phase xG"] = _number(rows["Second-Phase xG"]).round(3)
    return rows[display].sort_values(["Minute", "Team"]).reset_index(drop=True)
