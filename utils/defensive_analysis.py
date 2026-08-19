"""Calculations and visuals for the Team Analysis defensive-actions page.

The module keeps two data grains separate:

* Impect squad/player match-level KPI facts supply trusted totals and success rates.
* Event rows supply locations and possession-sequence context only.

That separation prevents event labels from being mistaken for provider totals
and prevents season aggregates from being averaged a second time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils import charting, pitch, ui


RED = ui.CHARLTON_RED
DARK = ui.CHARLTON_BLACK
GREEN = "#16803c"
AMBER = "#d89216"
GREY = "#7a7f87"
LIGHT_GREY = "#e6edf5"
BLUE = "#344054"

REGAIN_ACTION_TYPES = {"LOOSE_BALL_REGAIN", "INTERCEPTION", "GK_CATCH"}

TEAM_SUM_COLUMNS = [
    "Ball Wins",
    "Ball Losses",
    "Opponents Removed",
    "Defenders Removed",
    "Ball Win Value",
    "Defensive Touches",
    "Presses",
    "Counterpresses",
    "Build-Up Presses",
    "Between-Lines Presses",
    "Second Balls",
    "Second Balls Won",
    "Ground Duels Won",
    "Ground Duels Lost",
    "Aerial Duels Won",
    "Aerial Duels Lost",
    "Suffered Bypassed Opponents",
    "Suffered Bypassed Defenders",
    "Goals Conceded",
    "xG Conceded",
    "First-Third Ball Wins",
    "Middle-Third Ball Wins",
    "Final-Third Ball Wins",
    "Opponent-Box Ball Wins",
    "Wide-Left Ball Wins",
    "Half-Left Ball Wins",
    "Centre Ball Wins",
    "Half-Right Ball Wins",
    "Wide-Right Ball Wins",
    "Out-of-Possession Ball Wins",
    "Defensive-Transition Ball Wins",
    "Set-Piece Ball Wins",
    "Second-Ball Phase Wins",
]

PLAYER_SUM_COLUMNS = [
    "Play Duration Seconds",
    "Match Share",
    "Ball Wins",
    "Ball Losses",
    "Opponents Removed",
    "Defenders Removed",
    "Ball Win Value",
    "Defensive Touches",
    "Presses",
    "Counterpresses",
    "Second Balls",
    "Second Balls Won",
    "Ground Duels Won",
    "Ground Duels Lost",
    "Aerial Duels Won",
    "Aerial Duels Lost",
]


def _number(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, scale: float = 1.0) -> pd.Series:
    numerator_values = _number(numerator)
    denominator_values = _number(denominator)
    return (numerator_values.div(denominator_values.where(denominator_values.ne(0))) * scale).replace(
        [np.inf, -np.inf], np.nan
    )


def aggregate_teams(rows: pd.DataFrame) -> pd.DataFrame:
    """Roll selected match-level KPI facts into one row per team.

    Counts are summed first. Rates and percentages are then calculated from
    those summed numerators and denominators, which is the correct weighted
    treatment for match totals.
    """
    if rows.empty or "Team" not in rows:
        return pd.DataFrame()

    clean = rows.copy()
    available = [column for column in TEAM_SUM_COLUMNS if column in clean]
    for column in available:
        clean[column] = _number(clean[column]).fillna(0.0)

    totals = clean.groupby("Team", dropna=False, as_index=False)[available].sum()
    match_counts = clean.groupby("Team", dropna=False)["MatchId"].nunique().rename("Matches").reset_index()
    totals = totals.merge(match_counts, on="Team", how="left")
    matches = _number(totals["Matches"]).replace(0, np.nan)

    per_match_sources = {
        "Ball Wins / Match": "Ball Wins",
        "Ball Losses / Match": "Ball Losses",
        "Ball Win Value / Match": "Ball Win Value",
        "Defensive Touches / Match": "Defensive Touches",
        "Presses / Match": "Presses",
        "Counterpresses / Match": "Counterpresses",
        "Build-Up Presses / Match": "Build-Up Presses",
        "Between-Lines Presses / Match": "Between-Lines Presses",
        "Second Balls / Match": "Second Balls",
        "Suffered Bypassed Opponents / Match": "Suffered Bypassed Opponents",
        "Suffered Bypassed Defenders / Match": "Suffered Bypassed Defenders",
        "Goals Conceded / Match": "Goals Conceded",
        "xG Conceded / Match": "xG Conceded",
        "First-Third Ball Wins / Match": "First-Third Ball Wins",
        "Middle-Third Ball Wins / Match": "Middle-Third Ball Wins",
        "Final-Third Ball Wins / Match": "Final-Third Ball Wins",
        "Opponent-Box Ball Wins / Match": "Opponent-Box Ball Wins",
    }
    for output, source in per_match_sources.items():
        totals[output] = _number(totals[source]).div(matches)

    totals["Opponents Removed / Ball Win"] = _safe_ratio(totals["Opponents Removed"], totals["Ball Wins"])
    totals["Defenders Removed / Ball Win"] = _safe_ratio(totals["Defenders Removed"], totals["Ball Wins"])
    totals["Counterpress Share %"] = _safe_ratio(totals["Counterpresses"], totals["Presses"], 100)
    totals["Second-Ball Win %"] = _safe_ratio(totals["Second Balls Won"], totals["Second Balls"], 100)
    totals["Ground Duel Win %"] = _safe_ratio(
        totals["Ground Duels Won"], totals["Ground Duels Won"] + totals["Ground Duels Lost"], 100
    )
    totals["Aerial Duel Win %"] = _safe_ratio(
        totals["Aerial Duels Won"], totals["Aerial Duels Won"] + totals["Aerial Duels Lost"], 100
    )
    totals["Duel Win %"] = _safe_ratio(
        totals["Ground Duels Won"] + totals["Aerial Duels Won"],
        totals["Ground Duels Won"]
        + totals["Ground Duels Lost"]
        + totals["Aerial Duels Won"]
        + totals["Aerial Duels Lost"],
        100,
    )
    totals["Final-Third Ball Win %"] = _safe_ratio(
        totals["Final-Third Ball Wins"], totals["Ball Wins"], 100
    )
    return totals.replace([np.inf, -np.inf], np.nan)


def aggregate_players(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned player-match rows over the selected match window."""
    if rows.empty or "Player" not in rows:
        return pd.DataFrame()

    clean = rows.copy()
    keys = [column for column in ["Team", "PlayerId", "Player"] if column in clean]
    available = [column for column in PLAYER_SUM_COLUMNS if column in clean]
    for column in available:
        clean[column] = _number(clean[column]).fillna(0.0)

    positions = (
        clean.sort_values("Play Duration Seconds", ascending=False)
        .drop_duplicates(keys)
        [keys + ["Position"]]
    )
    totals = clean.groupby(keys, dropna=False, as_index=False)[available].sum()
    match_counts = clean.groupby(keys, dropna=False)["MatchId"].nunique().rename("Matches").reset_index()
    totals = totals.merge(match_counts, on=keys, how="left").merge(positions, on=keys, how="left")
    totals["Minutes"] = _number(totals["Play Duration Seconds"]).div(60)
    minutes = totals["Minutes"].replace(0, np.nan)
    totals["Ball Wins /90"] = _number(totals["Ball Wins"]).div(minutes).mul(90)
    totals["Ball Losses /90"] = _number(totals["Ball Losses"]).div(minutes).mul(90)
    totals["Presses /90"] = _number(totals["Presses"]).div(minutes).mul(90)
    totals["Ball Win Value /90"] = _number(totals["Ball Win Value"]).div(minutes).mul(90)
    totals["Opponents Removed / Ball Win"] = _safe_ratio(totals["Opponents Removed"], totals["Ball Wins"])
    totals["Second-Ball Win %"] = _safe_ratio(totals["Second Balls Won"], totals["Second Balls"], 100)
    totals["Ground Duel Win %"] = _safe_ratio(
        totals["Ground Duels Won"], totals["Ground Duels Won"] + totals["Ground Duels Lost"], 100
    )
    totals["Aerial Duel Win %"] = _safe_ratio(
        totals["Aerial Duels Won"], totals["Aerial Duels Won"] + totals["Aerial Duels Lost"], 100
    )
    totals["Duel Win %"] = _safe_ratio(
        totals["Ground Duels Won"] + totals["Aerial Duels Won"],
        totals["Ground Duels Won"]
        + totals["Ground Duels Lost"]
        + totals["Aerial Duels Won"]
        + totals["Aerial Duels Lost"],
        100,
    )
    return totals.replace([np.inf, -np.inf], np.nan)


def comparison_pool(
    league_summary: pd.DataFrame,
    selected_summary: pd.DataFrame,
    selected_team: str,
) -> pd.DataFrame:
    """Substitute the selected window into the full-season league benchmark."""
    if league_summary.empty or selected_summary.empty:
        return pd.DataFrame()
    selected = selected_summary[selected_summary["Team"].astype(str).eq(str(selected_team))]
    if selected.empty:
        return pd.DataFrame()
    others = league_summary[~league_summary["Team"].astype(str).eq(str(selected_team))]
    return pd.concat([others, selected], ignore_index=True, sort=False)


def metric_benchmark(
    league_summary: pd.DataFrame,
    selected_summary: pd.DataFrame,
    selected_team: str,
    metric: str,
    higher_is_better: bool = True,
) -> dict[str, float | int]:
    pool = comparison_pool(league_summary, selected_summary, selected_team)
    selected = selected_summary[selected_summary["Team"].astype(str).eq(str(selected_team))]
    if pool.empty or selected.empty or metric not in pool:
        return {"value": np.nan, "average": np.nan, "rank": 0, "teams": 0, "percentile": np.nan}

    values = _number(pool[metric])
    selected_value = float(_number(selected[metric]).iloc[0])
    valid = pool.loc[values.notna(), ["Team"]].copy()
    valid[metric] = values[values.notna()].to_numpy()
    valid["Rank"] = valid[metric].rank(ascending=not higher_is_better, method="min")
    valid["Percentile"] = valid[metric].rank(ascending=higher_is_better, pct=True).mul(100)
    selected_row = valid[valid["Team"].astype(str).eq(str(selected_team))]
    average = float(_number(league_summary[metric]).mean()) if metric in league_summary else np.nan
    return {
        "value": selected_value,
        "average": average,
        "rank": int(selected_row["Rank"].iloc[0]) if not selected_row.empty else 0,
        "teams": int(len(valid)),
        "percentile": float(selected_row["Percentile"].iloc[0]) if not selected_row.empty else np.nan,
    }


IDENTITY_COMPONENTS = {
    "High": [
        ("Final-Third Ball Wins / Match", True),
        ("Final-Third Ball Win %", True),
        ("Opponent-Box Ball Wins / Match", True),
    ],
    "Active": [
        ("Presses / Match", True),
        ("Counterpresses / Match", True),
        ("Second-Ball Win %", True),
    ],
    "Tight": [
        ("xG Conceded / Match", False),
        ("Goals Conceded / Match", False),
        ("Suffered Bypassed Opponents / Match", False),
    ],
}


def identity_components(
    league_summary: pd.DataFrame,
    selected_summary: pd.DataFrame,
    selected_team: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for identity, components in IDENTITY_COMPONENTS.items():
        for metric, higher_is_better in components:
            benchmark = metric_benchmark(
                league_summary,
                selected_summary,
                selected_team,
                metric,
                higher_is_better,
            )
            rows.append(
                {
                    "Identity": identity,
                    "Metric": metric,
                    "Selected Value": benchmark["value"],
                    "League Average": benchmark["average"],
                    "Rank": benchmark["rank"],
                    "Teams": benchmark["teams"],
                    "Percentile": benchmark["percentile"],
                    "Better Direction": "Higher" if higher_is_better else "Lower",
                }
            )
    return pd.DataFrame(rows)


def identity_chart(components: pd.DataFrame, title: str) -> go.Figure:
    if components.empty:
        return charting.polish_figure(go.Figure(), title, height=350)

    scores = (
        components.groupby("Identity", as_index=False)["Percentile"]
        .mean()
        .set_index("Identity")
        .reindex(["Tight", "Active", "High"])
        .reset_index()
    )
    scores["Percentile"] = _number(scores["Percentile"]).fillna(0)
    scores["Colour"] = np.select(
        [scores["Percentile"].ge(67), scores["Percentile"].ge(33)],
        [GREEN, AMBER],
        default=RED,
    )
    detail_lookup = {}
    for identity, group in components.groupby("Identity"):
        detail_lookup[identity] = "<br>".join(
            f"{row['Metric']}: {float(row['Percentile']):.0f}th percentile"
            for _, row in group.iterrows()
            if pd.notna(row["Percentile"])
        )
    scores["Detail"] = scores["Identity"].map(detail_lookup).fillna("")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[100] * len(scores),
            y=scores["Identity"],
            orientation="h",
            marker_color="#eef2f6",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Bar(
            x=scores["Percentile"],
            y=scores["Identity"],
            orientation="h",
            marker=dict(color=scores["Colour"], line=dict(color="#ffffff", width=1)),
            text=[f"{value:.0f}" for value in scores["Percentile"]],
            textposition="inside",
            textfont=dict(color="#ffffff", size=14),
            customdata=np.stack([scores["Detail"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.0f}/100<br>%{customdata[0]}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(x=50, line=dict(color=GREY, width=1.4, dash="dash"))
    fig.update_layout(barmode="overlay", showlegend=False)
    fig.update_xaxes(range=[0, 100], dtick=25, title="League Percentile Score")
    fig.update_yaxes(categoryorder="array", categoryarray=["Tight", "Active", "High"], title="")
    return charting.polish_figure(fig, title, height=360)


def filter_regain_events(events: pd.DataFrame, view: str = "All Regains") -> pd.DataFrame:
    if events.empty:
        return events.copy()
    out = events.copy()
    action_type = out["Action Type"].fillna("").astype(str).str.upper()
    out = out[action_type.isin(REGAIN_ACTION_TYPES)].copy()
    out["Start X"] = _number(out["Start X"])
    out["Start Y"] = _number(out["Start Y"])
    out = out.dropna(subset=["Start X", "Start Y"])
    view_key = str(view).lower()
    if "opposition-half" in view_key:
        out = out[out["Start X"].ge(0)]
    elif "final-third" in view_key:
        out = out[out["Start X"].ge(pitch.FINAL_THIRD_X)]
    elif "second-ball" in view_key:
        out = out[out["Phase"].fillna("").astype(str).str.upper().eq("SECOND_BALL")]
    elif "interception" in view_key:
        out = out[out["Action Type"].fillna("").astype(str).str.upper().eq("INTERCEPTION")]
    return out.reset_index(drop=True)


def regain_density_map(regains: pd.DataFrame, title: str) -> go.Figure:
    fig = pitch.pitch_figure(title, height=620, legend=False)
    if regains.empty:
        fig.add_annotation(x=0, y=0, text="No regain locations", showarrow=False, font=dict(size=16, color=GREY))
        return fig

    x = _number(regains["Start X"]).dropna()
    y = _number(regains.loc[x.index, "Start Y"])
    valid = y.notna()
    x = x[valid]
    y = y[valid]
    x_edges = np.linspace(pitch.PITCH_X_MIN, pitch.PITCH_X_MAX, 15)
    y_edges = np.linspace(pitch.PITCH_Y_MIN, pitch.PITCH_Y_MAX, 11)
    histogram, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    x_centres = (x_edges[:-1] + x_edges[1:]) / 2
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2
    max_count = max(float(histogram.max()), 1.0)

    fig.add_trace(
        go.Heatmap(
            x=x_centres,
            y=y_centres,
            z=histogram.T,
            zmin=0,
            zmax=max_count,
            colorscale=[
                [0.0, "rgba(255,255,255,0.00)"],
                [0.15, "rgba(255,229,188,0.42)"],
                [0.48, "rgba(242,154,46,0.58)"],
                [0.75, "rgba(211,54,54,0.68)"],
                [1.0, "rgba(126,0,20,0.82)"],
            ],
            colorbar=dict(title="Events", thickness=14, len=0.72, x=1.01),
            hovertemplate="Regain-zone count: %{z:.0f}<extra></extra>",
            name="Regain density",
        )
    )
    customdata = np.stack(
        [
            regains["Player"].fillna("Unknown"),
            regains["Action Type"].fillna("Regain"),
            _number(regains["Minute"]).fillna(0),
            regains["Phase"].fillna("Unknown"),
        ],
        axis=-1,
    )
    fig.add_trace(
        go.Scatter(
            x=regains["Start X"],
            y=regains["Start Y"],
            mode="markers",
            marker=dict(size=6, color="rgba(17,17,17,0.20)", line=dict(width=0)),
            customdata=customdata,
            hovertemplate=(
                "%{customdata[0]} - %{customdata[1]}"
                "<br>Minute: %{customdata[2]:.0f}"
                "<br>Phase: %{customdata[3]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    for boundary in [-17.5, 17.5]:
        fig.add_shape(
            type="line",
            x0=boundary,
            x1=boundary,
            y0=pitch.PITCH_Y_MIN,
            y1=pitch.PITCH_Y_MAX,
            line=dict(color="rgba(52,64,84,0.45)", width=1, dash="dot"),
            layer="above",
        )
    fig.add_annotation(
        x=0,
        y=pitch.PITCH_Y_MIN - 0.8,
        text="Attacking direction →",
        showarrow=False,
        font=dict(size=12, color=DARK),
        yanchor="top",
    )
    fig.add_annotation(
        x=pitch.PITCH_X_MIN + 1,
        y=pitch.PITCH_Y_MAX - 1,
        text="Own goal",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=11, color=GREY),
    )
    fig.add_annotation(
        x=pitch.PITCH_X_MAX - 1,
        y=pitch.PITCH_Y_MAX - 1,
        text="Opponent goal",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font=dict(size=11, color=GREY),
    )
    fig.update_layout(margin=dict(l=28, r=94, t=104, b=42))
    return fig


def pressing_context_chart(
    selected_summary: pd.DataFrame,
    league_summary: pd.DataFrame,
    selected_team: str,
    title: str,
) -> go.Figure:
    metrics = [
        ("All Presses", "Presses / Match"),
        ("Counterpress", "Counterpresses / Match"),
        ("Build-Up", "Build-Up Presses / Match"),
        ("Between Lines", "Between-Lines Presses / Match"),
    ]
    selected = selected_summary[selected_summary["Team"].astype(str).eq(str(selected_team))]
    if selected.empty:
        return charting.polish_figure(go.Figure(), title, height=430)
    selected_values = [float(_number(selected[metric]).iloc[0]) for _, metric in metrics]
    league_values = [float(_number(league_summary[metric]).mean()) for _, metric in metrics]
    labels = [label for label, _ in metrics]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=selected_values,
            y=labels,
            orientation="h",
            name=selected_team,
            marker_color=RED,
            text=[f"{value:.1f}" for value in selected_values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.2f} per match<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=league_values,
            y=labels,
            orientation="h",
            name="League Average",
            marker_color="#c4cbd4",
            text=[f"{value:.1f}" for value in league_values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.2f} per match<extra></extra>",
        )
    )
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Actions Per Match", rangemode="tozero")
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(labels)), title="")
    return charting.polish_figure(fig, title, height=440)


def regain_sequence_outcomes(events: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Link spatial-regain proxies to later actions in the same sequence."""
    if events.empty:
        return pd.DataFrame(columns=["Stage", "Sequences", "Conversion %"]), 0.0

    working = events.copy()
    action_type = working["Action Type"].fillna("").astype(str).str.upper()
    working["_Order"] = _number(working.get("Event Number", pd.Series(index=working.index, dtype=float)))
    working["_Order"] = working["_Order"].fillna(pd.Series(np.arange(len(working)), index=working.index))
    working["_Sequence"] = working["Sequence Index"]
    working = working.dropna(subset=["MatchId", "_Sequence"])
    if working.empty:
        return pd.DataFrame(columns=["Stage", "Sequences", "Conversion %"]), 0.0

    regain_mask = action_type.loc[working.index].isin(REGAIN_ACTION_TYPES)
    first_regains = (
        working[regain_mask]
        .groupby(["MatchId", "_Sequence"], as_index=False)["_Order"]
        .min()
        .rename(columns={"_Order": "_Regain Order"})
    )
    if first_regains.empty:
        return pd.DataFrame(columns=["Stage", "Sequences", "Conversion %"]), 0.0

    post = working.merge(first_regains, on=["MatchId", "_Sequence"], how="inner")
    post = post[post["_Order"].ge(post["_Regain Order"])].copy()
    post_action_type = post["Action Type"].fillna("").astype(str).str.upper()
    post_action = post["Action"].fillna("").astype(str).str.upper()
    start_x = _number(post["Start X"])
    end_x = _number(post["End X"])
    post["_Reached Final Third"] = start_x.ge(pitch.FINAL_THIRD_X) | end_x.ge(pitch.FINAL_THIRD_X)
    post["_Shot"] = post_action_type.eq("SHOT")
    post["_Goal"] = post_action.eq("GOAL")
    post["_Shot xG"] = _number(post["Shot xG"]).fillna(0).where(post["_Shot"], 0)

    sequences = post.groupby(["MatchId", "_Sequence"], as_index=False).agg(
        **{
            "Reached Final Third": ("_Reached Final Third", "max"),
            "Shot": ("_Shot", "max"),
            "Goal": ("_Goal", "max"),
            "Shot xG": ("_Shot xG", "sum"),
        }
    )
    sequences["Reached Final Third"] = sequences[["Reached Final Third", "Shot"]].max(axis=1)
    sequences["Shot"] = sequences[["Shot", "Goal"]].max(axis=1)
    total = len(sequences)
    counts = [
        total,
        int(sequences["Reached Final Third"].sum()),
        int(sequences["Shot"].sum()),
        int(sequences["Goal"].sum()),
    ]
    stages = ["Regain Sequences", "Reached Final Third", "Produced a Shot", "Produced a Goal"]
    summary = pd.DataFrame({"Stage": stages, "Sequences": counts})
    summary["Conversion %"] = summary["Sequences"].div(max(total, 1)).mul(100)
    return summary, float(sequences["Shot xG"].sum())


def regain_conversion_chart(summary: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if summary.empty:
        return charting.polish_figure(fig, title, height=430)
    fig.add_trace(
        go.Funnel(
            y=summary["Stage"],
            x=summary["Sequences"],
            textinfo="value+percent initial",
            marker=dict(color=[DARK, BLUE, RED, GREEN]),
            connector=dict(line=dict(color=LIGHT_GREY, width=1)),
            customdata=np.stack([summary["Conversion %"]], axis=-1),
            hovertemplate="%{y}: %{x:.0f} sequences<br>%{customdata[0]:.1f}% of regains<extra></extra>",
        )
    )
    return charting.polish_figure(fig, title, height=440)


def player_diverging_chart(
    players: pd.DataFrame,
    won_column: str,
    lost_column: str,
    title: str,
    won_label: str = "Won",
    lost_label: str = "Lost",
    top_n: int = 18,
) -> go.Figure:
    fig = go.Figure()
    if players.empty or won_column not in players or lost_column not in players:
        return charting.polish_figure(fig, title, height=440)

    plot_df = players.copy()
    plot_df[won_column] = _number(plot_df[won_column]).fillna(0)
    plot_df[lost_column] = _number(plot_df[lost_column]).fillna(0)
    plot_df["_Involvement"] = plot_df[won_column] + plot_df[lost_column]
    plot_df = plot_df[plot_df["_Involvement"].gt(0)].nlargest(top_n, "_Involvement")
    plot_df = plot_df.sort_values("_Involvement", ascending=True)
    if plot_df.empty:
        return charting.polish_figure(fig, title, height=440)
    plot_df["_Win %"] = _safe_ratio(plot_df[won_column], plot_df["_Involvement"], 100)
    customdata = np.stack(
        [plot_df["_Win %"].fillna(0), plot_df["Minutes"].fillna(0), plot_df["Position"].fillna("")],
        axis=-1,
    )
    fig.add_trace(
        go.Bar(
            x=-plot_df[lost_column],
            y=plot_df["Player"],
            orientation="h",
            name=lost_label,
            marker_color=RED,
            text=[f"{value:.0f}" for value in plot_df[lost_column]],
            customdata=customdata,
            hovertemplate=(
                "%{y}<br>" + lost_label + ": %{x:.0f}"
                "<br>Win rate: %{customdata[0]:.1f}%"
                "<br>Minutes: %{customdata[1]:.0f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=plot_df[won_column],
            y=plot_df["Player"],
            orientation="h",
            name=won_label,
            marker_color=GREEN,
            text=[f"{value:.0f}" for value in plot_df[won_column]],
            customdata=customdata,
            hovertemplate=(
                "%{y}<br>" + won_label + ": %{x:.0f}"
                "<br>Win rate: %{customdata[0]:.1f}%"
                "<br>Minutes: %{customdata[1]:.0f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line=dict(color=DARK, width=1.2))
    fig.update_layout(barmode="relative")
    fig.update_xaxes(title=f"← {lost_label} | {won_label} →", zeroline=False)
    fig.update_yaxes(title="")
    return charting.polish_figure(
        fig,
        title,
        height=charting.horizontal_bar_height(len(plot_df), min_height=450, row_height=34, max_height=780),
    )


def second_ball_player_chart(players: pd.DataFrame, title: str, top_n: int = 18) -> go.Figure:
    fig = go.Figure()
    if players.empty:
        return charting.polish_figure(fig, title, height=440)
    plot_df = players.copy()
    plot_df["Second Balls"] = _number(plot_df["Second Balls"]).fillna(0)
    plot_df["Second-Ball Win %"] = _number(plot_df["Second-Ball Win %"])
    plot_df = plot_df[plot_df["Second Balls"].gt(0)].nlargest(top_n, "Second Balls")
    plot_df = plot_df.sort_values("Second-Ball Win %", ascending=True)
    if plot_df.empty:
        return charting.polish_figure(fig, title, height=440)
    fig.add_trace(
        go.Bar(
            x=plot_df["Second-Ball Win %"],
            y=plot_df["Player"],
            orientation="h",
            marker=dict(
                color=plot_df["Second-Ball Win %"],
                colorscale=[[0, RED], [0.5, AMBER], [1, GREEN]],
                cmin=0,
                cmax=100,
                colorbar=dict(title="Win %", thickness=14),
            ),
            text=[f"{value:.1f}%" for value in plot_df["Second-Ball Win %"]],
            customdata=np.stack(
                [plot_df["Second Balls"], plot_df["Second Balls Won"], plot_df["Minutes"]], axis=-1
            ),
            hovertemplate=(
                "%{y}<br>Win rate: %{x:.1f}%"
                "<br>Won: %{customdata[1]:.0f} of %{customdata[0]:.0f}"
                "<br>Minutes: %{customdata[2]:.0f}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(range=[0, 100], title="Second-Ball Win %", ticksuffix="%")
    fig.update_yaxes(title="")
    return charting.polish_figure(
        fig,
        title,
        height=charting.horizontal_bar_height(len(plot_df), min_height=450, row_height=34, max_height=780),
    )


def player_contribution_scatter(players: pd.DataFrame, min_minutes: float, title: str) -> go.Figure:
    fig = go.Figure()
    if players.empty:
        return charting.polish_figure(fig, title, height=560)
    plot_df = players[
        _number(players["Minutes"]).ge(float(min_minutes))
        & _number(players["Ball Wins /90"]).notna()
        & _number(players["Opponents Removed / Ball Win"]).notna()
    ].copy()
    if plot_df.empty:
        return charting.polish_figure(fig, title, height=560)

    minutes = _number(plot_df["Minutes"]).fillna(0)
    max_minutes = max(float(minutes.max()), 1.0)
    marker_sizes = 15 + np.sqrt(minutes / max_minutes) * 28
    labels = pd.Series("", index=plot_df.index)
    label_indices = plot_df.nlargest(min(6, len(plot_df)), "Ball Wins /90").index
    labels.loc[label_indices] = plot_df.loc[label_indices, "Player"]
    customdata = np.stack(
        [
            plot_df["Player"],
            plot_df["Position"].fillna(""),
            minutes,
            plot_df["Ball Wins"],
            plot_df["Duel Win %"].fillna(0),
            plot_df["Ball Win Value /90"].fillna(0),
        ],
        axis=-1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["Ball Wins /90"],
            y=plot_df["Opponents Removed / Ball Win"],
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont=dict(size=11, color=DARK),
            marker=dict(
                size=marker_sizes,
                color=plot_df["Duel Win %"].fillna(0),
                colorscale=[[0, RED], [0.5, AMBER], [1, GREEN]],
                cmin=0,
                cmax=100,
                colorbar=dict(title="Duel Win %", thickness=14),
                opacity=0.86,
                line=dict(color="#ffffff", width=1.2),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b> · %{customdata[1]}"
                "<br>Minutes: %{customdata[2]:.0f}"
                "<br>Ball wins: %{customdata[3]:.0f}"
                "<br>Ball wins /90: %{x:.2f}"
                "<br>Opponents removed / win: %{y:.2f}"
                "<br>Duel win: %{customdata[4]:.1f}%"
                "<br>Ball-win value /90: %{customdata[5]:.3f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_vline(x=float(_number(plot_df["Ball Wins /90"]).mean()), line=dict(color=GREY, width=1, dash="dash"))
    fig.add_hline(
        y=float(_number(plot_df["Opponents Removed / Ball Win"]).mean()),
        line=dict(color=GREY, width=1, dash="dash"),
    )
    fig.update_xaxes(title="Ball Wins /90", rangemode="tozero")
    fig.update_yaxes(title="Opponents Removed Per Ball Win", rangemode="tozero")
    return charting.polish_figure(fig, title, height=580)


def add_match_context(team_rows: pd.DataFrame, matches: pd.DataFrame, team_name: str) -> pd.DataFrame:
    if team_rows.empty:
        return team_rows.copy()
    out = team_rows.copy()
    out["_Match Key"] = out["MatchId"].astype(str)
    match_context = matches.copy()
    match_context["_Match Key"] = match_context["MatchId"].astype(str)
    columns = [column for column in ["_Match Key", "Home", "Away", "Date"] if column in match_context]
    match_context = match_context[columns].drop_duplicates("_Match Key")
    if "Date" in out and "Date" in match_context:
        match_context = match_context.rename(columns={"Date": "_Fixture Date"})
    out = out.merge(match_context, on="_Match Key", how="left")
    if "_Fixture Date" in out:
        out["Date"] = pd.to_datetime(out["_Fixture Date"], errors="coerce").fillna(
            pd.to_datetime(out["Date"], errors="coerce")
        )
    home = out.get("Home", pd.Series("", index=out.index)).fillna("").astype(str)
    away = out.get("Away", pd.Series("", index=out.index)).fillna("").astype(str)
    out["Opponent"] = np.where(home.eq(str(team_name)), away, home)
    out["Venue"] = np.where(home.eq(str(team_name)), "Home", "Away")
    date_text = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%d %b %Y").fillna("Date unknown")
    out["Match Label"] = date_text + " · " + out["Venue"] + " vs " + out["Opponent"].fillna("Unknown")

    out["Ball Wins / Match"] = _number(out["Ball Wins"])
    out["Presses / Match"] = _number(out["Presses"])
    out["Opponents Removed / Ball Win"] = _safe_ratio(out["Opponents Removed"], out["Ball Wins"])
    out["Second-Ball Win %"] = _safe_ratio(out["Second Balls Won"], out["Second Balls"], 100)
    out["Ground Duel Win %"] = _safe_ratio(
        out["Ground Duels Won"], out["Ground Duels Won"] + out["Ground Duels Lost"], 100
    )
    out["Aerial Duel Win %"] = _safe_ratio(
        out["Aerial Duels Won"], out["Aerial Duels Won"] + out["Aerial Duels Lost"], 100
    )
    out["xG Conceded / Match"] = _number(out["xG Conceded"])
    return out.sort_values(["Date", "MatchId"]).reset_index(drop=True)


def match_trend_chart(match_rows: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = go.Figure()
    if match_rows.empty or metric not in match_rows:
        return charting.polish_figure(fig, title, height=470)
    plot_df = match_rows.copy()
    plot_df[metric] = _number(plot_df[metric])
    plot_df = plot_df[plot_df[metric].notna()].sort_values("Date")
    if plot_df.empty:
        return charting.polish_figure(fig, title, height=470)
    x_values = pd.to_datetime(plot_df["Date"], errors="coerce")
    if x_values.isna().all():
        x_values = np.arange(len(plot_df))
    customdata = np.stack(
        [plot_df["Match Label"], plot_df["Opponent"], plot_df["Venue"]], axis=-1
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=plot_df[metric],
            mode="lines+markers",
            name=metric,
            line=dict(color=RED, width=2.5),
            marker=dict(size=9, color=RED, line=dict(color="#ffffff", width=1.2)),
            customdata=customdata,
            hovertemplate="%{customdata[0]}<br>" + metric + ": %{y:.2f}<extra></extra>",
        )
    )
    average = float(plot_df[metric].mean())
    if len(plot_df) > 1:
        fig.add_trace(
            go.Scatter(
                x=[x_values.iloc[0], x_values.iloc[-1]] if hasattr(x_values, "iloc") else [x_values[0], x_values[-1]],
                y=[average, average],
                mode="lines",
                name="Selected-Window Average",
                line=dict(color=GREY, width=1.5, dash="dash"),
                hovertemplate=f"Average: {average:.2f}<extra></extra>",
            )
        )
    fig.update_xaxes(title="Match")
    fig.update_yaxes(title=metric, rangemode="tozero")
    return charting.polish_figure(fig, title, height=480)
