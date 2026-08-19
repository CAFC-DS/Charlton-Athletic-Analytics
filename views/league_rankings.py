# =============================================================================
# LEAGUE RANKINGS - direction-aware team comparison across league metrics
# =============================================================================
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charting, data, match_analysis as ma, team_analysis as ta, ui


PAGE_SOURCE = (
    "Results, attacking and defending measures are calculated from completed regular-season match rows. "
    "Possession, progression and additional output measures use provider-authored CAFC_DB Impect squad-iteration KPI facts."
)

METRIC_GROUPS: dict[str, list[tuple[str, bool, str]]] = {
    "Results": [
        ("Points / Match", True, "Match results"),
        ("Win %", True, "Match results"),
        ("Loss %", False, "Match results"),
        ("Goal Difference / Match", True, "Match results"),
    ],
    "Attacking": [
        ("Goals For / Match", True, "Match results"),
        ("Scoring Match %", True, "Match results"),
        ("Goals /90", True, "Team style rollup"),
        ("xG /90", True, "Team style rollup"),
        ("Packing xG /90", True, "Team style rollup"),
        ("Shots /90", True, "Team style rollup"),
        ("Assists /90", True, "Team style rollup"),
    ],
    "Defending": [
        ("Goals Against / Match", False, "Match results"),
        ("Clean Sheet %", True, "Match results"),
        ("Ball Wins /90", True, "Team style rollup"),
        ("Ball Win Value /90", True, "Team style rollup"),
    ],
    "Possession": [
        ("Pass %", True, "Team style rollup"),
    ],
    "Progression": [
        ("Bypassed Opponents /90", True, "Team style rollup"),
        ("Passes to Final 3rd /90", True, "Team style rollup"),
        ("Dribble Progression /90", True, "Team style rollup"),
    ],
}

RED = ui.CHARLTON_RED
DARK = ui.CHARLTON_BLACK
GREEN = "#16803c"
AMBER = "#d89216"
GREY = "#667085"
LIGHT_GREY = "#d0d5dd"
LOW_STANDING = "#7b8794"
STYLE_METRICS = {"Pass %", "Bypassed Opponents /90", "Passes to Final 3rd /90", "Dribble Progression /90"}


def _inject_rankings_css() -> None:
    st.markdown(
        """
        <style>
        .lr-overview-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 8px 0 18px;
        }
        .lr-overview-card {
            min-width: 0;
            padding: 16px 17px;
            border: 1px solid #e2e7ee;
            border-top: 4px solid #98a2b3;
            border-radius: 9px;
            background: #ffffff;
            box-shadow: 0 5px 18px rgba(16, 24, 40, 0.055);
        }
        .lr-overview-card.strong { border-top-color: #16803c; }
        .lr-overview-card.focus { border-top-color: #c30017; }
        .lr-overview-card.watch { border-top-color: #7b8794; }
        .lr-overview-card.neutral { border-top-color: #344054; }
        .lr-card-label {
            min-height: 2.4em;
            color: #667085;
            font-size: 0.74rem;
            font-weight: 850;
            letter-spacing: 0.065em;
            line-height: 1.25;
            text-transform: uppercase;
        }
        .lr-card-value {
            color: #101828;
            font-size: 1.58rem;
            font-weight: 850;
            letter-spacing: -0.035em;
            line-height: 1.08;
            margin: 9px 0 7px;
        }
        .lr-card-note {
            color: #667085;
            font-size: 0.78rem;
            line-height: 1.4;
        }
        .lr-insight {
            display: grid;
            grid-template-columns: minmax(190px, 1.1fr) repeat(4, minmax(130px, 0.8fr));
            gap: 0;
            margin: 8px 0 20px;
            overflow: hidden;
            border: 1px solid #dfe5ec;
            border-radius: 10px;
            background: #ffffff;
            box-shadow: 0 6px 22px rgba(16, 24, 40, 0.06);
        }
        .lr-insight-cell {
            min-width: 0;
            padding: 15px 16px;
            border-left: 1px solid #eaecf0;
        }
        .lr-insight-cell:first-child {
            border-left: 0;
            background: linear-gradient(135deg, #fff1f3 0%, #ffffff 100%);
        }
        .lr-insight-label {
            color: #667085;
            font-size: 0.7rem;
            font-weight: 850;
            letter-spacing: 0.055em;
            text-transform: uppercase;
        }
        .lr-insight-value {
            color: #172033;
            font-size: 1.12rem;
            font-weight: 850;
            line-height: 1.2;
            margin-top: 5px;
        }
        .lr-insight-note {
            color: #667085;
            font-size: 0.74rem;
            line-height: 1.35;
            margin-top: 4px;
        }
        .lr-callout {
            padding: 15px 17px;
            margin: 8px 0 18px;
            border: 1px solid #dfe5ec;
            border-left: 4px solid #c30017;
            border-radius: 9px;
            background: #f8fafc;
        }
        .lr-callout-label {
            color: #c30017;
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            margin-bottom: 5px;
            text-transform: uppercase;
        }
        .lr-callout-copy {
            color: #253045;
            font-size: 0.92rem;
            line-height: 1.5;
        }
        .lr-direction {
            display: inline-block;
            padding: 4px 8px;
            margin-left: 5px;
            border-radius: 999px;
            background: #f2f4f7;
            color: #344054;
            font-size: 0.72rem;
            font-weight: 800;
        }
        @media (max-width: 980px) {
            .lr-overview-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .lr-insight { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .lr-insight-cell { border-left: 0; }
            .lr-insight-cell:nth-child(even) { border-left: 1px solid #eaecf0; }
            .lr-insight-cell:nth-child(n+3) { border-top: 1px solid #eaecf0; }
            .lr-insight-cell:last-child { grid-column: 1 / -1; border-left: 0; }
        }
        @media (max-width: 600px) {
            .lr-overview-grid, .lr-insight { grid-template-columns: minmax(0, 1fr); }
            .lr-insight-cell { border-left: 0; border-top: 1px solid #eaecf0; }
            .lr-insight-cell:first-child { border-top: 0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _team_key(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", "" if value is None else str(value).lower()).strip()
    return " ".join(word for word in text.split() if word not in {"fc", "afc", "cf", "football", "club"})


def _metric_text(metric: str, value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "N/A"
    if metric.endswith("%"):
        return f"{float(number):.1f}%"
    return f"{float(number):.2f}"


def _amount_text(metric: str, value: float) -> str:
    if "%" in metric:
        decimals = 2 if 0 < abs(float(value)) < 0.1 else 1
        return f"{abs(float(value)):.{decimals}f} pp"
    return f"{abs(float(value)):.2f}"


def _performance_gap_text(metric: str, value: float, benchmark: float, higher_is_better: bool) -> str:
    raw_gap = float(value) - float(benchmark)
    tolerance = 0.05 if "%" in metric else 0.005
    if abs(raw_gap) < tolerance:
        return "Level with league average"
    if metric in STYLE_METRICS:
        relation = "above" if raw_gap > 0 else "below"
        return f"{_amount_text(metric, raw_gap)} {relation} league average"
    performance_gap = raw_gap if higher_is_better else -raw_gap
    outcome = "better" if performance_gap > 0 else "worse"
    return f"{_amount_text(metric, performance_gap)} {outcome} than league average"


def _relative_text(
    metric: str,
    peer_value: float,
    selected_value: float,
    higher_is_better: bool,
) -> str:
    performance_gap = (
        float(peer_value) - float(selected_value)
        if higher_is_better
        else float(selected_value) - float(peer_value)
    )
    tolerance = 0.05 if "%" in metric else 0.005
    if abs(performance_gap) < tolerance:
        return "Level"
    relation = "ahead" if performance_gap > 0 else "behind"
    return f"{_amount_text(metric, performance_gap)} {relation}"


def _match_metric_table(fixtures: pd.DataFrame) -> pd.DataFrame:
    table = ma.team_record_table(fixtures)
    if table.empty:
        return pd.DataFrame()

    table = table.copy()
    numeric_columns = ["Played", "Wins", "Losses", "Points", "GF", "GA", "GD"]
    for column in numeric_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")

    played = table["Played"].replace(0, pd.NA).astype("Float64")
    metrics = pd.DataFrame({"Team": table["Team"].astype(str)})
    metrics["Matches Played"] = table["Played"]
    metrics["Points / Match"] = (table["Points"] / played).astype(float).round(2)
    metrics["Win %"] = (table["Wins"] / played * 100).astype(float).round(1)
    metrics["Loss %"] = (table["Losses"] / played * 100).astype(float).round(1)
    metrics["Goal Difference / Match"] = (table["GD"] / played).astype(float).round(2)
    metrics["Goals For / Match"] = (table["GF"] / played).astype(float).round(2)
    metrics["Goals Against / Match"] = (table["GA"] / played).astype(float).round(2)

    clean_sheet_rates: list[float] = []
    scoring_match_rates: list[float] = []
    for team_name in metrics["Team"]:
        team_rows = ma.team_match_rows(fixtures, team_name)
        goals_for = pd.to_numeric(team_rows["Goals For"], errors="coerce")
        goals_against = pd.to_numeric(team_rows["Goals Against"], errors="coerce")
        denominator = len(team_rows)
        clean_sheet_rates.append(float(goals_against.eq(0).sum()) / denominator * 100 if denominator else float("nan"))
        scoring_match_rates.append(float(goals_for.gt(0).sum()) / denominator * 100 if denominator else float("nan"))

    metrics["Clean Sheet %"] = pd.Series(clean_sheet_rates, index=metrics.index).round(1)
    metrics["Scoring Match %"] = pd.Series(scoring_match_rates, index=metrics.index).round(1)
    return metrics


def _combine_metric_sources(match_metrics: pd.DataFrame, style_metrics: pd.DataFrame) -> pd.DataFrame:
    if match_metrics.empty and style_metrics.empty:
        return pd.DataFrame()
    if match_metrics.empty:
        return style_metrics.copy()
    if style_metrics.empty:
        return match_metrics.copy()

    combined = match_metrics.copy()
    combined["_Team Key"] = combined["Team"].map(_team_key)
    style = style_metrics.copy()
    style["_Team Key"] = style["Team"].map(_team_key)
    style_columns = [metric for metric in data.TEAM_METRICS if metric in style]
    style = style[["_Team Key", *style_columns]].drop_duplicates("_Team Key")
    return combined.merge(style, on="_Team Key", how="left").drop(columns="_Team Key")


def _available_metric_groups(metrics: pd.DataFrame) -> dict[str, list[tuple[str, bool, str]]]:
    groups = {
        group: [
            (metric, higher_is_better, source)
            for metric, higher_is_better, source in specs
            if metric in metrics and pd.to_numeric(metrics[metric], errors="coerce").notna().any()
        ]
        for group, specs in METRIC_GROUPS.items()
    }
    return {group: specs for group, specs in groups.items() if specs}


def _league_rank_table(metrics: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    """Return a page-local rank table with rank-consistent tie percentiles."""
    ranked = ta.metric_rank_table(metrics, metric, higher_is_better).copy()
    team_count = len(ranked)
    if team_count:
        rank_values = pd.to_numeric(ranked["Rank"], errors="coerce")
        ranked["Percentile"] = ((team_count - rank_values + 1) / team_count * 100).round(1)
    return ranked


def _team_metric_profile(metrics: pd.DataFrame, team_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category, specs in _available_metric_groups(metrics).items():
        for metric, higher_is_better, source in specs:
            ranked = _league_rank_table(metrics, metric, higher_is_better)
            selected = ranked[ranked["Team"].astype(str).eq(str(team_name))]
            if selected.empty:
                continue
            selected_row = selected.iloc[0]
            average = float(ranked["Value"].mean())
            raw_gap = float(selected_row["Value"]) - average
            rows.append(
                {
                    "Category": category,
                    "Metric": metric,
                    "Value": float(selected_row["Value"]),
                    "League Average": average,
                    "Raw Gap": raw_gap,
                    "Performance Gap": raw_gap if higher_is_better else -raw_gap,
                    "Rank": int(selected_row["Rank"]),
                    "Teams": len(ranked),
                    "Percentile": float(selected_row["Percentile"]),
                    "Vs League Average": _performance_gap_text(
                        metric,
                        float(selected_row["Value"]),
                        average,
                        higher_is_better,
                    ),
                    "Ranking Direction": "Higher ranks higher" if higher_is_better else "Lower ranks higher",
                    "Higher Is Better": higher_is_better,
                    "Source": source,
                }
            )
    return pd.DataFrame(rows)


def _ranking_profile(metrics: pd.DataFrame, team_name: str) -> pd.DataFrame:
    profile = _team_metric_profile(metrics, team_name)
    if profile.empty:
        return profile
    display = profile.copy()
    display["Value"] = [
        _metric_text(metric, value) for metric, value in zip(display["Metric"], display["Value"], strict=False)
    ]
    display["League Average"] = [
        _metric_text(metric, value)
        for metric, value in zip(display["Metric"], display["League Average"], strict=False)
    ]
    display["Rank"] = [
        f"{int(rank)} of {int(teams)}" for rank, teams in zip(display["Rank"], display["Teams"], strict=False)
    ]
    return display[
        [
            "Category",
            "Metric",
            "Value",
            "League Average",
            "Vs League Average",
            "Rank",
            "Percentile",
            "Ranking Direction",
            "Source",
        ]
    ]


def _callout(label: str, copy: str) -> None:
    st.markdown(
        f'<div class="lr-callout"><div class="lr-callout-label">{ui.esc(label)}</div>'
        f'<div class="lr-callout-copy">{ui.esc(copy)}</div></div>',
        unsafe_allow_html=True,
    )


def _profile_overview_cards(profile: pd.DataFrame, team_name: str) -> None:
    if profile.empty:
        return
    outcomes = profile[~profile["Metric"].isin(STYLE_METRICS)].copy()
    summary = outcomes if not outcomes.empty else profile.copy()
    is_outcome_summary = not outcomes.empty
    strongest = summary.sort_values(["Rank", "Metric"], ascending=[True, True]).iloc[0]
    weakest = summary.sort_values(["Rank", "Metric"], ascending=[False, True]).iloc[0]
    top_third = int(summary["Percentile"].ge(67).sum())
    median_percentile = float(summary["Percentile"].median())
    total_metrics = len(summary)
    measure_label = "outcome" if is_outcome_summary else "style"
    cards = [
        (
            "strong" if is_outcome_summary else "neutral",
            f"Highest {measure_label} rank",
            f"#{int(strongest['Rank'])}",
            f"{strongest['Metric']} · {_metric_text(str(strongest['Metric']), strongest['Value'])}",
        ),
        (
            "watch" if is_outcome_summary else "neutral",
            f"Lowest {measure_label} rank",
            f"#{int(weakest['Rank'])}",
            f"{weakest['Metric']} · {_metric_text(str(weakest['Metric']), weakest['Value'])}",
        ),
        (
            "neutral",
            f"Top-third {measure_label}s",
            f"{top_third} of {total_metrics}",
            f"Direction-adjusted {measure_label} standings",
        ),
        (
            "neutral",
            f"Median {measure_label} standing",
            f"Score {median_percentile:.0f}",
            f"Across {team_name}'s available {measure_label} measures",
        ),
    ]
    html = []
    for status, label, value, note in cards:
        html.append(
            f'<div class="lr-overview-card {status}">'
            f'<div class="lr-card-label">{ui.esc(label)}</div>'
            f'<div class="lr-card-value">{ui.esc(value)}</div>'
            f'<div class="lr-card-note">{ui.esc(note)}</div></div>'
        )
    st.markdown(f'<div class="lr-overview-grid">{"".join(html)}</div>', unsafe_allow_html=True)


def _next_place_context(
    ranked: pd.DataFrame,
    selected_team: str,
    metric: str,
    higher_is_better: bool,
) -> tuple[str, str, str]:
    selected_rows = ranked[ranked["Team"].astype(str).eq(str(selected_team))]
    if selected_rows.empty:
        return "Next place", "N/A", "No comparison available"
    selected_row = selected_rows.iloc[0]
    selected_rank = int(selected_row["Rank"])
    selected_value = float(selected_row["Value"])
    tied = ranked[ranked["Rank"].eq(selected_rank)]
    tied_peers = tied[~tied["Team"].astype(str).eq(str(selected_team))]
    if not tied_peers.empty:
        names = " / ".join(tied_peers["Team"].astype(str).head(2))
        if len(tied_peers) > 2:
            names += f" +{len(tied_peers) - 2}"
        return "Tied league rank", f"#{selected_rank} · {len(tied)} teams", f"Level with {names}"

    if selected_rank == 1:
        candidates = ranked[ranked["Rank"].gt(1)].sort_values("Rank")
        if candidates.empty:
            return "League lead", "N/A", "No second valid team"
        target_rank = int(candidates.iloc[0]["Rank"])
        targets = candidates[candidates["Rank"].eq(target_rank)]
        target = targets.iloc[0]
        names = " / ".join(targets["Team"].astype(str).head(2))
        if len(targets) > 2:
            names += f" +{len(targets) - 2}"
        gap = selected_value - float(target["Value"]) if higher_is_better else float(target["Value"]) - selected_value
        return "Lead over next", _amount_text(metric, gap), f"Over {names}"

    candidates = ranked[ranked["Rank"].lt(selected_rank)].sort_values("Rank", ascending=False)
    if candidates.empty:
        return "Next place", "N/A", "No stronger valid team"
    target_rank = int(candidates.iloc[0]["Rank"])
    targets = candidates[candidates["Rank"].eq(target_rank)]
    target = targets.iloc[0]
    names = " / ".join(targets["Team"].astype(str).head(2))
    if len(targets) > 2:
        names += f" +{len(targets) - 2}"
    gap = float(target["Value"]) - selected_value if higher_is_better else selected_value - float(target["Value"])
    return "Gap to next place", _amount_text(metric, gap), f"To match {names}"


def _metric_insight_strip(
    ranked: pd.DataFrame,
    selected: pd.Series,
    metric: str,
    team_name: str,
    higher_is_better: bool,
) -> None:
    league_average = float(ranked["Value"].mean())
    leaders = ranked[ranked["Rank"].eq(1)].copy()
    leader = leaders.iloc[0]
    leader_names = " / ".join(leaders["Team"].astype(str).head(2))
    if len(leaders) > 2:
        leader_names += f" +{len(leaders) - 2}"
    leader_label = "Joint leaders" if len(leaders) > 1 else "League leader"
    leader_note = (
        f"{_metric_text(metric, leader['Value'])} · {len(leaders)} teams tied"
        if len(leaders) > 1
        else f"{_metric_text(metric, leader['Value'])} · {metric}"
    )
    next_label, next_value, next_note = _next_place_context(ranked, team_name, metric, higher_is_better)
    cells = [
        (
            metric,
            f"#{int(selected['Rank'])} of {len(ranked)}",
            f"Percentile score {float(selected['Percentile']):.0f}",
        ),
        (team_name, _metric_text(metric, selected["Value"]), "Selected value"),
        (
            "League average",
            _metric_text(metric, league_average),
            _performance_gap_text(metric, float(selected["Value"]), league_average, higher_is_better),
        ),
        (next_label, next_value, next_note),
        (
            leader_label,
            leader_names,
            leader_note,
        ),
    ]
    html = []
    for label, value, note in cells:
        html.append(
            '<div class="lr-insight-cell">'
            f'<div class="lr-insight-label">{ui.esc(label)}</div>'
            f'<div class="lr-insight-value">{ui.esc(value)}</div>'
            f'<div class="lr-insight-note">{ui.esc(note)}</div></div>'
        )
    st.markdown(f'<div class="lr-insight">{"".join(html)}</div>', unsafe_allow_html=True)


def _profile_story(profile: pd.DataFrame, team_name: str) -> str:
    if profile.empty:
        return "No cross-metric profile is available for this team."
    outcomes = profile[~profile["Metric"].isin(STYLE_METRICS)].copy()
    if outcomes.empty:
        strongest = profile.sort_values(["Rank", "Metric"], ascending=[True, True]).iloc[0]
        weakest = profile.sort_values(["Rank", "Metric"], ascending=[False, True]).iloc[0]
        top_half = int(profile["Percentile"].ge(50).sum())
        return (
            f"{team_name}'s highest style standing is {strongest['Metric']} at rank {int(strongest['Rank'])} of "
            f"{int(strongest['Teams'])}; the lowest is {weakest['Metric']} at rank {int(weakest['Rank'])}. "
            f"{top_half} of {len(profile)} available style indicators sit in the upper half of the league distribution."
        )

    source = outcomes
    strongest = source.sort_values(["Rank", "Metric"], ascending=[True, True]).iloc[0]
    weakest = source.sort_values(["Rank", "Metric"], ascending=[False, True]).iloc[0]
    top_half = int(source["Percentile"].ge(50).sum())
    story = (
        f"{team_name}'s highest outcome standing is {strongest['Metric']} at rank {int(strongest['Rank'])} of "
        f"{int(strongest['Teams'])}; the lowest is {weakest['Metric']} at rank {int(weakest['Rank'])}. "
        f"{top_half} of {len(source)} outcome measures sit in the top half of the league."
    )

    lookup = source.set_index("Metric")
    if {"Scoring Match %", "Goals For / Match"}.issubset(lookup.index):
        scoring = float(lookup.loc["Scoring Match %", "Percentile"])
        output = float(lookup.loc["Goals For / Match", "Percentile"])
        if scoring - output >= 20:
            story += " Scoring-match frequency ranks notably higher than goals per match, indicating more regular scoring than multi-goal output."
        elif output - scoring >= 20:
            story += " Goals per match ranks notably higher than scoring-match frequency, indicating output is concentrated in fewer fixtures."
    return story


def _ranking_figure(
    ranked: pd.DataFrame,
    metric: str,
    selected_team: str,
    scope: str = "Full league",
) -> go.Figure:
    rows = ranked.copy().reset_index(drop=True)
    selected_rows = rows[rows["Team"].astype(str).eq(str(selected_team))]
    if scope == "Nearest ranks" and not selected_rows.empty:
        selected_index = int(selected_rows.index[0])
        keep = set(range(max(0, selected_index - 2), min(len(rows), selected_index + 3)))
        keep.add(0)
        rows = rows.loc[sorted(keep)].copy()

    rows = rows.sort_values(["Rank", "Team"], ascending=[False, True]).reset_index(drop=True)
    rows["Label"] = [
        f"{int(rank)}. {charting.wrap_label(team, width=22, max_lines=2)}"
        for team, rank in zip(rows["Team"], rows["Rank"], strict=False)
    ]
    rows["Raw Label"] = [
        _metric_text(metric, value) for value in rows["Value"]
    ]
    selected_mask = rows["Team"].astype(str).eq(str(selected_team))
    leader_mask = rows["Rank"].eq(1) & ~selected_mask
    colours = np.select([selected_mask, leader_mask], [RED, "#344054"], default="#d0d5dd")
    customdata = np.stack([rows["Team"], rows["Raw Label"], rows["Rank"]], axis=-1)

    fig = go.Figure()
    fig.add_vrect(x0=25, x1=75, fillcolor="rgba(102,112,133,0.08)", line_width=0, layer="below")
    fig.add_trace(
        go.Bar(
            x=rows["Percentile"],
            y=rows["Label"],
            orientation="h",
            marker=dict(color=colours, line=dict(color="#ffffff", width=1)),
            text=rows["Raw Label"],
            textposition="outside",
            textfont=dict(color=DARK, size=11),
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Rank: %{customdata[2]}<br>"
                "League percentile: %{x:.1f}<br>Raw value: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=50, line=dict(color="#667085", width=1.4, dash="dash"))
    fig.add_annotation(
        x=50,
        y=1.025,
        xref="x",
        yref="paper",
        text="League midpoint",
        showarrow=False,
        xanchor="left",
        font=dict(size=11, color=GREY),
        bgcolor="rgba(255,255,255,0.88)",
    )
    fig.update_xaxes(
        range=[0, 112],
        tickvals=[0, 25, 50, 75, 100],
        ticktext=["Low", "25", "Midpoint", "75", "High"],
        title="Direction-adjusted league percentile (farther right ranks higher)",
    )
    fig.update_yaxes(title="")
    height = min(880, max(440, len(rows) * 29 + 155))
    fig = charting.polish_figure(fig, f"{metric}: league ranking", height=height)
    fig.update_layout(showlegend=False, margin=dict(l=40, r=96, t=78, b=64), bargap=0.25)
    return fig


def _profile_figure(profile: pd.DataFrame, team_name: str) -> go.Figure:
    if profile.empty:
        return charting.polish_figure(go.Figure(), f"{team_name}: league profile", height=430)
    category_order = {category: index for index, category in enumerate(METRIC_GROUPS)}
    rows = profile.copy()
    rows["_Category Order"] = rows["Category"].map(category_order).fillna(len(category_order))
    rows = rows.sort_values(["_Category Order", "Percentile"], ascending=[True, False]).reset_index(drop=True)
    rows["Label"] = [
        f"{str(category).upper()} · {charting.wrap_label(metric, width=27, max_lines=2)}"
        for category, metric in zip(rows["Category"], rows["Metric"], strict=False)
    ]
    rows["Value Label"] = [
        f"{_metric_text(metric, value)} · #{int(rank)}/{int(teams)}"
        for metric, value, rank, teams in zip(
            rows["Metric"], rows["Value"], rows["Rank"], rows["Teams"], strict=False
        )
    ]
    rows["Colour"] = [
        "#2563eb"
        if metric in STYLE_METRICS
        else GREEN
        if percentile >= 67
        else AMBER
        if percentile >= 33
        else LOW_STANDING
        for metric, percentile in zip(rows["Metric"], rows["Percentile"], strict=False)
    ]
    customdata = np.column_stack(
        [
            rows["Metric"],
            [
                _metric_text(metric, value)
                for metric, value in zip(rows["Metric"], rows["Value"], strict=False)
            ],
            [
                _metric_text(metric, value)
                for metric, value in zip(rows["Metric"], rows["League Average"], strict=False)
            ],
            rows["Rank"],
            rows["Teams"],
            rows["Ranking Direction"],
            rows["Source"],
        ]
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
            textfont=dict(color="#ffffff", size=11),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Percentile: %{x:.1f}<br>Value: %{customdata[1]}<br>"
                "League average: %{customdata[2]}<br>Rank: %{customdata[3]} of %{customdata[4]}<br>"
                "Ranking direction: %{customdata[5]}<br>Source: %{customdata[6]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[103] * len(rows),
            y=rows["Label"],
            mode="text",
            text=rows["Value Label"],
            textposition="middle left",
            textfont=dict(color=DARK, size=10),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_vline(x=50, line=dict(color="#667085", width=1.3, dash="dash"))
    fig.update_layout(barmode="overlay", bargap=0.32)
    fig.update_xaxes(
        range=[0, 122],
        tickvals=[0, 25, 50, 75, 100],
        title="Direction-adjusted league percentile",
    )
    fig.update_yaxes(autorange="reversed", title="")
    fig = charting.polish_figure(fig, f"{team_name}: all-metric league profile", height=max(540, len(rows) * 52 + 150))
    fig.update_layout(margin=dict(l=42, r=110, t=78, b=64))
    return fig


def _head_to_head_rows(
    metrics: pd.DataFrame,
    first_team: str,
    second_team: str,
    specs: list[tuple[str, bool, str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, higher_is_better, source in specs:
        ranked = _league_rank_table(metrics, metric, higher_is_better)
        first = ranked[ranked["Team"].astype(str).eq(str(first_team))]
        second = ranked[ranked["Team"].astype(str).eq(str(second_team))]
        if first.empty or second.empty:
            continue
        first_row = first.iloc[0]
        second_row = second.iloc[0]
        rows.append(
            {
                "Metric": metric,
                "First Team": first_team,
                "First Value": float(first_row["Value"]),
                "First Rank": int(first_row["Rank"]),
                "First Percentile": float(first_row["Percentile"]),
                "Second Team": second_team,
                "Second Value": float(second_row["Value"]),
                "Second Rank": int(second_row["Rank"]),
                "Second Percentile": float(second_row["Percentile"]),
                "Higher Is Better": higher_is_better,
                "Source": source,
            }
        )
    return pd.DataFrame(rows)


def _head_to_head_figure(rows: pd.DataFrame, first_team: str, second_team: str, title: str) -> go.Figure:
    if rows.empty:
        return charting.polish_figure(go.Figure(), title, height=430)
    rows = rows.sort_values("First Percentile", ascending=False).reset_index(drop=True)
    labels = rows["Metric"].map(lambda value: charting.wrap_label(value, width=25, max_lines=2))
    line_x: list[object] = []
    line_y: list[object] = []
    for label, first, second in zip(labels, rows["First Percentile"], rows["Second Percentile"], strict=False):
        line_x.extend([first, second, None])
        line_y.extend([label, label, None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line=dict(color="#cbd3dd", width=3),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    first_custom = np.column_stack(
        [
            [
                _metric_text(metric, value)
                for metric, value in zip(rows["Metric"], rows["First Value"], strict=False)
            ],
            rows["First Rank"],
            rows["Metric"],
        ]
    )
    second_custom = np.column_stack(
        [
            [
                _metric_text(metric, value)
                for metric, value in zip(rows["Metric"], rows["Second Value"], strict=False)
            ],
            rows["Second Rank"],
            rows["Metric"],
        ]
    )
    fig.add_trace(
        go.Scatter(
            x=rows["First Percentile"],
            y=labels,
            mode="markers+text",
            name=first_team,
            marker=dict(size=17, color=RED, line=dict(color="#ffffff", width=2)),
            text=[f"{value:.0f}" for value in rows["First Percentile"]],
            textposition="top center",
            textfont=dict(size=10, color=RED),
            customdata=first_custom,
            cliponaxis=False,
            hovertemplate=(
                f"<b>{first_team}</b><br>%{{customdata[2]}}<br>Percentile: %{{x:.1f}}<br>"
                "Value: %{customdata[0]}<br>Rank: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rows["Second Percentile"],
            y=labels,
            mode="markers+text",
            name=second_team,
            marker=dict(size=15, color="#344054", line=dict(color="#ffffff", width=2)),
            text=[f"{value:.0f}" for value in rows["Second Percentile"]],
            textposition="bottom center",
            textfont=dict(size=10, color="#344054"),
            customdata=second_custom,
            cliponaxis=False,
            hovertemplate=(
                f"<b>{second_team}</b><br>%{{customdata[2]}}<br>Percentile: %{{x:.1f}}<br>"
                "Value: %{customdata[0]}<br>Rank: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=50, line=dict(color="#98a2b3", width=1.2, dash="dash"))
    fig.update_xaxes(range=[-3, 107], dtick=25, title="Direction-adjusted league percentile")
    fig.update_yaxes(title="", autorange="reversed")
    fig = charting.polish_figure(fig, title, height=max(430, len(rows) * 82 + 170))
    fig.update_layout(legend=dict(orientation="h", y=1.02, x=0), margin=dict(l=40, r=35, t=88, b=62))
    return fig


def _matrix_rows(
    metrics: pd.DataFrame,
    groups: dict[str, list[tuple[str, bool, str]]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category, specs in groups.items():
        for metric, higher_is_better, source in specs:
            ranked = _league_rank_table(metrics, metric, higher_is_better)
            for _, row in ranked.iterrows():
                rows.append(
                    {
                        "Category": category,
                        "Metric": metric,
                        "Team": str(row["Team"]),
                        "Value": float(row["Value"]),
                        "Rank": int(row["Rank"]),
                        "Percentile": float(row["Percentile"]),
                        "Ranking Direction": "Higher ranks higher" if higher_is_better else "Lower ranks higher",
                        "Source": source,
                    }
                )
    return pd.DataFrame(rows)


def _league_matrix_figure(
    rows: pd.DataFrame,
    selected_team: str,
    display_mode: str,
    title: str,
    ordered_teams: list[str],
) -> go.Figure:
    if rows.empty:
        return charting.polish_figure(go.Figure(), title, height=430)
    metric_order = rows[["Category", "Metric"]].drop_duplicates()["Metric"].tolist()
    available_teams = set(rows["Team"].astype(str))
    team_order = [str(team) for team in ordered_teams if str(team) in available_teams]
    team_order.extend(sorted(available_teams.difference(team_order)))

    percentile = rows.pivot(index="Team", columns="Metric", values="Percentile").reindex(
        index=team_order, columns=metric_order
    )
    ranks = rows.pivot(index="Team", columns="Metric", values="Rank").reindex(index=team_order, columns=metric_order)
    values = rows.pivot(index="Team", columns="Metric", values="Value").reindex(index=team_order, columns=metric_order)
    text_values: list[list[str]] = []
    for team in team_order:
        row_text: list[str] = []
        for metric in metric_order:
            if display_mode == "Rank":
                rank_value = ranks.loc[team, metric]
                row_text.append(f"#{int(rank_value)}" if pd.notna(rank_value) else "")
            elif display_mode == "Raw value":
                row_text.append(_metric_text(metric, values.loc[team, metric]))
            else:
                pct_value = percentile.loc[team, metric]
                row_text.append(f"{float(pct_value):.0f}" if pd.notna(pct_value) else "")
        text_values.append(row_text)

    customdata = np.empty((len(team_order), len(metric_order), 3), dtype=object)
    for team_index, team in enumerate(team_order):
        for metric_index, metric in enumerate(metric_order):
            customdata[team_index, metric_index] = [
                _metric_text(metric, values.loc[team, metric]),
                ranks.loc[team, metric],
                percentile.loc[team, metric],
            ]
    labels = [f"★ {team}" if str(team) == str(selected_team) else str(team) for team in team_order]
    # Favourability-based colorscale: Red (Unfavourable) -> White (Median) -> Green (Favourable)
    # Using slightly muted tones to ensure text readability
    colourscale = [
        [0.0, "#e45756"],   # Unfavourable (Muted Red)
        [0.5, "#ffffff"],   # Median (White)
        [1.0, "#1c8c44"],   # Favourable (Muted Green)
    ]
    fig = go.Figure(
        go.Heatmap(
            z=percentile.to_numpy(),
            x=[charting.wrap_label(metric, width=15, max_lines=3) for metric in metric_order],
            y=labels,
            zmin=0,
            zmax=100,
            colorscale=colourscale,
            text=text_values,
            texttemplate="%{text}",
            textfont=dict(size=10, color=DARK),
            customdata=customdata,
            colorbar=dict(
                title="League Rank<br>Standing",
                tickvals=[0, 50, 100],
                ticktext=["Unfavourable", "Median", "Favourable"],
                len=0.75,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>%{x}<br>Value: %{customdata[0]}<br>Rank: %{customdata[1]}<br>"
                "Percentile: %{customdata[2]:.1f}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(side="top", title="")
    fig.update_yaxes(autorange="reversed", title="")
    fig = charting.polish_figure(fig, title, height=min(900, max(560, len(team_order) * 29 + 180)))
    fig.update_layout(margin=dict(l=45, r=40, t=130, b=45))
    return fig


_inject_rankings_css()

ta.page_header(
    "League Rankings",
    "Understand league position at a glance, explore any metric across all clubs, and compare two teams without crowding every view onto one screen.",
    PAGE_SOURCE,
    (
        "Style-based measures appear only where squad-iteration KPI data exists. Possession and progression describe a "
        "team's league standing for that style indicator; a higher value is not automatically better football."
    ),
)

season_inventory = data.list_seasons()
match_seasons = set(season_inventory.get("matches", []))
style_seasons = set(season_inventory.get("players", []))
season_options = sorted(match_seasons | style_seasons)
if not season_options:
    st.warning("No team or match seasons are available for league rankings.")
    st.stop()

ta.section_heading("Comparison Setup")
with st.container(border=True):
    st.caption("Choose the season and focus team once. Metric-specific controls sit inside the relevant comparison view.")
    setup_columns = st.columns([0.8, 1.35])
    with setup_columns[0]:
        season = st.selectbox(
            "Season",
            season_options,
            index=len(season_options) - 1,
            key="league_rankings_season",
        )

    with st.spinner("Building league comparison..."):
        match_rows = ma.load_matches(season) if season in match_seasons else pd.DataFrame()
        regular_fixtures = ta.regular_season_fixtures(match_rows)
        match_metrics = _match_metric_table(regular_fixtures)
        style_metrics = ta.load_team_style_data(season) if season in style_seasons else pd.DataFrame()
        teams = _combine_metric_sources(match_metrics, style_metrics)

    if teams.empty:
        st.warning("No league-wide team metrics are available for this season.")
        st.stop()

    with setup_columns[1]:
        team_name = ta.team_selector(teams, key="league_rankings_team", label="Focus team")

available_groups = _available_metric_groups(teams)
profile = _team_metric_profile(teams, team_name)
if profile.empty:
    st.warning("No league profile is available for the selected team.")
    st.stop()

if style_metrics.empty:
    st.caption(
        "Possession and progression views are unavailable for this season; results, attacking and defending comparisons remain available."
    )

ta.section_heading("League Profile at a Glance")
_profile_overview_cards(profile, team_name)
_callout("Profile diagnosis", _profile_story(profile, team_name))

ranking_tab, profile_tab, head_to_head_tab, matrix_tab = st.tabs(
    ["Metric Ranking", "Team Profile", "Head-to-Head", "League Matrix"]
)

with ranking_tab:
    ta.section_heading("Metric Explorer")
    st.caption("Choose one statistic, then read every club from the same direction-adjusted scale.")
    ranking_controls = st.columns([1.0, 1.45])
    with ranking_controls[0]:
        metric_group = st.selectbox(
            "Metric group",
            list(available_groups),
            key="league_rankings_metric_group",
        )
    metric_specs = available_groups[metric_group]
    metric_options = [metric_name for metric_name, _, _ in metric_specs]
    with ranking_controls[1]:
        metric = st.selectbox(
            "Metric",
            metric_options,
            key="league_rankings_metric",
        )
    ranking_scope = st.radio(
        "Teams shown",
        ["Full league", "Nearest ranks"],
        horizontal=True,
        key="league_rankings_scope",
        help="Nearest ranks keeps the league leader and the two ranks either side of the focus team.",
    )

    metric_lookup = {
        metric_name: (higher_is_better, source)
        for metric_name, higher_is_better, source in metric_specs
    }
    higher_is_better, metric_source = metric_lookup[metric]
    ranked = _league_rank_table(teams, metric, higher_is_better)
    selected_rows = ranked[ranked["Team"].astype(str).eq(str(team_name))]
    if selected_rows.empty:
        st.warning("The selected team has no value for this metric.")
    else:
        selected = selected_rows.iloc[0]
        _metric_insight_strip(ranked, selected, metric, team_name, higher_is_better)
        st.plotly_chart(
            _ranking_figure(ranked, metric, team_name, ranking_scope),
            width="stretch",
            key="league_rankings_primary_chart",
        )
        direction_copy = (
            "Higher values rank higher; this is a style indicator rather than an automatic quality judgement."
            if metric in STYLE_METRICS
            else "Higher is better."
            if higher_is_better
            else "Lower is better."
        )
        st.caption(
            f"Farther right always means a higher league rank after direction is applied. Red identifies the focus team, "
            f"{team_name}; charcoal identifies the leader and the pale band is the middle 50% of league standings. "
            f"{direction_copy} Source: {metric_source}."
        )

        with st.expander("Open full ranking table and download"):
            context = ranked.copy()
            context["Focus"] = ["Focus team" if str(team) == str(team_name) else "" for team in context["Team"]]
            context["Relative to Focus"] = [
                _relative_text(metric, value, float(selected["Value"]), higher_is_better)
                for value in context["Value"]
            ]
            context["Value"] = context["Value"].map(lambda value: _metric_text(metric, value))
            context = context[["Focus", "Rank", "Team", "Value", "Percentile", "Relative to Focus"]]
            st.dataframe(
                context,
                width="stretch",
                hide_index=True,
                column_config={
                    "Percentile": st.column_config.ProgressColumn(
                        "Percentile score",
                        help="Direction-adjusted league standing; a larger score always ranks higher.",
                        min_value=0,
                        max_value=100,
                        format="%.0f",
                    ),
                },
            )
            download_table = ranked.rename(columns={"Value": metric}).copy()
            download_table["Ranking Direction"] = (
                "Higher ranks higher" if higher_is_better else "Lower ranks higher"
            )
            download_table["Source"] = metric_source
            st.download_button(
                "Download selected metric CSV",
                download_table.to_csv(index=False),
                file_name=(
                    f"league_rankings_{season.replace('/', '-')}_"
                    f"{re.sub(r'[^a-z0-9]+', '_', metric.lower()).strip('_')}.csv"
                ),
                mime="text/csv",
                key="league_rankings_metric_download",
            )

with profile_tab:
    ta.section_heading("All-Metric Team Profile")
    st.plotly_chart(
        _profile_figure(profile, team_name),
        width="stretch",
        key="league_rankings_profile_chart",
    )
    st.caption(
        "Every bar is direction-adjusted, so farther right means a higher league standing. Green, amber and slate show "
        "top, middle and bottom thirds for outcome measures. Blue marks style indicators, which are descriptive rather "
        "than automatically good or bad. Raw value and rank are printed at the right of each row."
    )
    with st.expander("Open the all-metric profile table"):
        st.dataframe(
            _ranking_profile(teams, team_name),
            width="stretch",
            hide_index=True,
            column_config={
                "Percentile": st.column_config.ProgressColumn(
                    "Percentile score",
                    min_value=0,
                    max_value=100,
                    format="%.0f",
                ),
            },
        )

with head_to_head_tab:
    ta.section_heading("Team-to-Team Comparison")
    comparison_options = sorted(
        team for team in teams["Team"].dropna().astype(str).unique().tolist() if str(team) != str(team_name)
    )
    if not comparison_options:
        st.info("No second team is available for comparison.")
    else:
        points_ranking = _league_rank_table(teams, "Points / Match", True)
        suggested = next(
            (str(team) for team in points_ranking["Team"] if str(team) != str(team_name)),
            comparison_options[0],
        )
        head_controls = st.columns([1.35, 1.0])
        with head_controls[0]:
            comparison_team = st.selectbox(
                "Comparison team",
                comparison_options,
                index=comparison_options.index(suggested) if suggested in comparison_options else 0,
                key="league_rankings_comparison_team",
            )
        with head_controls[1]:
            head_group = st.selectbox(
                "Metric group",
                list(available_groups),
                key="league_rankings_head_group",
            )
        head_rows = _head_to_head_rows(teams, team_name, comparison_team, available_groups[head_group])
        first_higher = int(head_rows["First Percentile"].gt(head_rows["Second Percentile"]).sum()) if not head_rows.empty else 0
        second_higher = int(head_rows["Second Percentile"].gt(head_rows["First Percentile"]).sum()) if not head_rows.empty else 0
        ties = len(head_rows) - first_higher - second_higher
        _callout(
            "Head-to-head read",
            f"{team_name} ranks higher on {first_higher} of {len(head_rows)} {head_group.lower()} measures; "
            f"{comparison_team} ranks higher on {second_higher}, with {ties} tied.",
        )
        st.plotly_chart(
            _head_to_head_figure(
                head_rows,
                team_name,
                comparison_team,
                f"{team_name} vs {comparison_team}: {head_group}",
            ),
            width="stretch",
            key="league_rankings_head_chart",
        )
        st.caption(
            "Dots use direction-adjusted league percentiles, allowing unlike raw units to be compared on one scale. "
            "Hover for each team's raw value and league rank."
        )

with matrix_tab:
    ta.section_heading("Whole-League Matrix")
    matrix_controls = st.columns([1.05, 1.25, 1.0])
    with matrix_controls[0]:
        matrix_group = st.selectbox(
            "Metrics shown",
            [*list(available_groups), "All metrics"],
            key="league_rankings_matrix_group",
        )
    all_metric_specs = [spec for specs in available_groups.values() for spec in specs]
    all_metric_names = list(dict.fromkeys(metric_name for metric_name, _, _ in all_metric_specs))
    default_order_metric = "Points / Match" if "Points / Match" in all_metric_names else all_metric_names[0]
    order_options = [default_order_metric, *[name for name in all_metric_names if name != default_order_metric]]
    with matrix_controls[1]:
        matrix_order_metric = st.selectbox(
            "Order teams by metric",
            order_options,
            key="league_rankings_matrix_order",
        )
    with matrix_controls[2]:
        matrix_display = st.radio(
            "Cell labels",
            ["Percentile", "Rank", "Raw value"],
            horizontal=True,
            key="league_rankings_matrix_display",
        )
    matrix_groups = available_groups if matrix_group == "All metrics" else {matrix_group: available_groups[matrix_group]}
    matrix_rows = _matrix_rows(teams, matrix_groups)
    order_lookup = {
        metric_name: higher_is_better
        for metric_name, higher_is_better, _ in all_metric_specs
    }
    matrix_order = _league_rank_table(
        teams,
        matrix_order_metric,
        order_lookup[matrix_order_metric],
    )["Team"].astype(str).tolist()
    st.plotly_chart(
        _league_matrix_figure(
            matrix_rows,
            team_name,
            matrix_display,
            f"{season}: {matrix_group.lower()} landscape · ordered by {matrix_order_metric}",
            matrix_order,
        ),
        width="stretch",
        key="league_rankings_matrix_chart",
    )
    st.caption(
        "Green reflects a favourable league standing (closer to rank 1), while red indicates an unfavourable position. "
        "The star marks the focus team, and row order follows the metric chosen above. Cell text can switch between "
        "percentile, rank and raw value."
    )
    with st.expander("Open matrix data and download"):
        st.dataframe(matrix_rows, width="stretch", hide_index=True)
        st.download_button(
            "Download league matrix CSV",
            matrix_rows.to_csv(index=False),
            file_name=f"league_matrix_{season.replace('/', '-')}_{matrix_group.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            key="league_rankings_matrix_download",
        )

with st.expander("Terminology and data scope"):
    st.markdown(
        """
        - **Rank:** league position for one metric. Rank 1 is the highest standing after the configured direction is applied.
        - **Percentile score:** a 4–100 direction-adjusted score derived from league rank for a 24-team league. Rank 1 scores 100, and tied teams share the same score.
        - **Outcome measure:** a statistic such as points, goals, losses or clean sheets where the preferred direction is explicit.
        - **Style indicator:** possession and progression volume show how a team plays or where it sits in the distribution; higher is not automatically better.
        - **League average:** the mean across clubs with a valid value for the statistic.
        - **Regular season:** only the first two completed meetings between each opponent pair are retained, excluding postseason rematches.
        - **Match results source:** completed match rows power results, scoring and defending rates.
        - **Team style KPIs:** provider-authored Impect squad-iteration facts supply possession, progression and additional output measures.
        """
    )
