# =============================================================================
# SHOOTING DASHBOARD - player shooting metrics and mapped shot analysis
# =============================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charting, data, match_analysis as ma, pitch, player_analysis as pa, ui


SHOOTING_BASIS = (
    "Season aggregate shooting metrics come from CAFC_DB Impect player-iteration KPI facts. "
    "Selected-fixture visuals use provider shot events with adjusted shot coordinates, "
    "SHOT_XG, POSTSHOT_XG, shot distance, angle, body part and target coordinates where available."
)

GOAL_HALF_WIDTH = pitch.GOAL_HALF_WIDTH
GOAL_HEIGHT = pitch.GOAL_HEIGHT

OUTCOME_ORDER = ["Goal", "On Target / Saved", "Blocked", "Woodwork", "Off Target", "Other Shot"]
OUTCOME_COLORS = {
    "Goal": ui.CHARLTON_RED,
    "On Target / Saved": "#15803d",
    "Blocked": "#f59e0b",
    "Woodwork": "#9333ea",
    "Off Target": "#667085",
    "Other Shot": "#344054",
}
OUTCOME_SYMBOLS = {
    "Goal": "star",
    "On Target / Saved": "circle",
    "Blocked": "square",
    "Woodwork": "diamond",
    "Off Target": "x",
    "Other Shot": "diamond",
}
SHOT_TYPE_ORDER = ["Header", "Penalty Kick", "Free Kick", "Long Range Shot", "Mid Range Shot", "Other Shot"]
SHOT_TYPE_SYMBOLS = {
    "Header": "triangle-up",
    "Penalty Kick": "star",
    "Free Kick": "hexagon",
    "Long Range Shot": "diamond",
    "Mid Range Shot": "circle",
    "Other Shot": "square",
}
SHOT_DISTANCE_ZONES = [
    {"min": 0.0, "max": 11.0, "label": "Close Range", "range_label": "0-11m", "color": "rgba(21,128,61,0.16)", "line": "#15803d"},
    {"min": 11.0, "max": 18.0, "label": "Box / Good Range", "range_label": "11-18m", "color": "rgba(245,158,11,0.16)", "line": "#f59e0b"},
    {"min": 18.0, "max": 25.0, "label": "Edge Range", "range_label": "18-25m", "color": "rgba(251,191,36,0.14)", "line": "#d97706"},
    {"min": 25.0, "max": None, "label": "Long Range", "range_label": "25m+", "color": "rgba(220,38,38,0.12)", "line": "#dc2626"},
]
PERIOD_BASE_MINUTES = {1: 0.0, 2: 45.0, 3: 90.0, 4: 105.0}
ADDED_TIME_START_MINUTE = 90.0
MAX_TIMELINE_MINUTE = 110.0


def _numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _last_name_sort(value: object) -> tuple[str, str]:
    text = "" if value is None else str(value)
    parts = text.strip().split()
    return (parts[-1].casefold() if parts else text.casefold(), text.casefold())


def _title_text(value: object, fallback: str = "Unknown") -> str:
    text = fallback if value is None or str(value).lower() == "nan" else str(value)
    return text.replace("_", " ").replace("-", " ").title()


def _shot_type_symbol(value: object) -> str:
    text = "" if value is None else str(value).upper()
    if "HEADER" in text:
        return SHOT_TYPE_SYMBOLS["Header"]
    if "PENALTY" in text:
        return SHOT_TYPE_SYMBOLS["Penalty Kick"]
    if "FREE KICK" in text:
        return SHOT_TYPE_SYMBOLS["Free Kick"]
    if "LONG RANGE" in text:
        return SHOT_TYPE_SYMBOLS["Long Range Shot"]
    if "MID RANGE" in text:
        return SHOT_TYPE_SYMBOLS["Mid Range Shot"]
    return SHOT_TYPE_SYMBOLS["Other Shot"]


def _is_goal(row: pd.Series) -> bool:
    action = str(row.get("Action", "")).upper()
    result = str(row.get("Result", "")).upper()
    return result == "SUCCESS" or action in {"GOAL", "SHOT_GOAL"}


def _shot_outcome(row: pd.Series) -> str:
    action = str(row.get("Action", "")).upper()
    result = str(row.get("Result", "")).upper()
    text = f"{action} {result}"
    if _is_goal(row):
        return "Goal"
    if "BLOCK" in text:
        return "Blocked"
    if any(token in text for token in ["WOODWORK", "POST", "BAR"]):
        return "Woodwork"
    if any(token in text for token in ["SAVE", "SAVED", "ON_TARGET", "ON TARGET", "SHOT_AT_GOAL"]):
        return "On Target / Saved"
    target_y = pd.to_numeric(pd.Series([row.get("Shot Target Y")]), errors="coerce").iloc[0]
    target_z = pd.to_numeric(pd.Series([row.get("Shot Target Z")]), errors="coerce").iloc[0]
    if pd.notna(target_y) and pd.notna(target_z) and -GOAL_HALF_WIDTH <= target_y <= GOAL_HALF_WIDTH and 0 <= target_z <= GOAL_HEIGHT:
        return "On Target / Saved"
    if any(token in text for token in ["OFF_TARGET", "OFF TARGET", "WIDE"]):
        return "Off Target"
    if result == "FAIL":
        return "Off Target"
    return "Other Shot"


def _quality_band(value: object) -> str:
    xg = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(xg):
        return "Unknown"
    if xg >= 0.30:
        return "Very High xG"
    if xg >= 0.15:
        return "High xG"
    if xg >= 0.07:
        return "Medium xG"
    return "Low xG"


def _distance_band(value: object) -> str:
    distance = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(distance):
        return "Unknown"
    if distance <= 11:
        return "Inside 11m"
    if distance <= 18:
        return "11-18m"
    if distance <= 25:
        return "18-25m"
    return "25m+"


def _prepare_shots(shots: pd.DataFrame) -> pd.DataFrame:
    if shots.empty:
        return shots.copy()
    out = shots.copy()
    for col in ["Shot xG", "Post-Shot xG", "Packing xG", "PXT Shot", "Shot Distance", "Shot Angle", "Minute", "Second", "Shot Target Y", "Shot Target Z"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["Shot xG"] = out["Shot xG"].fillna(0)
    out["Outcome"] = out.apply(_shot_outcome, axis=1)
    out["Shot Type"] = out["Action"].apply(lambda value: _title_text(value, "Shot")) if "Action" in out else "Shot"
    out["Body Part Display"] = out["Body Part"].apply(lambda value: _title_text(value, "Unknown")) if "Body Part" in out else "Unknown"
    out["xG Band"] = out["Shot xG"].apply(_quality_band)
    out["Distance Band"] = out["Shot Distance"].apply(_distance_band) if "Shot Distance" in out else "Unknown"
    out["Goal"] = out["Outcome"].eq("Goal")
    out["On Target"] = out["Outcome"].isin(["Goal", "On Target / Saved"])
    out["Post-Shot xG Added"] = out["Post-Shot xG"] - out["Shot xG"] if "Post-Shot xG" in out else np.nan
    return out


def _shot_summary(shots: pd.DataFrame) -> dict[str, float]:
    if shots.empty:
        return {
            "Shots": 0,
            "Goals": 0,
            "xG": 0.0,
            "Post-Shot xG": 0.0,
            "Avg xG": 0.0,
            "Conversion %": 0.0,
            "On Target %": 0.0,
            "Avg Distance": 0.0,
        }
    shots_count = len(shots)
    goals = int(shots["Goal"].sum()) if "Goal" in shots else 0
    xg = float(_numeric(shots, "Shot xG").fillna(0).sum())
    psxg = float(_numeric(shots, "Post-Shot xG").fillna(0).sum()) if "Post-Shot xG" in shots else 0.0
    return {
        "Shots": shots_count,
        "Goals": goals,
        "xG": xg,
        "Post-Shot xG": psxg,
        "Avg xG": xg / shots_count if shots_count else 0.0,
        "Conversion %": goals / shots_count * 100 if shots_count else 0.0,
        "On Target %": float(shots["On Target"].sum()) / shots_count * 100 if "On Target" in shots and shots_count else 0.0,
        "Avg Distance": float(_numeric(shots, "Shot Distance").mean()) if "Shot Distance" in shots else 0.0,
    }


def _empty_pitch_message(fig: go.Figure, message: str) -> go.Figure:
    fig.add_annotation(
        text=message,
        x=27,
        y=0,
        xref="x",
        yref="y",
        showarrow=False,
        font=dict(size=16, color=ui.CHARLTON_MUTED),
        bgcolor="rgba(255,255,255,0.78)",
        bordercolor=ui.CHARLTON_BORDER,
        borderpad=8,
    )
    return fig


def _attacking_pitch(title: str, height: int = 650) -> go.Figure:
    fig = pitch.pitch_image_figure(title, height=height)
    fig.update_xaxes(range=[-3, pitch.PITCH_X_MAX + 2])
    fig.update_yaxes(range=[pitch.PITCH_Y_MIN - 2, pitch.PITCH_Y_MAX + 2])
    return fig


def _shot_map(
    shots: pd.DataFrame,
    title: str,
    show_labels: bool = True,
    context_shots: pd.DataFrame | None = None,
    selected_player: str | None = None,
    height: int = 650,
) -> go.Figure:
    fig = _attacking_pitch(title, height=height)

    if context_shots is not None and not context_shots.empty:
        context = context_shots.dropna(subset=["Start X", "Start Y"]).copy()
        if selected_player:
            context = context[context["Player"].astype(str) != str(selected_player)]
        if not context.empty:
            fig.add_trace(
                go.Scatter(
                    x=context["Start X"],
                    y=context["Start Y"],
                    mode="markers",
                    name="Team shot context",
                    showlegend=False,
                    marker=dict(size=8, color="rgba(102,112,133,0.28)", line=dict(color="#ffffff", width=0.8)),
                    customdata=np.stack(
                        [
                            context["Player"].fillna("Unknown"),
                            context["Outcome"].fillna("Unknown"),
                            context["Shot xG"].fillna(0),
                            context["Minute"].fillna(0),
                        ],
                        axis=-1,
                    ),
                    hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>Minute: %{customdata[3]:.0f}<br>xG: %{customdata[2]:.3f}<extra></extra>",
                )
            )

    plotted = shots.dropna(subset=["Start X", "Start Y"]).copy()
    if plotted.empty:
        return _empty_pitch_message(fig, "No mapped shot locations")

    plotted["_Shot Type Key"] = plotted["Shot Type"].fillna("Other Shot").astype(str)
    max_xg = max(float(plotted["Shot xG"].max()), 0.08)
    for outcome in OUTCOME_ORDER:
        group = plotted[plotted["Outcome"] == outcome].copy()
        if group.empty:
            continue
        group["_Label"] = ""
        if show_labels:
            group["_Label"] = np.where(
                group["Outcome"].eq("Goal") | (group["Shot xG"].fillna(0) >= 0.18),
                group["Player"].apply(lambda value: charting.wrap_label(value, 12, 2)),
                "",
            )
        customdata = np.stack(
            [
                group["Player"].fillna("Unknown"),
                group["Minute"].fillna(0),
                group["Shot Type"].fillna("Shot"),
                group["Body Part Display"].fillna("Unknown"),
                group["Shot xG"].fillna(0),
                group["Post-Shot xG"].fillna(0),
                group["Shot Distance"].fillna(0),
                group["Shot Angle"].fillna(0),
                group["Outcome"].fillna("Unknown"),
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=group["Start X"],
                y=group["Start Y"],
                mode="markers+text",
                name=outcome,
                text=group["_Label"],
                textposition="top center",
                textfont=dict(size=10, color=ui.CHARLTON_BLACK),
                marker=dict(
                    size=10 + (group["Shot xG"] / max_xg) * 26,
                    color=OUTCOME_COLORS.get(outcome, "#344054"),
                    symbol=group["_Shot Type Key"].apply(_shot_type_symbol),
                    opacity=0.9,
                    line=dict(color="#ffffff", width=1.2),
                ),
                showlegend=False,
                customdata=customdata,
                hovertemplate=(
                    "%{customdata[0]} - %{customdata[8]}"
                    "<br>Minute: %{customdata[1]:.0f}"
                    "<br>Type: %{customdata[2]}"
                    "<br>Body part: %{customdata[3]}"
                    "<br>xG: %{customdata[4]:.3f}"
                    "<br>Post-shot xG: %{customdata[5]:.3f}"
                    "<br>Distance: %{customdata[6]:.1f}m"
                    "<br>Angle: %{customdata[7]:.1f}<extra></extra>"
                ),
            )
        )

    for outcome in OUTCOME_ORDER:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"Outcome: {outcome}",
                legendgroup="outcome-key",
                marker=dict(
                    size=11,
                    color=OUTCOME_COLORS.get(outcome, "#344054"),
                    symbol="circle",
                    line=dict(color="#ffffff", width=1.2),
                ),
                hoverinfo="skip",
            )
        )

    present_types = plotted["_Shot Type Key"].dropna().astype(str).unique().tolist()
    ordered_types = [shot_type for shot_type in SHOT_TYPE_ORDER if shot_type in present_types]
    ordered_types.extend(sorted(shot_type for shot_type in present_types if shot_type not in ordered_types))
    for shot_type in ordered_types:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"Type: {shot_type}",
                legendgroup="shot-type-key",
                marker=dict(
                    size=11,
                    color="#667085",
                    symbol=_shot_type_symbol(shot_type),
                    line=dict(color="#ffffff", width=1.2),
                ),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        legend=dict(
            orientation="h",
            title_text="<b>Key: colour = outcome · shape = shot type</b>",
            yanchor="top",
            y=-0.12,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
        margin=dict(l=18, r=18, t=74, b=142),
    )
    return fig


def _shot_density_grid(
    mapped: pd.DataFrame,
    value_mode: str,
    grid_x: int = 110,
    grid_y: int = 90,
    sigma: float = 4.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_grid = np.linspace(0, pitch.PITCH_X_MAX, grid_x)
    y_grid = np.linspace(pitch.PITCH_Y_MIN, pitch.PITCH_Y_MAX, grid_y)
    z = np.zeros((len(y_grid), len(x_grid)), dtype="float64")

    x_values = pd.to_numeric(mapped["Start X"], errors="coerce")
    y_values = pd.to_numeric(mapped["Start Y"], errors="coerce")
    if value_mode == "xG":
        weights = pd.to_numeric(mapped["Shot xG"], errors="coerce").fillna(0.0).clip(lower=0.004)
    else:
        weights = pd.Series(1.0, index=mapped.index)

    for shot_x, shot_y, weight in zip(x_values, y_values, weights):
        if pd.isna(shot_x) or pd.isna(shot_y) or pd.isna(weight):
            continue
        x_kernel = np.exp(-0.5 * ((x_grid - float(shot_x)) / sigma) ** 2)
        y_kernel = np.exp(-0.5 * ((y_grid - float(shot_y)) / sigma) ** 2)
        z += float(weight) * np.outer(y_kernel, x_kernel)

    max_value = float(np.nanmax(z)) if z.size else 0.0
    if max_value > 0:
        z[z < max_value * 0.018] = np.nan
    return x_grid, y_grid, z


def _shot_zone_heatmap(shots: pd.DataFrame, title: str, value_mode: str = "Shot Count", height: int = 650) -> go.Figure:
    fig = _attacking_pitch(title, height=height)
    mapped = shots.dropna(subset=["Start X", "Start Y"]).copy()
    if mapped.empty:
        return _empty_pitch_message(fig, "No mapped shot locations")

    x_grid, y_grid, z = _shot_density_grid(mapped, value_mode=value_mode)
    max_value = float(np.nanmax(z)) if z.size else 0.0

    fig.add_trace(
        go.Heatmap(
            x=x_grid,
            y=y_grid,
            z=z,
            colorscale=[
                [0.0, "rgba(255,255,255,0.0)"],
                [0.18, "rgba(21,128,61,0.22)"],
                [0.50, "rgba(245,158,11,0.58)"],
                [1.0, "rgba(220,38,38,0.86)"],
            ],
            zmin=0,
            zmax=max(max_value, 1.0 if value_mode == "Shot Count" else 0.05),
            zsmooth="best",
            opacity=0.94,
            colorbar=dict(title=f"<b>{value_mode}<br>Density</b>"),
            hovertemplate=f"Smoothed {value_mode.lower()} density: %{{z:.2f}}<extra></extra>",
            showscale=True,
        )
    )
    return fig


def _empty_chart(title: str, message: str, height: int = 460) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    return charting.polish_figure(fig, title, height=height)


def _outcome_chart(shots: pd.DataFrame, title: str) -> go.Figure:
    if shots.empty:
        return _empty_chart(title, "No Shot Outcomes")
    summary = shots.groupby("Outcome", as_index=False).agg(
        Shots=("Outcome", "size"),
        Goals=("Goal", "sum"),
        xG=("Shot xG", "sum"),
        **{"Post-Shot xG": ("Post-Shot xG", "sum")},
    )
    summary["Outcome"] = pd.Categorical(summary["Outcome"], OUTCOME_ORDER, ordered=True)
    summary = summary.set_index("Outcome").reindex(OUTCOME_ORDER, fill_value=0).reset_index()
    summary["Avg xG"] = summary["xG"] / summary["Shots"].replace(0, np.nan)
    plot_df = summary[summary["Shots"] > 0].copy()
    if plot_df.empty:
        return _empty_chart(title, "No Shot Outcomes")

    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=plot_df["Outcome"].astype(str),
            values=plot_df["Shots"],
            hole=0.58,
            sort=False,
            direction="clockwise",
            marker=dict(colors=[OUTCOME_COLORS.get(str(outcome), "#344054") for outcome in plot_df["Outcome"].astype(str)], line=dict(color="#ffffff", width=2)),
            textinfo="label+percent",
            textposition="inside",
            insidetextorientation="radial",
            customdata=np.stack([plot_df["Goals"], plot_df["xG"], plot_df["Post-Shot xG"], plot_df["Avg xG"].fillna(0)], axis=-1),
            hovertemplate="%{label}<br>Shots: %{value:.0f}<br>Share: %{percent}<br>Goals: %{customdata[0]:.0f}<br>xG: %{customdata[1]:.2f}<br>Post-shot xG: %{customdata[2]:.2f}<br>Avg xG: %{customdata[3]:.3f}<extra></extra>",
        )
    )
    total_shots = int(plot_df["Shots"].sum())
    total_xg = float(plot_df["xG"].sum())
    fig.add_annotation(
        text=f"<b>{total_shots}</b><br>Shots<br><span style='font-size:12px'>{total_xg:.2f} xG</span>",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color=ui.CHARLTON_BLACK),
    )
    fig.update_layout(height=470, showlegend=True)
    fig = charting.polish_figure(fig, title)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5))
    return fig


def _breakdown_chart(shots: pd.DataFrame, group_col: str, title: str) -> go.Figure:
    if shots.empty or group_col not in shots:
        return charting.polish_figure(go.Figure(), title, height=430)
    summary = shots.groupby(group_col, as_index=False).agg(
        Shots=(group_col, "size"),
        Goals=("Goal", "sum"),
        xG=("Shot xG", "sum"),
    )
    summary = summary.sort_values("Shots", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=summary["Shots"],
            y=summary[group_col].astype(str).apply(lambda value: charting.wrap_label(value, 18, 2)),
            orientation="h",
            marker_color=ui.CHARLTON_RED,
            text=[charting.metric_text(value, "Actions") for value in summary["Shots"]],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([summary[group_col], summary["Goals"], summary["xG"]], axis=-1),
            hovertemplate="%{customdata[0]}<br>Shots: %{x:.0f}<br>Goals: %{customdata[1]:.0f}<br>xG: %{customdata[2]:.2f}<extra></extra>",
        )
    )
    fig.update_layout(height=charting.horizontal_bar_height(len(summary), min_height=430), xaxis_title="Shots", yaxis_title="", showlegend=False)
    fig.update_xaxes(tickformat=".0f")
    return charting.polish_figure(fig, title)


def _shot_mix_treemap(shots: pd.DataFrame, title: str) -> go.Figure:
    required = {"Outcome", "Shot Type", "Body Part Display"}
    if shots.empty or not required.issubset(shots.columns):
        return _empty_chart(title, "No Shot Mix Data", height=520)

    plot_df = shots.copy()
    for col in required:
        plot_df[col] = plot_df[col].fillna("Unknown").astype(str)

    labels = ["All Shots"]
    ids = ["all"]
    parents = [""]
    values = [len(plot_df)]
    colors = [float(plot_df["Shot xG"].fillna(0).mean()) if "Shot xG" in plot_df else 0.0]
    custom = [[int(plot_df["Goal"].sum()) if "Goal" in plot_df else 0, float(plot_df["Shot xG"].fillna(0).sum()), colors[0]]]

    def add_node(node_id: str, label: str, parent: str, frame: pd.DataFrame) -> None:
        shots_count = len(frame)
        xg_total = float(frame["Shot xG"].fillna(0).sum()) if "Shot xG" in frame else 0.0
        avg_xg = xg_total / shots_count if shots_count else 0.0
        goals = int(frame["Goal"].sum()) if "Goal" in frame else 0
        ids.append(node_id)
        labels.append(label)
        parents.append(parent)
        values.append(shots_count)
        colors.append(avg_xg)
        custom.append([goals, xg_total, avg_xg])

    for outcome in OUTCOME_ORDER:
        outcome_frame = plot_df[plot_df["Outcome"] == outcome]
        if outcome_frame.empty:
            continue
        outcome_id = f"outcome::{outcome}"
        add_node(outcome_id, outcome, "all", outcome_frame)
        for shot_type, type_frame in outcome_frame.groupby("Shot Type", sort=False):
            type_id = f"type::{outcome}::{shot_type}"
            add_node(type_id, shot_type, outcome_id, type_frame)
            for body_part, body_frame in type_frame.groupby("Body Part Display", sort=False):
                body_id = f"body::{outcome}::{shot_type}::{body_part}"
                add_node(body_id, body_part, type_id, body_frame)

    max_color = max(max(colors), 0.18)
    fig = go.Figure(
        go.Treemap(
            labels=labels,
            ids=ids,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(
                colors=colors,
                colorscale=[
                    [0.0, "#fee2e2"],
                    [0.5, "#f59e0b"],
                    [1.0, "#15803d"],
                ],
                cmin=0,
                cmax=max_color,
                colorbar=dict(title="<b>Avg xG</b>"),
                line=dict(color="#ffffff", width=2),
            ),
            customdata=np.array(custom, dtype=object),
            texttemplate="<b>%{label}</b><br>%{value} shots",
            hovertemplate="%{label}<br>Shots: %{value:.0f}<br>Goals: %{customdata[0]:.0f}<br>xG: %{customdata[1]:.2f}<br>Avg xG: %{customdata[2]:.3f}<extra></extra>",
        )
    )
    fig.update_layout(height=620, margin=dict(l=20, r=20, t=70, b=20))
    return charting.polish_figure(fig, title)


def _body_part_efficiency_bubble(shots: pd.DataFrame, title: str) -> go.Figure:
    if shots.empty or "Body Part Display" not in shots:
        return _empty_chart(title, "No Body Part Data")

    summary = shots.groupby("Body Part Display", as_index=False).agg(
        Shots=("Body Part Display", "size"),
        Goals=("Goal", "sum"),
        xG=("Shot xG", "sum"),
        **{"Post-Shot xG": ("Post-Shot xG", "sum"), "On Target": ("On Target", "sum")},
    )
    if summary.empty:
        return _empty_chart(title, "No Body Part Data")
    summary["Avg xG"] = summary["xG"] / summary["Shots"].replace(0, np.nan)
    summary["On Target %"] = summary["On Target"] / summary["Shots"].replace(0, np.nan) * 100
    summary["PSxG - xG"] = summary["Post-Shot xG"].fillna(0) - summary["xG"].fillna(0)
    max_shots = max(float(summary["Shots"].max()), 1.0)

    fig = go.Figure(
        go.Scatter(
            x=summary["Shots"],
            y=summary["Avg xG"].fillna(0),
            mode="markers+text",
            text=summary["Body Part Display"].apply(lambda value: charting.wrap_label(value, 12, 2)),
            textposition="top center",
            marker=dict(
                size=18 + (summary["Shots"] / max_shots) * 34,
                color=summary["On Target %"].fillna(0),
                colorscale=[
                    [0.0, "#fee2e2"],
                    [0.5, "#f59e0b"],
                    [1.0, "#15803d"],
                ],
                cmin=0,
                cmax=100,
                colorbar=dict(
                    title=dict(text="<b>On Target %</b>", side="top"),
                    ticksuffix="%",
                    x=1.08,
                    len=0.78,
                    y=0.52,
                ),
                opacity=0.9,
                line=dict(color="#ffffff", width=1.4),
            ),
            customdata=np.stack([summary["Goals"], summary["xG"], summary["Post-Shot xG"], summary["PSxG - xG"], summary["On Target %"].fillna(0)], axis=-1),
            hovertemplate="%{text}<br>Shots: %{x:.0f}<br>Avg xG: %{y:.3f}<br>Goals: %{customdata[0]:.0f}<br>xG: %{customdata[1]:.2f}<br>Post-shot xG: %{customdata[2]:.2f}<br>PSxG - xG: %{customdata[3]:+.2f}<br>On target: %{customdata[4]:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(height=470, xaxis_title="<b>Shots</b>", yaxis_title="<b>Avg xG / Shot</b>", showlegend=False)
    fig = charting.polish_figure(fig, title)
    fig.update_layout(margin=dict(l=74, r=106, t=72, b=64))
    fig.update_xaxes(tickformat=".0f", rangemode="tozero", title_standoff=18)
    fig.update_yaxes(rangemode="tozero", tickformat=".2f", title_standoff=30)
    return fig


def _timeline_minute_label(value: object) -> str:
    minute = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(minute):
        return "Minute N/A"
    minute_value = int(round(float(minute)))
    if minute_value <= ADDED_TIME_START_MINUTE:
        return f"{minute_value}'"
    return f"90+{minute_value - int(ADDED_TIME_START_MINUTE)}'"


def _timeline_ticks(x_max: float) -> tuple[list[int], list[str]]:
    base_ticks = [0, 15, 30, 45, 60, 75, 90]
    added_ticks = [95, 100, 105, 110]
    ticks = [tick for tick in base_ticks + added_ticks if tick <= x_max]
    if int(round(x_max)) not in ticks:
        ticks.append(int(round(x_max)))
    ticks = sorted(set(ticks))
    labels = [str(tick) if tick <= 90 else f"90+{tick - 90}" for tick in ticks]
    return ticks, labels


def _add_timeline_display_minutes(shots: pd.DataFrame, timing_events: pd.DataFrame | None = None) -> pd.DataFrame:
    out = shots.copy()
    raw_minute = pd.to_numeric(out["Minute"], errors="coerce") if "Minute" in out else pd.Series(np.nan, index=out.index)
    seconds = pd.to_numeric(out["Second"], errors="coerce") if "Second" in out else pd.Series(np.nan, index=out.index)
    periods = pd.to_numeric(out["Period"], errors="coerce") if "Period" in out else pd.Series(np.nan, index=out.index)

    out["_Sort Second"] = seconds.fillna(raw_minute.fillna(0) * 60)
    out["_Display Minute"] = raw_minute

    timing_source = timing_events if timing_events is not None and not timing_events.empty else out
    if {"Period", "Second"}.issubset(timing_source.columns) and seconds.notna().any() and periods.notna().any():
        timing_periods = pd.to_numeric(timing_source["Period"], errors="coerce")
        timing_seconds = pd.to_numeric(timing_source["Second"], errors="coerce")
        period_starts = timing_source.assign(_Period=timing_periods, _Second=timing_seconds).dropna(subset=["_Period", "_Second"]).groupby("_Period")["_Second"].min()
        display_minutes = pd.Series(np.nan, index=out.index, dtype="float64")
        for idx in out.index:
            period = periods.loc[idx]
            second = seconds.loc[idx]
            if pd.isna(period) or pd.isna(second) or period not in period_starts.index:
                continue
            period_id = int(period)
            base_minute = PERIOD_BASE_MINUTES.get(period_id, max((period_id - 1) * 45.0, 0.0))
            elapsed_seconds = max(float(second) - float(period_starts.loc[period]), 0.0)
            display_minutes.loc[idx] = base_minute + np.floor(elapsed_seconds / 60.0) + 1.0
        out["_Display Minute"] = display_minutes.fillna(out["_Display Minute"])

    out["_Display Minute"] = pd.to_numeric(out["_Display Minute"], errors="coerce").fillna(0).clip(lower=0, upper=MAX_TIMELINE_MINUTE)
    out["_Minute Label"] = out["_Display Minute"].apply(_timeline_minute_label)
    return out


def _timeline_observed_end_minute(timing_events: pd.DataFrame | None, fallback_minute: float = 96.0) -> float:
    if timing_events is None or timing_events.empty:
        return fallback_minute
    timing = _add_timeline_display_minutes(timing_events, timing_events=timing_events)
    display_minutes = pd.to_numeric(timing["_Display Minute"], errors="coerce").dropna()
    if display_minutes.empty:
        return fallback_minute
    return max(fallback_minute, float(display_minutes.max()))


def _xg_timeline(shots: pd.DataFrame, title: str, timing_events: pd.DataFrame | None = None) -> go.Figure:
    if shots.empty:
        fig = go.Figure()
        fig.add_annotation(text="No Shot xG Events", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return charting.polish_figure(fig, title, height=460)
    plot_df = _add_timeline_display_minutes(shots, timing_events=timing_events)
    plot_df = plot_df.sort_values(["_Display Minute", "_Sort Second", "Event Number"]).copy()
    plot_df["Cumulative xG"] = plot_df["Shot xG"].fillna(0).cumsum()
    x = pd.concat([pd.Series([0.0]), plot_df["_Display Minute"]], ignore_index=True)
    y = pd.concat([pd.Series([0.0]), plot_df["Cumulative xG"]], ignore_index=True)
    minute_labels = ["Kick-off"] + plot_df["_Minute Label"].tolist()
    max_minute = float(plot_df["_Display Minute"].max()) if plot_df["_Display Minute"].notna().any() else 96.0
    observed_end_minute = _timeline_observed_end_minute(timing_events, fallback_minute=max_minute)
    x_max = max(96.0, min(MAX_TIMELINE_MINUTE, float(np.ceil((observed_end_minute + 4) / 5) * 5)))
    marker_sizes = [7] * len(x)
    if len(x) and float(x.iloc[-1]) < x_max:
        x = pd.concat([x, pd.Series([x_max])], ignore_index=True)
        y = pd.concat([y, pd.Series([float(y.iloc[-1])])], ignore_index=True)
        minute_labels.append(_timeline_minute_label(x_max))
        marker_sizes.append(0)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name="Cumulative xG",
            cliponaxis=False,
            line=dict(color=ui.CHARLTON_RED, width=3, shape="hv"),
            marker=dict(size=marker_sizes, color=ui.CHARLTON_RED, line=dict(color="#ffffff", width=1)),
            customdata=np.array(minute_labels, dtype=object),
            hovertemplate="%{customdata}<br>Cumulative xG: %{y:.2f}<extra></extra>",
        )
    )
    goals = plot_df[plot_df["Goal"]]
    if not goals.empty:
        fig.add_trace(
            go.Scatter(
                x=goals["_Display Minute"],
                y=goals["Cumulative xG"],
                mode="markers",
                name="Goals",
                cliponaxis=False,
                marker=dict(symbol="star", size=16, color="#111111", line=dict(color="#ffffff", width=1)),
                customdata=np.stack([goals["Player"].fillna("Unknown"), goals["Shot xG"].fillna(0), goals["_Minute Label"]], axis=-1),
                hovertemplate="%{customdata[0]} goal<br>%{customdata[2]}<br>Shot xG: %{customdata[1]:.3f}<extra></extra>",
            )
        )
    total_xg = float(plot_df["Cumulative xG"].max()) if plot_df["Cumulative xG"].notna().any() else 0.0
    y_max = max(0.12, total_xg * 1.18)
    if x_max > ADDED_TIME_START_MINUTE:
        fig.add_shape(
            type="rect",
            x0=ADDED_TIME_START_MINUTE,
            x1=x_max,
            y0=0,
            y1=y_max,
            line=dict(width=0),
            fillcolor="rgba(102,112,133,0.08)",
            layer="below",
        )
        fig.add_shape(
            type="line",
            x0=ADDED_TIME_START_MINUTE,
            x1=ADDED_TIME_START_MINUTE,
            y0=0,
            y1=y_max,
            line=dict(color="#667085", width=1.3, dash="dot"),
            layer="below",
        )
        fig.add_annotation(
            x=(ADDED_TIME_START_MINUTE + x_max) / 2,
            y=y_max * 0.96,
            text="<b>Added Time</b>",
            showarrow=False,
            font=dict(size=11, color="#475467"),
            bgcolor="rgba(255,255,255,0.76)",
            borderpad=2,
        )
    fig.add_annotation(
        x=min(max_minute, x_max - 1),
        y=total_xg,
        text=f"<b>{total_xg:.2f} xG</b>",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        xshift=8,
        yshift=8,
        bgcolor="rgba(255,255,255,0.78)",
        borderpad=2,
        font=dict(size=12, color=ui.CHARLTON_BLACK),
    )
    fig.update_layout(height=480, xaxis_title="<b>Minute</b>", yaxis_title="<b>Cumulative xG</b>")
    fig = charting.polish_figure(fig, title)
    tickvals, ticktext = _timeline_ticks(x_max)
    fig.update_xaxes(range=[0, x_max], tickvals=tickvals, ticktext=ticktext, fixedrange=True)
    fig.update_yaxes(range=[0, y_max], tickformat=".2f", fixedrange=True)
    return fig


def _xg_timeline_key(title: str = "Cumulative xG Timeline Key", height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0.06, 0.16, 0.16, 0.27, 0.27, 0.38],
            y=[0.35, 0.35, 0.55, 0.55, 0.72, 0.72],
            mode="lines+markers",
            line=dict(color=ui.CHARLTON_RED, width=3, shape="hv"),
            marker=dict(size=7, color=ui.CHARLTON_RED, line=dict(color="#ffffff", width=1)),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.53],
            y=[0.55],
            mode="markers",
            marker=dict(symbol="star", size=18, color="#111111", line=dict(color="#ffffff", width=1)),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_shape(
        type="rect",
        x0=0.70,
        x1=0.91,
        y0=0.22,
        y1=0.82,
        line=dict(width=0),
        fillcolor="rgba(102,112,133,0.08)",
        layer="below",
    )
    fig.add_shape(
        type="line",
        x0=0.70,
        x1=0.70,
        y0=0.22,
        y1=0.82,
        line=dict(color="#667085", width=1.4, dash="dot"),
    )
    fig.add_annotation(
        x=0.06,
        y=0.14,
        text="<b>Cumulative xG</b><br>Running total after each shot",
        showarrow=False,
        xanchor="left",
        align="left",
        font=dict(size=11, color=ui.CHARLTON_BLACK),
    )
    fig.add_annotation(
        x=0.53,
        y=0.14,
        text="<b>Goal</b><br>Scored shot",
        showarrow=False,
        xanchor="center",
        align="center",
        font=dict(size=11, color=ui.CHARLTON_BLACK),
    )
    fig.add_annotation(
        x=0.805,
        y=0.14,
        text="<b>Added Time</b><br>Minutes beyond 90 shown as 90+",
        showarrow=False,
        xanchor="center",
        align="center",
        font=dict(size=11, color=ui.CHARLTON_BLACK),
    )
    fig = charting.polish_figure(fig, title, height=height)
    fig.update_layout(showlegend=False, margin=dict(l=18, r=18, t=58, b=18))
    fig.update_xaxes(range=[0, 1], visible=False, fixedrange=True)
    fig.update_yaxes(range=[0, 1], visible=False, fixedrange=True)
    return fig


def _distance_quality_scatter(shots: pd.DataFrame, title: str) -> go.Figure:
    plot_df = shots.dropna(subset=["Shot Distance", "Shot xG"]).copy()
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No Distance/xG Shot Data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return charting.polish_figure(fig, title, height=520)

    plot_df["Post-Shot xG Display"] = plot_df["Post-Shot xG"].fillna(plot_df["Shot xG"]).fillna(0)
    max_psxg = max(float(plot_df["Post-Shot xG Display"].max()), 0.08)
    max_distance = float(plot_df["Shot Distance"].max()) if plot_df["Shot Distance"].notna().any() else 25.0
    x_max = max(30.0, float(np.ceil((max_distance + 3) / 5) * 5))
    max_xg = float(plot_df["Shot xG"].max()) if plot_df["Shot xG"].notna().any() else 0.0
    y_max = max(0.35, max_xg * 1.22)
    avg_distance = float(plot_df["Shot Distance"].mean())
    avg_xg = float(plot_df["Shot xG"].mean())

    fig = go.Figure()
    for zone in SHOT_DISTANCE_ZONES:
        x0 = float(zone["min"])
        x1 = x_max if zone["max"] is None else float(zone["max"])
        if x0 >= x_max:
            continue
        zone_end = min(x1, x_max)
        fig.add_shape(type="rect", x0=x0, x1=zone_end, y0=0, y1=y_max, line=dict(width=0), fillcolor=zone["color"], layer="below")
        fig.add_annotation(
            x=(x0 + zone_end) / 2,
            y=y_max * 0.97,
            text=f"<b>{zone['label']}</b>",
            showarrow=False,
            font=dict(size=10, color="#667085"),
            bgcolor="rgba(255,255,255,0.55)",
            borderpad=1,
        )

    fig.add_shape(type="line", x0=avg_distance, x1=avg_distance, y0=0, y1=y_max, line=dict(color="#667085", width=1.4, dash="dot"), layer="below")
    fig.add_shape(type="line", x0=0, x1=x_max, y0=avg_xg, y1=avg_xg, line=dict(color="#667085", width=1.4, dash="dot"), layer="below")
    fig.add_annotation(
        x=avg_distance,
        y=y_max * 0.08,
        text=f"<b>Avg distance</b><br>{avg_distance:.1f}m",
        showarrow=False,
        xanchor="left",
        font=dict(size=10, color="#475467"),
        bgcolor="rgba(255,255,255,0.72)",
        borderpad=2,
    )
    fig.add_annotation(
        x=x_max,
        y=avg_xg,
        text=f"<b>Avg xG {avg_xg:.2f}</b>",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        yshift=6,
        font=dict(size=10, color="#475467"),
        bgcolor="rgba(255,255,255,0.72)",
        borderpad=2,
    )

    for outcome in OUTCOME_ORDER:
        group = plot_df[plot_df["Outcome"] == outcome].copy()
        if group.empty:
            continue
        group["_Label"] = np.where(
            group["Outcome"].eq("Goal") | (group["Shot xG"].fillna(0) >= 0.20),
            group["Player"].apply(lambda value: charting.wrap_label(value, 12, 2)),
            "",
        )
        fig.add_trace(
            go.Scatter(
                x=group["Shot Distance"],
                y=group["Shot xG"],
                mode="markers+text",
                name=outcome,
                text=group["_Label"],
                textposition="top center",
                textfont=dict(size=10, color=ui.CHARLTON_BLACK),
                cliponaxis=False,
                marker=dict(
                    size=10 + (group["Post-Shot xG Display"] / max_psxg) * 24,
                    color=OUTCOME_COLORS.get(outcome, "#344054"),
                    symbol=OUTCOME_SYMBOLS.get(outcome, "circle"),
                    opacity=0.88,
                    line=dict(color="#ffffff", width=1),
                ),
                customdata=np.stack(
                    [
                        group["Player"].fillna("Unknown"),
                        group["Minute"].fillna(0),
                        group["Post-Shot xG Display"].fillna(0),
                        group["Shot Angle"].fillna(0),
                        group["Shot Type"].fillna("Shot"),
                        group["Body Part Display"].fillna("Unknown"),
                    ],
                    axis=-1,
                ),
                hovertemplate="%{customdata[0]}<br>Minute: %{customdata[1]:.0f}<br>Type: %{customdata[4]}<br>Body part: %{customdata[5]}<br>Distance: %{x:.1f}m<br>xG: %{y:.3f}<br>Post-shot xG: %{customdata[2]:.3f}<br>Angle: %{customdata[3]:.1f}<extra></extra>",
            )
        )
    fig.update_layout(
        height=520,
        xaxis_title="<b>Shot Distance (m)</b>",
        yaxis_title="<b>Shot Quality (xG)</b>",
        legend_title_text="<b>Outcome</b>",
    )
    fig = charting.polish_figure(fig, title)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0, title_text="<b>Outcome</b>"), margin=dict(l=58, r=34, t=76, b=112))
    fig.update_xaxes(range=[0, x_max], tickformat=".0f", ticksuffix="m", fixedrange=True)
    fig.update_yaxes(range=[0, y_max], tickformat=".2f", fixedrange=True)
    return fig


def _shot_distance_zone_pitch_key(title: str = "Shot Distance Zone Key") -> go.Figure:
    fig = pitch.pitch_image_figure(title, height=360, legend=True)
    goal_x = pitch.PITCH_X_MAX
    goal_y = 0.0
    x_min = goal_x - 34.0
    x_max = goal_x + 2.0
    y_min = -25.5
    y_max = 25.5

    # Long range is shown as the base layer; the closer ranges are overlaid as
    # semi-circular distance bands from the centre of the attacking goal.
    long_zone = SHOT_DISTANCE_ZONES[-1]
    fig.add_trace(
        go.Scatter(
            x=[x_min, goal_x, goal_x, x_min],
            y=[y_min, y_min, y_max, y_max],
            mode="lines",
            name=f"{long_zone['label']} ({long_zone['range_label']})",
            fill="toself",
            fillcolor=long_zone["color"],
            line=dict(color=long_zone["line"], width=1.4),
            hovertemplate=f"{long_zone['label']}<br>{long_zone['range_label']} from goal<extra></extra>",
            legendrank=4,
        )
    )

    angles = np.linspace(np.pi / 2, 3 * np.pi / 2, 96)
    for legend_rank, zone in reversed(list(enumerate(SHOT_DISTANCE_ZONES[:-1], start=1))):
        inner = float(zone["min"])
        outer = float(zone["max"])
        outer_x = goal_x + outer * np.cos(angles)
        outer_y = goal_y + outer * np.sin(angles)
        if inner <= 0:
            band_x = np.concatenate([[goal_x], outer_x, [goal_x]])
            band_y = np.concatenate([[goal_y], outer_y, [goal_y]])
        else:
            reverse_angles = angles[::-1]
            inner_x = goal_x + inner * np.cos(reverse_angles)
            inner_y = goal_y + inner * np.sin(reverse_angles)
            band_x = np.concatenate([outer_x, inner_x])
            band_y = np.concatenate([outer_y, inner_y])

        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=band_y,
                mode="lines",
                name=f"{zone['label']} ({zone['range_label']})",
                fill="toself",
                fillcolor=zone["color"],
                line=dict(color=zone["line"], width=1.7),
                hovertemplate=f"{zone['label']}<br>{zone['range_label']} from goal<extra></extra>",
                legendrank=legend_rank,
            )
        )

    fig.add_annotation(
        x=goal_x - 0.6,
        y=0,
        text="<b>Goal</b>",
        showarrow=False,
        xanchor="right",
        font=dict(size=11, color=ui.CHARLTON_BLACK),
        bgcolor="rgba(255,255,255,0.72)",
        borderpad=2,
    )
    fig.add_annotation(
        x=x_min + 1.0,
        y=y_max - 2.2,
        text="Colours match the distance bands used in the chart above.",
        showarrow=False,
        xanchor="left",
        font=dict(size=11, color="#475467"),
        bgcolor="rgba(255,255,255,0.76)",
        borderpad=3,
    )
    fig.update_xaxes(range=[x_min, x_max], visible=False, fixedrange=True)
    fig.update_yaxes(range=[y_min, y_max], visible=False, fixedrange=True, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        margin=dict(l=18, r=18, t=64, b=82),
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0, title_text="<b>Distance Band</b>"),
    )
    return fig


def _finishing_table(shots: pd.DataFrame) -> pd.DataFrame:
    if shots.empty:
        return pd.DataFrame(columns=["Player", "Shots", "Goals", "xG", "Post-Shot xG", "Avg xG", "Conversion %", "On Target %", "PSxG - xG"])
    grouped = shots.groupby("Player", as_index=False).agg(
        Shots=("Player", "size"),
        Goals=("Goal", "sum"),
        xG=("Shot xG", "sum"),
        **{
            "Post-Shot xG": ("Post-Shot xG", "sum"),
            "On Target": ("On Target", "sum"),
        },
    )
    grouped["Avg xG"] = grouped["xG"] / grouped["Shots"].replace(0, np.nan)
    grouped["Conversion %"] = grouped["Goals"] / grouped["Shots"].replace(0, np.nan) * 100
    grouped["On Target %"] = grouped["On Target"] / grouped["Shots"].replace(0, np.nan) * 100
    grouped["PSxG - xG"] = grouped["Post-Shot xG"].fillna(0) - grouped["xG"].fillna(0)
    for col in ["xG", "Post-Shot xG", "Avg xG", "Conversion %", "On Target %", "PSxG - xG"]:
        grouped[col] = grouped[col].round(2 if "%" not in col else 1)
    return grouped.drop(columns=["On Target"]).sort_values(["Shots", "xG"], ascending=False).reset_index(drop=True)


def _shooting_profile_table(players: pd.DataFrame) -> pd.DataFrame:
    out = players.copy()
    if "xG /90" in out and "Shots /90" in out:
        out["xG per Shot"] = out["xG /90"] / out["Shots /90"].replace(0, np.nan)
    else:
        out["xG per Shot"] = np.nan
    if "Goals /90" in out and "xG /90" in out:
        out["Goals - xG /90"] = out["Goals /90"] - out["xG /90"]
    else:
        out["Goals - xG /90"] = np.nan
    if "Post-Shot xG /90" in out and "xG /90" in out:
        out["PSxG - xG /90"] = out["Post-Shot xG /90"] - out["xG /90"]
    else:
        out["PSxG - xG /90"] = np.nan
    return out


pa.page_header(
    "Shooting Dashboard",
    "Analyse player shooting output, mapped shot locations, shot outcomes, chance quality and post-shot execution.",
    basis=SHOOTING_BASIS,
    limitation=(
        "Shot outcome categories are inferred from the available Impect action/result labels. "
        "Blocked, saved and off-target splits depend on those labels being populated in the event feed."
    ),
)

pa.section_heading("Season Shooting Profile")
season = pa.select_season(key="shooting_dashboard_season")
players = _shooting_profile_table(pa.load_player_data(season))
selected_player: str | None = None

if players.empty:
    st.warning("No players are available for the selected player season.")
else:
    selected_player = pa.player_selector(players, key="shooting_dashboard_player", label="Season Player")
    row = pa.player_row(players, selected_player)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Goals /90", pa.metric_value(row.get("Goals /90"), "Goals /90"))
    c2.metric("xG /90", pa.metric_value(row.get("xG /90"), "xG /90"))
    c3.metric("Shots /90", pa.metric_value(row.get("Shots /90"), "Shots /90"))
    c4.metric("xG per Shot", charting.metric_text(row.get("xG per Shot"), "Shot xG"))
    c5.metric("Goals - xG /90", charting.metric_text(row.get("Goals - xG /90"), "Goals /90"))

    scatter_cols = st.columns(2)
    with scatter_cols[0]:
        st.plotly_chart(
            pa.metric_scatter(
                players,
                x="Shots /90",
                y="xG /90",
                selected=selected_player,
                size="Goals /90",
                title="Shot Volume vs Chance Quality",
                show_median_lines=True,
            ),
            width="stretch",
        )
    with scatter_cols[1]:
        y_metric = "Post-Shot xG /90" if "Post-Shot xG /90" in players else "Goals /90"
        st.plotly_chart(
            pa.metric_scatter(
                players,
                x="xG /90",
                y=y_metric,
                selected=selected_player,
                size="Shots /90",
                title=f"xG vs {y_metric}",
                show_median_lines=True,
            ),
            width="stretch",
        )

pa.section_heading("Selected-Match Shot Mapping")
st.caption(
    "Use the fixture controls to inspect where shots came from, how good the chances were, and how outcomes differed by player, body part and shot type."
)

match_season = ma.select_match_season(key="shooting_dashboard_match_season")
matches = ma.load_matches(match_season)
if matches.empty:
    st.warning("No match data is available for the selected match season.")
    st.stop()

match_row = ma.match_selector(matches, key="shooting_dashboard_match")
team_name = ma.team_selector_for_match(match_row, key="shooting_dashboard_team")

raw_events = data.load_match_events(
    season=match_season,
    match_id=match_row.get("MatchId"),
    team=team_name,
    limit=12000,
)
shot_events = raw_events[raw_events["Action Type"].astype(str).str.upper().eq("SHOT")].copy() if "Action Type" in raw_events else raw_events.copy()
shots = _prepare_shots(shot_events.dropna(subset=["Start X", "Start Y"]).copy())
shots = _add_timeline_display_minutes(shots, timing_events=raw_events)
shots["Match Minute"] = shots["_Display Minute"]
shots["Match Minute Label"] = shots["_Minute Label"]

if shots.empty:
    st.info("No mapped shot events are available for this selected fixture and team.")
    st.stop()

control_cols = st.columns(4)
body_parts = sorted(shots["Body Part Display"].dropna().astype(str).unique().tolist())
shot_types = sorted(shots["Shot Type"].dropna().astype(str).unique().tolist())
outcomes = [outcome for outcome in OUTCOME_ORDER if outcome in set(shots["Outcome"])]
selected_body_parts = control_cols[0].multiselect("Body Parts", body_parts, default=body_parts)
selected_shot_types = control_cols[1].multiselect("Shot Types", shot_types, default=shot_types)
selected_outcomes = control_cols[2].multiselect("Outcomes", outcomes, default=outcomes)
min_xg = control_cols[3].number_input("Minimum xG", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

filtered_shots = shots.copy()
if selected_body_parts:
    filtered_shots = filtered_shots[filtered_shots["Body Part Display"].astype(str).isin(selected_body_parts)]
if selected_shot_types:
    filtered_shots = filtered_shots[filtered_shots["Shot Type"].astype(str).isin(selected_shot_types)]
if selected_outcomes:
    filtered_shots = filtered_shots[filtered_shots["Outcome"].isin(selected_outcomes)]
filtered_shots = filtered_shots[filtered_shots["Shot xG"].fillna(0) >= min_xg]

player_options = sorted(filtered_shots["Player"].dropna().astype(str).unique().tolist(), key=_last_name_sort)
if not player_options:
    st.info("No shooters match the current filters.")
    st.stop()

default_player = selected_player if selected_player in player_options else st.session_state.get("selected_player")
default_index = player_options.index(default_player) if default_player in player_options else 0
if st.session_state.get("shooting_dashboard_event_player") not in player_options:
    st.session_state.pop("shooting_dashboard_event_player", None)
match_player = st.selectbox("Match Player", player_options, index=default_index, key="shooting_dashboard_event_player")
st.session_state["selected_player"] = match_player

player_shots = filtered_shots[filtered_shots["Player"].astype(str) == str(match_player)].copy()
summary = _shot_summary(player_shots)
team_summary = _shot_summary(filtered_shots)

summary_cols = st.columns(5)
summary_cols[0].metric("Player Shots", charting.metric_text(summary["Shots"], "Actions"))
summary_cols[1].metric("Goals", charting.metric_text(summary["Goals"], "Goals"))
summary_cols[2].metric("Player xG", charting.metric_text(summary["xG"], "Shot xG"))
summary_cols[3].metric("Avg xG", charting.metric_text(summary["Avg xG"], "Shot xG"))
summary_cols[4].metric("On Target", f"{summary['On Target %']:.1f}%")

summary_cols_2 = st.columns(5)
summary_cols_2[0].metric("Team Shots", charting.metric_text(team_summary["Shots"], "Actions"))
summary_cols_2[1].metric("Team xG", charting.metric_text(team_summary["xG"], "Shot xG"))
summary_cols_2[2].metric("Post-Shot xG", charting.metric_text(summary["Post-Shot xG"], "Post-Shot xG"))
summary_cols_2[3].metric("Avg Distance", charting.metric_text(summary["Avg Distance"], "Metres"))
summary_cols_2[4].metric("Conversion", f"{summary['Conversion %']:.1f}%")

tabs = st.tabs(["Shot Maps", "Outcomes & Types", "Quality & Timing", "Goalmouth", "Event Table"])

with tabs[0]:
    pa.section_heading("Mapped Shot Locations")
    map_height = 650
    map_control_cols = st.columns(2)
    with map_control_cols[0]:
        show_team_context = st.checkbox("Show Team Shot Context", value=True)
    with map_control_cols[1]:
        heatmap_mode = st.radio("Zone Heatmap Value", ["Shot Count", "xG"], horizontal=True)

    map_cols = st.columns(2)
    with map_cols[0]:
        st.plotly_chart(
            _shot_map(
                player_shots,
                f"{match_player}: Shot Map by Outcome and xG",
                context_shots=filtered_shots if show_team_context else None,
                selected_player=match_player,
                height=map_height,
            ),
            width="stretch",
        )
        st.caption(
            "Map key: colour shows the shot outcome, marker shape shows the shot type, and marker size scales with pre-shot xG. "
            "Hover a marker for body part, distance, angle, xG and post-shot xG."
        )
    with map_cols[1]:
        st.plotly_chart(
            _shot_zone_heatmap(filtered_shots, f"{team_name}: Shot Origin Heatmap", value_mode=heatmap_mode, height=map_height),
            width="stretch",
        )

with tabs[1]:
    pa.section_heading("Shot Outcomes and Types")
    st.caption("This view links shot outcome, type, body part and chance quality so the pattern of shots is visible, not just the raw count.")

    outcome_cols = st.columns(2)
    with outcome_cols[0]:
        st.plotly_chart(_outcome_chart(player_shots, f"{match_player}: Shot Outcomes"), width="stretch")
    with outcome_cols[1]:
        st.plotly_chart(_body_part_efficiency_bubble(player_shots, f"{match_player}: Body Part Efficiency"), width="stretch")

    st.plotly_chart(_shot_mix_treemap(player_shots, f"{match_player}: Shot Mix Tree"), width="stretch")

    with st.expander("Shooter summary table"):
        finishing = _finishing_table(filtered_shots)
        if finishing.empty:
            st.caption("No shooter summary is available.")
        else:
            st.dataframe(finishing, width="stretch", hide_index=True)

with tabs[2]:
    pa.section_heading("Shot Quality and Timing")
    quality_cols = st.columns(2)
    with quality_cols[0]:
        st.plotly_chart(_xg_timeline(player_shots, f"{match_player}: Cumulative xG Timeline", timing_events=raw_events), width="stretch")
    with quality_cols[1]:
        st.plotly_chart(_distance_quality_scatter(player_shots, f"{match_player}: Shot Distance vs Chance Quality"), width="stretch")

    key_cols = st.columns(2)
    with key_cols[0]:
        st.plotly_chart(_xg_timeline_key(height=360), width="stretch")
    with key_cols[1]:
        st.plotly_chart(_shot_distance_zone_pitch_key(), width="stretch")

with tabs[3]:
    pa.section_heading("Goalmouth and Shot Execution")
    st.caption(
        "Goalmouth view uses Shot Target Y/Z where available, shown from the shooter's view to match external shot maps. "
        "The key is below the chart: colour and symbol show shot outcome, while marker size shows post-shot xG "
        "(falling back to pre-shot xG when unavailable). The axis scale is fixed tightly around the goal so players "
        "can be compared on the same coordinates."
    )
    goal_cols = st.columns([0.04, 0.92, 0.04])
    with goal_cols[1]:
        st.plotly_chart(
            pitch.goalmouth_shot_map(
                player_shots,
                f"{match_player}: Goalmouth Shot Placement",
                group_col="Outcome",
                group_order=OUTCOME_ORDER,
                group_colors=OUTCOME_COLORS,
                group_symbols=OUTCOME_SYMBOLS,
                height=680,
            ),
            width="stretch",
        )

    pa.section_heading("Shot Execution Detail")
    execution_cols = ma.available_columns(
        player_shots,
        [
            "Match Minute Label",
            "Outcome",
            "Shot xG",
            "Post-Shot xG",
            "Post-Shot xG Added",
            "Shot Distance",
            "Shot Angle",
            "Shot Target Y",
            "Shot Target Z",
            "Body Part Display",
            "Shot Type",
        ],
    )
    if execution_cols:
        execution_table = player_shots[execution_cols + ["_Display Minute"]].copy() if "_Display Minute" in player_shots else player_shots[execution_cols].copy()
        if "_Display Minute" in execution_table:
            execution_table = execution_table.sort_values("_Display Minute").drop(columns=["_Display Minute"])
        execution_table = execution_table.rename(columns={"Match Minute Label": "Minute"})
        st.dataframe(execution_table, width="stretch", hide_index=True)
    else:
        st.caption("No post-shot execution fields are available.")

with tabs[4]:
    pa.section_heading("Mapped Shot Event Table")
    display_cols = ma.available_columns(
        filtered_shots,
        [
            "Match Minute Label",
            "Player",
            "Outcome",
            "Shot Type",
            "Body Part Display",
            "Result",
            "Shot xG",
            "Post-Shot xG",
            "Packing xG",
            "PXT Shot",
            "Shot Distance",
            "Shot Angle",
            "Start Lane",
            "Start Pitch Position",
            "Start X",
            "Start Y",
            "Shot Target Y",
            "Shot Target Z",
        ],
    )
    event_table = filtered_shots[display_cols + ["_Display Minute"]].copy() if "_Display Minute" in filtered_shots else filtered_shots[display_cols].copy()
    if "_Display Minute" in event_table:
        event_table = event_table.sort_values(["_Display Minute", "Player"]).drop(columns=["_Display Minute"])
    event_table = event_table.rename(columns={"Match Minute Label": "Minute"})
    st.dataframe(event_table, width="stretch", hide_index=True)
