# =============================================================================
# PASSING DASHBOARD - aggregate player passing plus event-backed pass maps
# =============================================================================
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charting, data, match_analysis as ma, pitch, player_analysis as pa, ui


PASSING_BASIS = (
    "Season aggregate player passing metrics come from CAFC_DB Impect player-iteration KPI facts. "
    "Selected-fixture maps, direction splits, receiver links and matrices use mapped Impect pass events "
    "with adjusted start/end coordinates. Passing networks are derived from successful event-level passes."
)

DIRECTION_ORDER = ["Progressive", "Lateral", "Regressive"]
DIRECTION_COLORS = {
    "Progressive": ui.CHARLTON_RED,
    "Lateral": "#c69214",
    "Regressive": "#344054",
}
OUTCOME_ORDER = ["Complete", "Incomplete", "Other"]
OUTCOME_COLORS = {
    "Complete": pitch.PASS_OUTCOME_COLORS["Complete"],
    "Incomplete": pitch.PASS_OUTCOME_COLORS["Incomplete"],
    "Other": pitch.PASS_OUTCOME_COLORS["Neutral"],
}


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _pct(part: float | int, whole: float | int) -> float:
    return (float(part) / float(whole) * 100) if whole else 0.0


def _sum(frame: pd.DataFrame, column: str) -> float:
    return float(_numeric(frame, column).fillna(0).sum()) if not frame.empty else 0.0


def _mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return np.nan
    value = _numeric(frame, column).mean()
    return float(value) if pd.notna(value) else np.nan


def _last_name_sort(value: object) -> tuple[str, str]:
    text = "" if value is None else str(value)
    parts = text.strip().split()
    return (parts[-1].casefold() if parts else text.casefold(), text.casefold())


def _add_pass_direction(passes: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = passes.copy()
    for column in ["Start X", "Start Y", "End X", "End Y", "Pass Distance", "PXT Pass"]:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out["Territory Gain"] = out["End X"] - out["Start X"]
    out["Lateral Shift"] = out["End Y"] - out["Start Y"]
    inferred_distance = np.hypot(out["Territory Gain"], out["Lateral Shift"])
    out["Pass Distance"] = _numeric(out, "Pass Distance").fillna(inferred_distance)

    out["Pass Direction"] = np.select(
        [
            out["Territory Gain"] >= threshold,
            out["Territory Gain"] <= -threshold,
        ],
        ["Progressive", "Regressive"],
        default="Lateral",
    )
    result = out["Result"].astype(str).str.upper() if "Result" in out else pd.Series("", index=out.index)
    out["Outcome"] = np.select(
        [
            result.eq("SUCCESS"),
            result.eq("FAIL"),
        ],
        ["Complete", "Incomplete"],
        default="Other",
    )
    out["_Completed"] = out["Outcome"].eq("Complete")
    out["Final Third Entry"] = (
        (pd.to_numeric(out["Start X"], errors="coerce") < pitch.FINAL_THIRD_X)
        & (pd.to_numeric(out["End X"], errors="coerce") >= pitch.FINAL_THIRD_X)
    )
    return out


def _player_pass_summary(passes: pd.DataFrame) -> dict[str, float | int]:
    attempts = len(passes)
    completed = int(passes["_Completed"].sum()) if "_Completed" in passes else 0
    return {
        "Attempts": attempts,
        "Completed": completed,
        "Completion %": _pct(completed, attempts),
        "Progressive": int((passes["Pass Direction"] == "Progressive").sum()) if "Pass Direction" in passes else 0,
        "Regressive": int((passes["Pass Direction"] == "Regressive").sum()) if "Pass Direction" in passes else 0,
        "Final Third Entries": int(passes["Final Third Entry"].sum()) if "Final Third Entry" in passes else 0,
        "PXT Pass": _sum(passes, "PXT Pass"),
        "Avg Territory Gain": _mean(passes, "Territory Gain"),
    }


def _direction_accuracy_table(passes: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Pass Direction",
        "Attempts",
        "Completed",
        "Completion %",
        "Final Third Entries",
        "Avg Territory Gain",
        "PXT Pass",
    ]
    if passes.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for direction in DIRECTION_ORDER:
        group = passes[passes["Pass Direction"] == direction]
        if group.empty:
            continue
        attempts = len(group)
        completed = int(group["_Completed"].sum())
        rows.append(
            {
                "Pass Direction": direction,
                "Attempts": attempts,
                "Completed": completed,
                "Completion %": round(_pct(completed, attempts), 1),
                "Final Third Entries": int(group["Final Third Entry"].sum()),
                "Avg Territory Gain": round(_mean(group, "Territory Gain"), 2),
                "PXT Pass": round(_sum(group, "PXT Pass"), 3),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _direction_accuracy_chart(passes: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if passes.empty:
        fig.add_annotation(
            text="No Pass Attempts Match the Selected Filters",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=ui.CHARLTON_MUTED),
        )
        return charting.polish_figure(fig, title, height=430)

    counts = passes.groupby(["Pass Direction", "Outcome"], as_index=False).size()
    for outcome in OUTCOME_ORDER:
        values = []
        for direction in DIRECTION_ORDER:
            match = counts[(counts["Pass Direction"] == direction) & (counts["Outcome"] == outcome)]
            values.append(int(match["size"].iloc[0]) if not match.empty else 0)
        if any(values):
            fig.add_trace(
                go.Bar(
                    x=DIRECTION_ORDER,
                    y=values,
                    name=outcome,
                    marker_color=OUTCOME_COLORS.get(outcome, "#7a7f87"),
                    text=[charting.metric_text(value, "Actions") for value in values],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate=f"{outcome}<br>%{{x}}: %{{y:.0f}} passes<extra></extra>",
                )
            )

    fig.update_layout(
        barmode="group",
        height=430,
        xaxis_title="Pass Direction",
        yaxis_title="Passes",
        bargap=0.24,
    )
    fig.update_yaxes(tickformat=".0f")
    fig = charting.polish_figure(fig, title)
    fig.update_layout(
        legend=dict(
            orientation="h",
            title_text="Key",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=28, r=34, t=72, b=96),
    )
    return fig


def _pass_flow_map(passes: pd.DataFrame, title: str, max_passes: int) -> go.Figure:
    mapped = passes.dropna(subset=["Start X", "Start Y", "End X", "End Y"]).copy()
    if len(mapped) > max_passes:
        sort_value = _numeric(mapped, "PXT Pass").fillna(0).abs() + _numeric(mapped, "Territory Gain").fillna(0).abs() / 100
        mapped = mapped.assign(_SortValue=sort_value).sort_values("_SortValue", ascending=False).head(max_passes)

    fig = pitch.pitch_image_figure(title)
    if mapped.empty:
        fig.add_annotation(
            text="No Mapped Passes",
            x=0,
            y=0,
            xref="x",
            yref="y",
            showarrow=False,
            font=dict(size=16, color=ui.CHARLTON_MUTED),
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor=ui.CHARLTON_BORDER,
            borderpad=8,
        )
        return fig

    for direction in DIRECTION_ORDER:
        for outcome in OUTCOME_ORDER:
            group = mapped[(mapped["Pass Direction"] == direction) & (mapped["Outcome"] == outcome)]
            if group.empty:
                continue

            x_values: list[float | None] = []
            y_values: list[float | None] = []
            for _, row in group.iterrows():
                x_values += [row["Start X"], row["End X"], None]
                y_values += [row["Start Y"], row["End Y"], None]

            complete = outcome == "Complete"
            line_color = DIRECTION_COLORS.get(direction, "#7a7f87")
            line_width = 4.6 if complete else 3.1
            line_dash = "solid" if complete else "dot"
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    name=f"{direction} - {outcome} underlay",
                    legendgroup=f"{direction}-{outcome}",
                    showlegend=False,
                    line=dict(
                        color="rgba(255, 255, 255, 0.86)",
                        width=line_width + 3.0,
                        dash=line_dash,
                    ),
                    opacity=0.82,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    name=f"{direction} - {outcome}",
                    legendgroup=f"{direction}-{outcome}",
                    line=dict(
                        color=line_color,
                        width=line_width,
                        dash=line_dash,
                    ),
                    opacity=0.96 if complete else 0.76,
                    hoverinfo="skip",
                )
            )

            customdata = np.stack(
                [
                    group["Player"].fillna("Unknown") if "Player" in group else pd.Series("Unknown", index=group.index),
                    group["Receiver"].fillna("Unknown") if "Receiver" in group else pd.Series("Unknown", index=group.index),
                    group["Minute"].fillna(0) if "Minute" in group else pd.Series(0, index=group.index),
                    group["Outcome"],
                    group["Territory Gain"].fillna(0),
                    group["Pass Distance"].fillna(0),
                    _numeric(group, "PXT Pass").fillna(0),
                ],
                axis=-1,
            )
            fig.add_trace(
                go.Scatter(
                    x=group["End X"],
                    y=group["End Y"],
                    mode="markers",
                    name=f"{direction} end",
                    legendgroup=f"{direction}-{outcome}",
                    showlegend=False,
                    marker=dict(
                        size=7.5 if complete else 6,
                        color=DIRECTION_COLORS.get(direction, "#7a7f87"),
                        opacity=0.94 if complete else 0.68,
                        line=dict(color="#ffffff", width=1.0),
                    ),
                    customdata=customdata,
                    hovertemplate=(
                        "%{customdata[0]} to %{customdata[1]}"
                        "<br>Minute: %{customdata[2]:.0f}"
                        "<br>Outcome: %{customdata[3]}"
                        "<br>Territory gain: %{customdata[4]:.1f}m"
                        "<br>Distance: %{customdata[5]:.1f}m"
                        "<br>PXT pass: %{customdata[6]:.3f}<extra></extra>"
                    ),
                )
            )
    return fig


CROSS_TYPE_ORDER = ["Low Cross", "High Cross"]


def _cross_type_label(action: object) -> str:
    text = str(action).upper()
    if text == "HIGH_CROSS":
        return "High Cross"
    if text == "LOW_CROSS":
        return "Low Cross"
    return str(action).replace("_", " ").title()


def _crossing_summary(passes: pd.DataFrame) -> dict[str, float | int]:
    crosses = passes[data.is_cross(passes)].copy() if not passes.empty else passes
    attempts = len(crosses)
    completed = int(crosses["_Completed"].sum()) if "_Completed" in crosses and attempts else 0
    return {
        "Attempts": attempts,
        "Completed": completed,
        "Completion %": _pct(completed, attempts),
        "Low Crosses": int(crosses["Action"].astype(str).str.upper().eq("LOW_CROSS").sum()) if attempts else 0,
        "High Crosses": int(crosses["Action"].astype(str).str.upper().eq("HIGH_CROSS").sum()) if attempts else 0,
        "PXT Pass": _sum(crosses, "PXT Pass"),
    }


def _crossing_by_player_table(passes: pd.DataFrame) -> pd.DataFrame:
    columns = ["Player", "Attempts", "Completed", "Completion %", "Low", "High", "PXT Pass"]
    crosses = passes[data.is_cross(passes)].copy() if not passes.empty else passes
    if crosses.empty or "Player" not in crosses:
        return pd.DataFrame(columns=columns)

    rows = []
    for player, group in crosses.groupby("Player"):
        attempts = len(group)
        completed = int(group["_Completed"].sum())
        rows.append(
            {
                "Player": player,
                "Attempts": attempts,
                "Completed": completed,
                "Completion %": round(_pct(completed, attempts), 1),
                "Low": int(group["Action"].astype(str).str.upper().eq("LOW_CROSS").sum()),
                "High": int(group["Action"].astype(str).str.upper().eq("HIGH_CROSS").sum()),
                "PXT Pass": round(_sum(group, "PXT Pass"), 3),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["Attempts", "Completed"], ascending=False).reset_index(drop=True)


def _crossing_type_chart(passes: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    crosses = passes[data.is_cross(passes)].copy() if not passes.empty else passes
    if crosses.empty:
        fig.add_annotation(
            text="No Crosses Match the Selected Filters",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
            font=dict(color=ui.CHARLTON_MUTED),
        )
        return charting.polish_figure(fig, title, height=380)

    crosses["Cross Type"] = crosses["Action"].map(_cross_type_label)
    counts = crosses.groupby(["Cross Type", "Outcome"], as_index=False).size()
    for outcome in OUTCOME_ORDER:
        values = []
        for cross_type in CROSS_TYPE_ORDER:
            match = counts[(counts["Cross Type"] == cross_type) & (counts["Outcome"] == outcome)]
            values.append(int(match["size"].iloc[0]) if not match.empty else 0)
        if any(values):
            fig.add_trace(
                go.Bar(
                    x=CROSS_TYPE_ORDER,
                    y=values,
                    name=outcome,
                    marker_color=OUTCOME_COLORS.get(outcome, "#7a7f87"),
                    text=[charting.metric_text(value, "Actions") for value in values],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate=f"{outcome}<br>%{{x}}: %{{y:.0f}} crosses<extra></extra>",
                )
            )
    fig.update_layout(barmode="group", height=380, xaxis_title="Cross Type", yaxis_title="Crosses", bargap=0.3)
    fig.update_yaxes(tickformat=".0f")
    fig = charting.polish_figure(fig, title)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0), margin=dict(l=28, r=34, t=72, b=96))
    return fig


def _receiver_summary(passes: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Receiver",
        "Attempts",
        "Completed",
        "Completion %",
        "Progressive",
        "Regressive",
        "Final Third Entries",
        "Avg Distance",
        "Avg Territory Gain",
        "PXT Pass",
    ]
    if passes.empty or "Receiver" not in passes:
        return pd.DataFrame(columns=columns)

    usable = passes.dropna(subset=["Receiver"]).copy()
    if usable.empty:
        return pd.DataFrame(columns=columns)

    summary = usable.groupby("Receiver", as_index=False).agg(
        Attempts=("Receiver", "size"),
        Completed=("_Completed", "sum"),
        Progressive=("Pass Direction", lambda values: int((values == "Progressive").sum())),
        Regressive=("Pass Direction", lambda values: int((values == "Regressive").sum())),
        **{
            "Final Third Entries": ("Final Third Entry", "sum"),
            "Avg Distance": ("Pass Distance", "mean"),
            "Avg Territory Gain": ("Territory Gain", "mean"),
            "PXT Pass": ("PXT Pass", "sum"),
        },
    )
    summary["Completion %"] = (summary["Completed"] / summary["Attempts"].replace(0, np.nan) * 100).round(1)
    for column in ["Avg Distance", "Avg Territory Gain", "PXT Pass"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").round(2 if column != "PXT Pass" else 3)
    summary["Completed"] = summary["Completed"].astype(int)
    summary["Final Third Entries"] = summary["Final Third Entries"].astype(int)
    return summary[columns].sort_values(["Attempts", "Completed"], ascending=False).reset_index(drop=True)


def _network_from_pass_events(passes: pd.DataFrame) -> pd.DataFrame:
    if passes.empty or "Receiver" not in passes:
        return pd.DataFrame(columns=data.PASS_NETWORK_COLUMNS)

    events = passes.dropna(subset=["Player", "Receiver", "Start X", "Start Y", "End X", "End Y"]).copy()
    if events.empty:
        return pd.DataFrame(columns=data.PASS_NETWORK_COLUMNS)

    events["_MatchId"] = events["MatchId"] if "MatchId" in events else ""
    events["_Team"] = events["Team"] if "Team" in events else ""
    events["_PlayerId"] = events["PlayerId"] if "PlayerId" in events else events["Player"]
    events["_ReceiverId"] = events["ReceiverId"] if "ReceiverId" in events else events["Receiver"]
    events["_PlayerId"] = events["_PlayerId"].fillna(events["Player"]).astype(str)
    events["_ReceiverId"] = events["_ReceiverId"].fillna(events["Receiver"]).astype(str)

    grouped = events.groupby(
        ["_MatchId", "_Team", "_PlayerId", "Player", "_ReceiverId", "Receiver"],
        as_index=False,
        dropna=False,
    ).agg(
        **{
            "Pass Count": ("Player", "size"),
            "Passer X": ("Start X", "mean"),
            "Passer Y": ("Start Y", "mean"),
            "Receiver X": ("End X", "mean"),
            "Receiver Y": ("End Y", "mean"),
        }
    )
    grouped = grouped.rename(
        columns={
            "_MatchId": "MatchId",
            "_Team": "Team",
            "_PlayerId": "PlayerId",
            "_ReceiverId": "ReceiverId",
        }
    )
    return grouped[data.PASS_NETWORK_COLUMNS].reset_index(drop=True)


def _bold_axis_label(value: object, width: int = 14, max_lines: int = 2) -> str:
    return f"<b>{charting.wrap_label(value, width=width, max_lines=max_lines)}</b>"


def _style_matrix_axes(
    fig: go.Figure,
    title: str,
    height: int,
    x_title: str,
    y_title: str,
    x_values: list[object],
    y_values: list[object],
    label_width: int = 14,
) -> go.Figure:
    fig = charting.polish_figure(fig, title, height=height)
    fig.update_xaxes(
        title_text=f"<b>{x_title}</b>",
        tickmode="array",
        tickvals=x_values,
        ticktext=[_bold_axis_label(value, width=label_width) for value in x_values],
        tickfont=dict(size=11, color=ui.CHARLTON_BLACK, family="Inter, Arial, sans-serif"),
        title_font=dict(size=14, color=ui.CHARLTON_BLACK, family="Inter, Arial, sans-serif"),
        automargin=True,
    )
    fig.update_yaxes(
        title_text=f"<b>{y_title}</b>",
        tickmode="array",
        tickvals=y_values,
        ticktext=[_bold_axis_label(value, width=label_width) for value in y_values],
        tickfont=dict(size=11, color=ui.CHARLTON_BLACK, family="Inter, Arial, sans-serif"),
        title_font=dict(size=14, color=ui.CHARLTON_BLACK, family="Inter, Arial, sans-serif"),
        automargin=True,
    )
    fig.update_layout(margin=dict(l=92, r=34, t=68, b=112))
    return fig


def _passing_matrix_heatmap(passes: pd.DataFrame, selected_player: str, title: str, max_players: int, height: int = 620) -> go.Figure:
    fig = go.Figure()
    if passes.empty or "Receiver" not in passes:
        fig.add_annotation(text="No Completed Pass Matrix", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return charting.polish_figure(fig, title, height=height)

    links = passes.dropna(subset=["Player", "Receiver"]).copy()
    if links.empty:
        fig.add_annotation(text="No Completed Pass Matrix", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return charting.polish_figure(fig, title, height=height)

    links["Player"] = links["Player"].astype(str)
    links["Receiver"] = links["Receiver"].astype(str)
    counts = links.groupby(["Player", "Receiver"], as_index=False).size().rename(columns={"size": "Passes"})
    involvement = pd.concat(
        [
            counts.groupby("Player")["Passes"].sum(),
            counts.groupby("Receiver")["Passes"].sum(),
        ]
    ).groupby(level=0).sum().sort_values(ascending=False)

    order = involvement.index.astype(str).tolist()
    if selected_player in order:
        order = [selected_player] + [name for name in order if name != selected_player]
    order = order[:max_players]
    if selected_player in involvement.index.astype(str).tolist() and selected_player not in order and order:
        order[-1] = selected_player

    matrix = counts.pivot_table(index="Player", columns="Receiver", values="Passes", fill_value=0)
    matrix = matrix.reindex(index=order, columns=order, fill_value=0)
    max_passes = float(matrix.to_numpy().max()) if not matrix.empty else 0.0

    fig.add_trace(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale=[
                [0.0, "#f2f4f7"],
                [0.0001, "#dc2626"],
                [0.45, "#f59e0b"],
                [1.0, "#15803d"],
            ],
            zmin=0,
            zmax=max(max_passes, 1.0),
            xgap=1,
            ygap=1,
            colorbar=dict(
                title=dict(text="<b>Passes<br>Low → High</b>", font=dict(size=12, color=ui.CHARLTON_BLACK)),
                tickfont=dict(size=11, color=ui.CHARLTON_BLACK),
            ),
            hovertemplate="Passer: %{y}<br>Receiver: %{x}<br>Completed passes: %{z:.0f}<extra></extra>",
        )
    )
    for passer in matrix.index.tolist():
        for receiver in matrix.columns.tolist():
            value = float(matrix.loc[passer, receiver])
            if value <= 0:
                continue
            fig.add_annotation(
                x=receiver,
                y=passer,
                xref="x",
                yref="y",
                text=f"<b>{value:.0f}</b>",
                showarrow=False,
                font=dict(
                    size=11,
                    color="#ffffff" if max_passes and value >= max_passes * 0.55 else ui.CHARLTON_BLACK,
                    family="Inter, Arial, sans-serif",
                ),
            )
    return _style_matrix_axes(
        fig,
        title=title,
        height=height,
        x_title="Receiver",
        y_title="Passer",
        x_values=matrix.columns.tolist(),
        y_values=matrix.index.tolist(),
        label_width=13,
    )


def _lane_matrix_heatmap(passes: pd.DataFrame, title: str, height: int = 620) -> go.Figure:
    fig = go.Figure()
    if passes.empty or not {"Start Lane", "End Lane"}.issubset(passes.columns):
        fig.add_annotation(text="No Lane Matrix", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return charting.polish_figure(fig, title, height=height)

    lane_passes = passes.dropna(subset=["Start Lane", "End Lane"]).copy()
    if lane_passes.empty:
        fig.add_annotation(text="No Lane Matrix", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return charting.polish_figure(fig, title, height=height)

    lane_passes["Start Lane"] = lane_passes["Start Lane"].astype(str).str.replace("_", " ").str.title()
    lane_passes["End Lane"] = lane_passes["End Lane"].astype(str).str.replace("_", " ").str.title()
    preferred = ["Left", "Half Left", "Center", "Half Right", "Right"]
    present = set(lane_passes["Start Lane"]) | set(lane_passes["End Lane"])
    lanes = [lane for lane in preferred if lane in present]
    lanes += sorted(present - set(lanes))

    matrix = lane_passes.pivot_table(index="Start Lane", columns="End Lane", values="Player", aggfunc="size", fill_value=0)
    matrix = matrix.reindex(index=lanes, columns=lanes, fill_value=0)
    max_passes = float(matrix.to_numpy().max()) if not matrix.empty else 0.0
    fig.add_trace(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale=[
                [0.0, "#f2f4f7"],
                [0.0001, "#dc2626"],
                [0.45, "#f59e0b"],
                [1.0, "#15803d"],
            ],
            zmin=0,
            zmax=max(max_passes, 1.0),
            xgap=1,
            ygap=1,
            colorbar=dict(
                title=dict(text="<b>Passes<br>Low → High</b>", font=dict(size=12, color=ui.CHARLTON_BLACK)),
                tickfont=dict(size=11, color=ui.CHARLTON_BLACK),
            ),
            hovertemplate="Start lane: %{y}<br>End lane: %{x}<br>Passes: %{z:.0f}<extra></extra>",
        )
    )
    for start_lane in matrix.index.tolist():
        for end_lane in matrix.columns.tolist():
            value = float(matrix.loc[start_lane, end_lane])
            if value <= 0:
                continue
            fig.add_annotation(
                x=end_lane,
                y=start_lane,
                xref="x",
                yref="y",
                text=f"<b>{value:.0f}</b>",
                showarrow=False,
                font=dict(
                    size=11,
                    color="#ffffff" if max_passes and value >= max_passes * 0.55 else ui.CHARLTON_BLACK,
                    family="Inter, Arial, sans-serif",
                ),
            )
    return _style_matrix_axes(
        fig,
        title=title,
        height=height,
        x_title="End Lane",
        y_title="Start Lane",
        x_values=matrix.columns.tolist(),
        y_values=matrix.index.tolist(),
        label_width=12,
    )


pa.page_header(
    "Passing Dashboard",
    "Analyse player passing through season output, selected-match accuracy, progression, pass flow maps, receiver links and player-to-player matrices.",
    basis=PASSING_BASIS,
)

pa.section_heading("Season Passing Profile")
season = pa.select_season(key="passing_dashboard_season")
players = pa.passing_table(pa.load_player_data(season))
selected_player: str | None = None

if players.empty:
    st.warning("No players are available for the selected player season.")
else:
    selected_player = pa.player_selector(players, key="passing_dashboard_player", label="Season Player")

    required = ["Pass %", "Passes to Final 3rd /90", "Bypassed Opponents /90"]
    missing = [metric for metric in required if metric not in players.columns]
    if missing:
        st.warning(f"Missing aggregate passing metrics: {', '.join(missing)}")
    else:
        row = pa.player_row(players, selected_player)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pass %", pa.metric_value(row["Pass %"], "Pass %"))
        c2.metric("Passes to Final 3rd /90", pa.metric_value(row["Passes to Final 3rd /90"], "Passes to Final 3rd /90"))
        c3.metric("Bypassed Opponents /90", pa.metric_value(row["Bypassed Opponents /90"], "Bypassed Opponents /90"))
        c4.metric("Passing Impact", f"{row['Passing Impact']:.0f}th percentile")

        st.plotly_chart(
            pa.metric_scatter(
                players,
                x="Pass %",
                y="Passes to Final 3rd /90",
                selected=selected_player,
                size="Bypassed Opponents /90",
                title="Pass Security vs Final-Third Progression",
                show_median_lines=True,
            ),
            width="stretch",
        )

pa.section_heading("Selected-Match Pass Mapping")
st.caption(
    "Progressive/regressive splits are calculated from adjusted pitch coordinates. "
    "A pass is progressive if its end x-coordinate is at least the selected threshold closer to goal; "
    "regressive is the same threshold in the opposite direction."
)

match_season = ma.select_match_season(key="passing_dashboard_match_season")
matches = ma.load_matches(match_season)
if matches.empty:
    st.warning("No match data is available for the selected match season.")
    st.stop()

match_row = ma.match_selector(matches, key="passing_dashboard_match")
team_name = ma.team_selector_for_match(match_row, key="passing_dashboard_team")

raw_events = data.load_match_events(
    season=match_season,
    match_id=match_row.get("MatchId"),
    team=team_name,
    action_types=["PASS"],
    limit=12000,
)
passes = raw_events.dropna(subset=["Start X", "Start Y", "End X", "End Y"]).copy()
pass_network_rows = data.load_pass_network(
    season=match_season,
    match_id=match_row.get("MatchId"),
    team=team_name,
)

if passes.empty:
    st.info("No mapped pass events are available for this selected fixture and team.")
    st.stop()

control_cols = st.columns(4)
threshold = control_cols[0].number_input("Progression Threshold (Metres)", min_value=0.0, max_value=30.0, value=5.0, step=1.0)
passes = _add_pass_direction(passes, threshold=threshold)

outcomes = [outcome for outcome in OUTCOME_ORDER if outcome in set(passes["Outcome"])]
selected_outcomes = control_cols[1].multiselect("Outcomes", outcomes, default=outcomes)
selected_directions = control_cols[2].multiselect("Directions", DIRECTION_ORDER, default=DIRECTION_ORDER)
min_distance = control_cols[3].number_input("Minimum Pass Distance", min_value=0.0, value=0.0, step=5.0)

filtered_passes = passes.copy()
if selected_outcomes:
    filtered_passes = filtered_passes[filtered_passes["Outcome"].isin(selected_outcomes)]
if selected_directions:
    filtered_passes = filtered_passes[filtered_passes["Pass Direction"].isin(selected_directions)]
filtered_passes = filtered_passes[_numeric(filtered_passes, "Pass Distance").fillna(0) >= min_distance]

player_options = sorted(filtered_passes["Player"].dropna().astype(str).unique().tolist(), key=_last_name_sort)
if not player_options:
    st.info("No passers match the current filters.")
    st.stop()

default_player = selected_player if selected_player in player_options else st.session_state.get("selected_player")
default_index = player_options.index(default_player) if default_player in player_options else 0
if st.session_state.get("passing_dashboard_event_player") not in player_options:
    st.session_state.pop("passing_dashboard_event_player", None)
match_player = st.selectbox("Match Player", player_options, index=default_index, key="passing_dashboard_event_player")
st.session_state["selected_player"] = match_player

player_passes = filtered_passes[filtered_passes["Player"].astype(str) == str(match_player)].copy()
summary = _player_pass_summary(player_passes)

summary_cols = st.columns(4)
summary_cols[0].metric("Pass Attempts", charting.metric_text(summary["Attempts"], "Actions"))
summary_cols[1].metric("Completion", f"{summary['Completion %']:.1f}%")
summary_cols[2].metric("Progressive Passes", charting.metric_text(summary["Progressive"], "Actions"))
summary_cols[3].metric("Regressive Passes", charting.metric_text(summary["Regressive"], "Actions"))

summary_cols_2 = st.columns(4)
summary_cols_2[0].metric("Final-Third Entries", charting.metric_text(summary["Final Third Entries"], "Actions"))
summary_cols_2[1].metric("Total PXT Pass", charting.metric_text(summary["PXT Pass"], "PXT Pass"))
summary_cols_2[2].metric("Avg Territory Gain", charting.metric_text(summary["Avg Territory Gain"], "Metres"))
summary_cols_2[3].metric("Mapped Team Passes", charting.metric_text(len(filtered_passes), "Actions"))

tabs = st.tabs(["Pass Map", "Accuracy & Direction", "Crosses", "Network", "Matrices", "Event Table"])

with tabs[0]:
    pa.section_heading("Player Pass Flow Map")
    map_limit = st.slider("Maximum Plotted Passes", 50, 800, min(450, max(len(player_passes), 50)), step=50)
    st.plotly_chart(
        _pass_flow_map(player_passes, f"{match_player}: Progressive, Lateral and Regressive Pass Flow", max_passes=map_limit),
        width="stretch",
    )

with tabs[1]:
    pa.section_heading("Pass Direction and Accuracy")
    col_chart, col_table = st.columns([1.15, 1])
    with col_chart:
        st.plotly_chart(_direction_accuracy_chart(player_passes, f"{match_player}: Pass Direction and Completion"), width="stretch")
    with col_table:
        direction_table = _direction_accuracy_table(player_passes)
        if direction_table.empty:
            st.caption("No direction split is available for the selected filters.")
        else:
            st.dataframe(direction_table, width="stretch", hide_index=True)

    pa.section_heading("Receiver Breakdown")
    receiver_table = _receiver_summary(player_passes)
    if receiver_table.empty:
        st.caption("No receiver-level passing rows are available for the selected player.")
    else:
        st.dataframe(receiver_table, width="stretch", hide_index=True)

with tabs[2]:
    pa.section_heading(f"{team_name} Crossing Profile")
    team_cross_summary = _crossing_summary(filtered_passes)
    cross_cols = st.columns(4)
    cross_cols[0].metric("Crosses Attempted", charting.metric_text(team_cross_summary["Attempts"], "Actions"))
    cross_cols[1].metric("Completion", f"{team_cross_summary['Completion %']:.1f}%")
    cross_cols[2].metric("Low / High Split", f"{team_cross_summary['Low Crosses']} / {team_cross_summary['High Crosses']}")
    cross_cols[3].metric("Total PXT From Crosses", charting.metric_text(team_cross_summary["PXT Pass"], "PXT Pass"))

    map_col, chart_col = st.columns([1.15, 1])
    with map_col:
        team_crosses = filtered_passes[data.is_cross(filtered_passes)].copy()
        st.plotly_chart(
            pitch.pass_map(team_crosses, team_name, f"{team_name}: Cross Delivery Map", max_passes=300),
            width="stretch",
        )
    with chart_col:
        st.plotly_chart(_crossing_type_chart(filtered_passes, f"{team_name}: Cross Type and Outcome"), width="stretch")

    pa.section_heading("Crossing by Player")
    cross_table = _crossing_by_player_table(filtered_passes)
    if cross_table.empty:
        st.caption("No crosses match the current filters for this fixture.")
    else:
        st.dataframe(cross_table, width="stretch", hide_index=True)

    pa.section_heading(f"{match_player}: Crossing Detail")
    player_cross_summary = _crossing_summary(player_passes)
    if player_cross_summary["Attempts"] == 0:
        st.caption(f"{match_player} did not attempt a cross in this selected fixture (within the current filters).")
    else:
        detail_cols = st.columns(3)
        detail_cols[0].metric("Attempted", charting.metric_text(player_cross_summary["Attempts"], "Actions"))
        detail_cols[1].metric("Completed", charting.metric_text(player_cross_summary["Completed"], "Actions"))
        detail_cols[2].metric("PXT From Crosses", charting.metric_text(player_cross_summary["PXT Pass"], "PXT Pass"))
        player_crosses = player_passes[data.is_cross(player_passes)].copy()
        cross_event_cols = ma.available_columns(
            player_crosses,
            ["Minute", "Receiver", "Action", "Outcome", "Pass Distance", "PXT Pass", "Start X", "Start Y", "End X", "End Y"],
        )
        st.dataframe(player_crosses[cross_event_cols].sort_values("Minute"), width="stretch", hide_index=True)

with tabs[3]:
    pa.section_heading("Selected Player Passing Network")
    network = pass_network_rows[
        (pass_network_rows["Player"].astype(str) == str(match_player))
        | (pass_network_rows["Receiver"].astype(str) == str(match_player))
    ].copy()
    network_source = "completed CAFC_DB Impect pass events for the selected fixture"
    if network.empty:
        completed_links = filtered_passes[
            (filtered_passes["Outcome"] == "Complete")
            & (
                (filtered_passes["Player"].astype(str) == str(match_player))
                | (filtered_passes["Receiver"].astype(str) == str(match_player))
            )
        ].copy()
        network = _network_from_pass_events(completed_links)
        network_source = "completed mapped pass events from the current filters"
    if network.empty:
        st.info("No completed player-to-receiver links are available for this selected player and filter set.")
    else:
        st.caption(
            f"Source: {network_source}. Link counts and average locations are derived from the underlying event rows."
        )
        max_count = int(pd.to_numeric(network["Pass Count"], errors="coerce").max()) if not network.empty else 2
        min_link = st.slider("Minimum Completed Passes per Link", 1, max(max_count, 2), min(3, max(max_count, 1)))
        st.plotly_chart(
            pitch.passing_network(
                network,
                team_name,
                f"{match_player}: Completed Pass Network",
                min_passes=min_link,
                use_pitch_image=True,
            ),
            width="stretch",
        )
        st.dataframe(
            network.sort_values("Pass Count", ascending=False),
            width="stretch",
            hide_index=True,
        )

with tabs[4]:
    pa.section_heading("Passing Matrices")
    completed_team_passes = filtered_passes[filtered_passes["Outcome"] == "Complete"].copy()
    matrix_size = st.slider("Players in Receiver Matrix", 6, 18, 12)
    matrix_height = min(max(600, matrix_size * 32 + 220), 780)
    matrix_cols = st.columns(2)
    with matrix_cols[0]:
        st.plotly_chart(
            _passing_matrix_heatmap(
                completed_team_passes,
                selected_player=match_player,
                title=f"{team_name}: Completed Pass Matrix",
                max_players=matrix_size,
                height=matrix_height,
            ),
            width="stretch",
        )
    with matrix_cols[1]:
        st.plotly_chart(
            _lane_matrix_heatmap(player_passes, f"{match_player}: Lane Movement Matrix", height=matrix_height),
            width="stretch",
        )

with tabs[5]:
    pa.section_heading("Mapped Pass Event Table")
    event_cols = ma.available_columns(
        player_passes,
        [
            "Minute",
            "Player",
            "Receiver",
            "Action",
            "Outcome",
            "Pass Direction",
            "Pass Distance",
            "Territory Gain",
            "PXT Pass",
            "Start Lane",
            "End Lane",
            "Start X",
            "Start Y",
            "End X",
            "End Y",
        ],
    )
    if player_passes.empty:
        st.caption("No mapped pass events match the current selected player and filters.")
    else:
        st.dataframe(player_passes[event_cols].sort_values(["Minute", "Receiver"]), width="stretch", hide_index=True)

pa.section_heading("Season Passing Impact Table")
if players.empty or "Passing Impact" not in players:
    st.caption("No season passing impact table is available for the selected player season.")
else:
    table_cols = [
        col
        for col in [
            "Player",
            "Team",
            "Position",
            "Minutes",
            "Pass %",
            "Successful Passes /90",
            "Passes to Final 3rd /90",
            "Pass Progression /90",
            "Bypassed Opponents /90",
            "Passing Impact",
        ]
        if col in players.columns
    ]
    st.dataframe(players[table_cols].sort_values("Passing Impact", ascending=False), width="stretch", hide_index=True)
