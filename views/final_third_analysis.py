# =============================================================================
# FINAL THIRD ANALYSIS - event-level entry quantity and quality
# =============================================================================
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charting, data, final_third as ft, match_analysis as ma, pitch, team_analysis as ta

FINAL_THIRD_SOURCE = (
    "Final-third analysis uses CAFC_DB Impect provider events: adjusted start/end locations, "
    "lanes, outcomes, receivers, PXT Pass, Team xT and shot sequence fields. It does not use aggregated "
    "player/team average tables."
)

LANE_ORDER = [
    "LEFT_WING",
    "LEFT_HALF_SPACE",
    "CENTER",
    "RIGHT_HALF_SPACE",
    "RIGHT_WING",
    "Unknown",
]
OUTCOME_COLORS = {
    "Successful": "#16a34a",
    "Unsuccessful": "#dc2626",
    "Other": "#98a2b3",
}

SUCCESS_RATE_COLOR_MIN = 25
SUCCESS_RATE_COLOR_MAX = 65
FINAL_THIRD_ENTRY_AXIS_MAX = 100
PENALTY_BOX_ENTRY_AXIS_MAX = 40
FINAL_THIRD_QUALITY_AXIS_MAX = 12
PENALTY_BOX_QUALITY_AXIS_MAX = 25


def _visual_key(items: list[tuple[str, str, str]]) -> None:
    key_items: list[str] = ['<span style="font-weight:600; color:#475467;">Key:</span>']

    for marker_type, color, label in items:
        safe_color = escape(str(color), quote=True)
        safe_label = escape(str(label))

        if marker_type == "dash":
            marker = (
                f'<span style="width:34px; border-top:2px dashed {safe_color}; '
                'display:inline-block;"></span>'
            )
        elif marker_type == "circle":
            marker = (
                f'<span style="width:11px; height:11px; border-radius:50%; '
                f'background:{safe_color}; display:inline-block; '
                'border:1px solid rgba(0,0,0,0.12);"></span>'
            )
        else:
            marker = (
                f'<span style="width:14px; height:11px; border-radius:3px; '
                f'background:{safe_color}; display:inline-block; '
                'border:1px solid rgba(0,0,0,0.12);"></span>'
            )

        key_items.append(
            '<span style="display:inline-flex; align-items:center; gap:6px; margin-right:16px;">'
            f"{marker}<span>{safe_label}</span>"
            "</span>"
        )

    st.markdown(
        '<div style="display:flex; flex-wrap:wrap; align-items:center; gap:8px 2px; '
        'font-size:0.9rem; color:#475467; margin:0.1rem 0 0.45rem 0;">'
        + "".join(key_items)
        + "</div>",
        unsafe_allow_html=True,
    )


def _team_options(matches: pd.DataFrame) -> list[str]:
    team_columns = [column for column in ["Home", "Away"] if column in matches]
    if not team_columns:
        return []

    values = matches[team_columns].to_numpy().ravel()
    teams = sorted({str(value) for value in values if pd.notna(value) and str(value).strip()})
    return teams


def _team_default_index(teams: list[str]) -> int:
    for index, team in enumerate(teams):
        if str(team).lower() == "charlton" or "charlton" in str(team).lower():
            return index
    return 0


def _team_matches(matches: pd.DataFrame, team_name: str) -> pd.DataFrame:
    home = matches["Home"].astype(str).eq(str(team_name)) if "Home" in matches else pd.Series(False, index=matches.index)
    away = matches["Away"].astype(str).eq(str(team_name)) if "Away" in matches else pd.Series(False, index=matches.index)
    return matches[home | away].copy()


def _linked_entry_sequence_shots(events: pd.DataFrame, entries: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"MatchId", "Sequence Index"}
    if events.empty or entries.empty or not required_columns.issubset(events.columns) or not required_columns.issubset(entries.columns):
        return pd.DataFrame(columns=events.columns)

    entry_sequences = entries.dropna(subset=["MatchId", "Sequence Index"])[["MatchId", "Sequence Index"]].drop_duplicates()
    if entry_sequences.empty:
        return pd.DataFrame(columns=events.columns)

    action_type = events.get("Action Type", pd.Series("", index=events.index)).astype(str).str.upper()
    shots = events[action_type.eq("SHOT")].copy()
    if shots.empty:
        return shots

    linked = shots.merge(entry_sequences, on=["MatchId", "Sequence Index"], how="inner")
    if {"MatchId", "Event Number"}.issubset(linked.columns):
        linked = linked.drop_duplicates(subset=["MatchId", "Event Number"])
    return linked


def _goal_count(shots: pd.DataFrame) -> int:
    if shots.empty:
        return 0

    result_goal = shots.get("Result", pd.Series("", index=shots.index)).astype(str).str.upper().eq("SUCCESS")
    action_goal = shots.get("Action", pd.Series("", index=shots.index)).astype(str).str.upper().str.contains("GOAL", na=False)
    return int((result_goal | action_goal).sum())


def _match_entry_summary(entries: pd.DataFrame, team_matches: pd.DataFrame) -> pd.DataFrame:
    if team_matches.empty or "MatchId" not in team_matches:
        return pd.DataFrame()

    base = team_matches[["MatchId"]].copy()
    base["_Match Key"] = base["MatchId"].astype(str)
    base["Match Label"] = team_matches.apply(ma.match_label, axis=1)
    if "Date" in team_matches:
        base["Date"] = pd.to_datetime(team_matches["Date"], errors="coerce")

    if entries.empty:
        base["Entries"] = 0
        base["Successful"] = 0
        base["Success %"] = 0.0
        base["Entry Value"] = 0.0
        base["Avg Entry Value"] = 0.0
        return base

    values = entries.copy()
    values["_Match Key"] = values["MatchId"].astype(str)
    values["_Entry Value"] = pd.to_numeric(values["_Entry Value"], errors="coerce").fillna(0)
    values["_Successful"] = values["_Outcome"].astype(str).eq("Successful") if "_Outcome" in values else False

    summary = values.groupby("_Match Key", as_index=False).agg(
        Entries=("_Match Key", "size"),
        Successful=("_Successful", "sum"),
        **{
            "Entry Value": ("_Entry Value", "sum"),
            "Avg Entry Value": ("_Entry Value", "mean"),
        },
    )
    summary["Success %"] = summary["Successful"] / summary["Entries"].replace(0, pd.NA) * 100

    out = base.merge(summary, on="_Match Key", how="left").drop(columns=["_Match Key"])
    for column in ["Entries", "Successful"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)
    for column in ["Success %", "Entry Value", "Avg Entry Value"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0)
    return out


def _quantity_quality_chart(match_summary: pd.DataFrame, team_name: str, zone_label: str) -> go.Figure:
    fig = go.Figure()
    title = f"{team_name}: {zone_label} Quantity vs Quality"

    plot_df = match_summary.copy()
    plot_df["Entries"] = pd.to_numeric(plot_df["Entries"], errors="coerce").fillna(0)
    plot_df["Avg Entry Value"] = pd.to_numeric(plot_df["Avg Entry Value"], errors="coerce").fillna(0)
    plot_df["Entry Value"] = pd.to_numeric(plot_df["Entry Value"], errors="coerce").fillna(0)
    plot_df["Success %"] = pd.to_numeric(plot_df["Success %"], errors="coerce").fillna(0)
    plot_df = plot_df[plot_df["Entries"] > 0].copy()
    plot_df["Quality Score"] = plot_df["Avg Entry Value"] * 1000

    if plot_df.empty:
        fig.add_annotation(
            text="No entry data",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return charting.polish_figure(fig, title, height=500)

    average_entries = plot_df["Entries"].mean()
    average_quality = plot_df["Quality Score"].mean()

    max_entry_value = max(float(plot_df["Entry Value"].max()), 0.01)
    plot_df["_Bubble Size"] = 8 + (plot_df["Entry Value"] / max_entry_value) * 20

    fig.add_trace(
        go.Scatter(
            x=plot_df["Entries"],
            y=plot_df["Quality Score"],
            mode="markers",
            marker=dict(
                size=plot_df["_Bubble Size"],
                color=plot_df["Success %"],
                cmin=SUCCESS_RATE_COLOR_MIN,
                cmax=SUCCESS_RATE_COLOR_MAX,
                colorscale=[
                    [0.0, "#dc2626"],
                    [0.5, "#f59e0b"],
                    [1.0, "#16a34a"],
                ],
                colorbar=dict(
                    title="Success %",
                    tickvals=[25, 35, 45, 55, 65],
                    ticktext=["<=25", "35", "45", "55", ">=65"],
                ),
                line=dict(color="#ffffff", width=1.2),
            ),
            customdata=plot_df[["Match Label", "Avg Entry Value", "Entry Value", "Success %", "Successful"]].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[0]}</b>"
                "<br>Entries: %{x:.0f}"
                "<br>Quality Score: %{y:.2f}"
                "<br>Average Entry Value: %{customdata[1]:.4f}"
                "<br>Total Entry Value: %{customdata[2]:.2f}"
                "<br>Successful Entries: %{customdata[4]:.0f}"
                "<br>Success Rate: %{customdata[3]:.1f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vline(
        x=average_entries,
        line=dict(color="#667085", width=1.4, dash="dash"),
    )

    fig.add_hline(
        y=average_quality,
        line=dict(color="#667085", width=1.4, dash="dash"),
    )

    fig.update_layout(
        xaxis_title="Entries",
        yaxis_title="Quality Score",
        showlegend=False,
    )

    if zone_label == "Penalty Box":
        x_axis_max = PENALTY_BOX_ENTRY_AXIS_MAX
        y_axis_max = PENALTY_BOX_QUALITY_AXIS_MAX
    else:
        x_axis_max = FINAL_THIRD_ENTRY_AXIS_MAX
        y_axis_max = FINAL_THIRD_QUALITY_AXIS_MAX

    x_axis_max = max(x_axis_max, float(plot_df["Entries"].max()) * 1.08)
    y_axis_max = max(y_axis_max, float(plot_df["Quality Score"].max()) * 1.12)

    fig.update_xaxes(range=[0, x_axis_max], gridcolor="#e8edf3", zeroline=False)
    fig.update_yaxes(range=[0, y_axis_max], gridcolor="#e8edf3", zeroline=False)

    return charting.polish_figure(fig, title, height=500)


def _outcome_chart(outcome_summary: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if outcome_summary.empty:
        fig.add_annotation(text="No outcome data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return charting.polish_figure(fig, title, height=360)

    plot_df = outcome_summary.copy()
    plot_df["Color"] = plot_df["Outcome"].map(OUTCOME_COLORS).fillna("#98a2b3")
    plot_df = plot_df.sort_values("Entries", ascending=True)

    fig.add_trace(
        go.Bar(
            x=plot_df["Entries"],
            y=plot_df["Outcome"],
            orientation="h",
            marker=dict(color=plot_df["Color"]),
            text=plot_df["Entries"],
            textposition="outside",
            cliponaxis=False,
            customdata=plot_df[["Entry Value", "Avg Entry Value"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b>"
                "<br>Entries: %{x:.0f}"
                "<br>Entry Value: %{customdata[0]:.2f}"
                "<br>Avg Entry Value: %{customdata[1]:.3f}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(xaxis_title="Entries", yaxis_title="", showlegend=False)
    return charting.polish_figure(fig, title, height=360)


def _funnel_chart(entries: pd.DataFrame, linked_shots: pd.DataFrame, title: str) -> go.Figure:
    successful = int(entries["_Outcome"].astype(str).eq("Successful").sum()) if "_Outcome" in entries else 0
    goals = _goal_count(linked_shots)

    fig = go.Figure(
        go.Funnel(
            y=["Entries", "Successful Entries", "Shots From Entry Sequences", "Goals From Entry Sequences"],
            x=[len(entries), successful, len(linked_shots), goals],
            marker=dict(color=[pitch.DARK, pitch.GREEN, pitch.GOLD, pitch.RED]),
            textinfo="value+percent initial",
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    return charting.polish_figure(fig, title, height=360)


def _lane_heatmap(lane_summary: pd.DataFrame, value_column: str, title: str) -> go.Figure:
    fig = go.Figure()
    if lane_summary.empty:
        fig.add_annotation(text="No lane data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return charting.polish_figure(fig, title, height=430)

    lanes = [lane for lane in LANE_ORDER if lane in set(lane_summary["Start Lane"]) or lane in set(lane_summary["End Lane"])]
    if not lanes:
        lanes = LANE_ORDER

    pivot = lane_summary.pivot_table(
        index="Start Lane",
        columns="End Lane",
        values=value_column,
        aggfunc="sum",
        fill_value=0,
    ).reindex(index=lanes, columns=lanes, fill_value=0)

    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=[
                [0.0, "#fff1f3"],
                [0.5, "#ef8890"],
                [1.0, pitch.RED],
            ],
            text=pivot.round(2).values,
            texttemplate="%{text}",
            hovertemplate="Start Lane: %{y}<br>End Lane: %{x}<br>" + value_column + ": %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="End Lane", yaxis_title="Start Lane")
    return charting.polish_figure(fig, title, height=430)

def _empty_lane_effectiveness() -> pd.DataFrame:
    lanes = [lane for lane in LANE_ORDER if lane != "Unknown"]

    return pd.DataFrame(
        {
            "Lane": lanes,
            "Entries": [0] * len(lanes),
            "Successful": [0] * len(lanes),
            "Success %": [0.0] * len(lanes),
            "Entry Value": [0.0] * len(lanes),
            "Avg Entry Value": [0.0] * len(lanes),
        }
    )


def _lane_effectiveness_summary(
    entries: pd.DataFrame,
    lane_column: str,
) -> pd.DataFrame:
    lanes = [lane for lane in LANE_ORDER if lane != "Unknown"]

    if entries.empty or lane_column not in entries.columns:
        return _empty_lane_effectiveness()

    values = entries.copy()

    values["Lane"] = (
        values[lane_column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    values["Lane"] = values["Lane"].replace(
        {
            "LEFT": "LEFT_WING",
            "HALF_LEFT": "LEFT_HALF_SPACE",
            "CENTRE": "CENTER",
            "HALF_RIGHT": "RIGHT_HALF_SPACE",
            "RIGHT": "RIGHT_WING",
        }
    )

    values = values[values["Lane"].isin(lanes)].copy()

    if values.empty:
        return _empty_lane_effectiveness()

    if "_Entry Value" not in values.columns:
        values["_Entry Value"] = ft.entry_value(values)

    values["_Entry Value"] = pd.to_numeric(
        values["_Entry Value"],
        errors="coerce",
    ).fillna(0)

    if "_Outcome" not in values.columns:
        results = (
            values["Result"]
            if "Result" in values.columns
            else pd.Series("", index=values.index)
        )

        values["_Outcome"] = ft.result_status(results)

    values["_Successful"] = (
        values["_Outcome"]
        .astype(str)
        .eq("Successful")
    )

    summary = values.groupby("Lane", as_index=False).agg(
        Entries=("Lane", "size"),
        Successful=("_Successful", "sum"),
        **{
            "Entry Value": ("_Entry Value", "sum"),
            "Avg Entry Value": ("_Entry Value", "mean"),
        },
    )

    summary = (
        summary
        .set_index("Lane")
        .reindex(lanes)
        .reset_index()
    )

    summary[["Entries", "Successful"]] = (
        summary[["Entries", "Successful"]]
        .fillna(0)
        .astype(int)
    )

    summary[["Entry Value", "Avg Entry Value"]] = (
        summary[["Entry Value", "Avg Entry Value"]]
        .fillna(0)
    )

    valid_entry_counts = summary["Entries"].where(
        summary["Entries"].gt(0)
    )

    summary["Success %"] = (
        summary["Successful"]
        .div(valid_entry_counts)
        .mul(100)
        .fillna(0)
    )

    for column in [
        "Success %",
        "Entry Value",
        "Avg Entry Value",
    ]:
        summary[column] = pd.to_numeric(
            summary[column],
            errors="coerce",
        ).round(2)

    return summary


def _success_fill_color(entries_count: int, success_pct: float) -> str:
    if entries_count <= 0:
        return "rgba(152, 162, 179, 0.22)"

    if success_pct >= 55:
        return "rgba(22, 163, 74, 0.72)"

    if success_pct >= 40:
        return "rgba(245, 158, 11, 0.68)"

    return "rgba(220, 38, 38, 0.62)"


def _lane_pitch_map(lane_summary: pd.DataFrame, title: str, selected_zone_label: str) -> go.Figure:
    fig = pitch.half_pitch_vertical_figure(title, height=660, legend=True)

    if lane_summary.empty:
        fig.add_annotation(text="No lane data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig

    zone_start = pitch.PENALTY_BOX_X if selected_zone_label == "Penalty Box" else pitch.FINAL_THIRD_X
    zone_end = pitch.PITCH_X_MAX

    left_to_right_lanes = [
        lane
        for lane in LANE_ORDER
        if lane != "Unknown"
    ]
    lane_width = (pitch.PITCH_Y_MAX - pitch.PITCH_Y_MIN) / len(left_to_right_lanes)
    lane_lookup = lane_summary.set_index("Lane").to_dict("index")

    legend_items = [
        ("High Effectiveness: 55%+ Success", "#16a34a"),
        ("Medium Effectiveness: 40-54% Success", "#f59e0b"),
        ("Low Effectiveness: Below 40% Success", "#dc2626"),
        ("No Entries", "#98a2b3"),
    ]
    for label, color in legend_items:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=label,
                line=dict(color=color, width=8),
                hoverinfo="skip",
            )
        )

    fig.add_shape(
        type="rect",
        x0=pitch.PITCH_Y_MIN,
        y0=zone_start,
        x1=pitch.PITCH_Y_MAX,
        y1=zone_end,
        line=dict(color="rgba(195, 0, 23, 0.35)", width=2),
        fillcolor="rgba(195, 0, 23, 0.04)",
        layer="below",
    )

    for index, lane in enumerate(left_to_right_lanes):
        x0 = pitch.PITCH_Y_MIN + index * lane_width
        x1 = x0 + lane_width
        record = lane_lookup.get(lane, {})

        entries_count = int(record.get("Entries", 0))
        success_pct = float(record.get("Success %", 0))
        entry_value = float(record.get("Entry Value", 0))
        avg_entry_value = float(record.get("Avg Entry Value", 0))
        lane_label = lane.replace("_", " ").title().replace("Half ", "Half<br>")

        fig.add_shape(
            type="rect",
            x0=x0,
            y0=zone_start,
            x1=x1,
            y1=zone_end,
            line=dict(color="rgba(255, 255, 255, 0.9)", width=2),
            fillcolor=_success_fill_color(entries_count, success_pct),
            layer="above",
        )

        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=(zone_start + zone_end) / 2,
            text=(
                f"<b>{lane_label}</b><br>"
                f"{entries_count} entries<br>"
                f"{success_pct:.0f}% success<br>"
                f"{avg_entry_value:.3f} avg EV"
            ),
            showarrow=False,
            align="center",
            font=dict(size=12, color="#101828"),
            bgcolor="rgba(255, 255, 255, 0.82)",
            bordercolor="rgba(16, 24, 40, 0.16)",
            borderwidth=1,
            borderpad=4,
            hovertext=(
                f"{lane_label}<br>"
                f"Entries: {entries_count}<br>"
                f"Successful: {int(record.get('Successful', 0))}<br>"
                f"Success %: {success_pct:.1f}%<br>"
                f"Entry Value: {entry_value:.2f}<br>"
                f"Avg Entry Value: {avg_entry_value:.3f}"
            ),
        )

    fig.add_annotation(
        x=0,
        y=zone_end + 2,
        text="<b>Attacking direction</b>",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#475467",
        ax=0,
        ay=45,
        font=dict(size=12, color="#475467"),
        bgcolor="rgba(255, 255, 255, 0.85)",
        borderpad=4,
    )

    return fig


def _player_value_chart(player_summary: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if player_summary.empty:
        fig.add_annotation(text="No player entry data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return charting.polish_figure(fig, title, height=420)

    values = player_summary.copy()

    for column in ["Entries", "Entry Value", "Avg Entry Value"]:
        values[column] = pd.to_numeric(
            values[column],
            errors="coerce",
        ).fillna(0)

    values = values.sort_values(
        ["Entry Value", "Entries"],
        ascending=[False, False],
    )

    plot_df = (
        values
        .head(12)
        .sort_values("Entry Value", ascending=True)
    )

    benchmark_pool = values[values["Entries"].ge(10)].copy()

    if benchmark_pool.empty:
        benchmark_pool = values.copy()

    average_total_value = benchmark_pool["Entry Value"].mean()

    total_benchmark_entries = benchmark_pool["Entries"].sum()
    average_value_per_entry = (
        benchmark_pool["Entry Value"].sum()
        / total_benchmark_entries
        if total_benchmark_entries > 0
        else 0.0
    )

    quality_pool = benchmark_pool["Avg Entry Value"]

    lower_quality = quality_pool.quantile(0.10)
    upper_quality = quality_pool.quantile(0.90)

    colour_spread = max(
        average_value_per_entry - lower_quality,
        upper_quality - average_value_per_entry,
        average_value_per_entry * 0.50,
        0.001,
    )
    colour_min = average_value_per_entry - colour_spread
    colour_max = average_value_per_entry + colour_spread

    fig.add_trace(
        go.Bar(
            x=plot_df["Entry Value"],
            y=plot_df["Player"],
            orientation="h",
            marker=dict(
                color=plot_df["Avg Entry Value"],
                colorscale=[
                    [0.0, "#dc2626"],
                    [0.5, "#f59e0b"],
                    [1.0, "#16a34a"],
                ],
                cmin=colour_min,
                cmax=colour_max,
                colorbar=dict(
                    title="Avg EV<br>per entry",
                    tickvals=[
                        colour_min,
                        average_value_per_entry,
                        colour_max,
                    ],
                    ticktext=[
                        "Lower",
                        f"Squad avg<br>{average_value_per_entry:.3f}",
                        "Higher",
                    ],
                    thickness=14,
                    len=0.76,
                ),
                line=dict(
                    color="rgba(16, 24, 40, 0.20)",
                    width=0.6,
                ),
            ),
            text=plot_df["Entry Value"].map(lambda value: f"{value:.2f}"),
            textposition="outside",
            cliponaxis=False,
            customdata=plot_df[["Entries", "Success %", "Avg Entry Value", "Primary Action", "Main Receiver"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b>"
                "<br>Total Entry Value: %{x:.3f}"
                "<br>Entries: %{customdata[0]:.0f}"
                "<br>Success: %{customdata[1]:.1f}%"
                "<br>Avg Entry Value: %{customdata[2]:.4f}"
                "<br>Primary Action: %{customdata[3]}"
                "<br>Main Receiver: %{customdata[4]}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vline(
        x=average_total_value,
        line_dash="dash",
        line_color="#667085",
        line_width=2,
    )

    fig.add_annotation(
        x=average_total_value,
        y=1.04,
        xref="x",
        yref="paper",
        text=f"Average total EV: {average_total_value:.2f}",
        showarrow=False,
        xanchor="left",
        font=dict(size=11, color="#475467"),
    )

    x_max = max(
        float(plot_df["Entry Value"].max()),
        float(average_total_value),
        0.1,
    ) * 1.18

    fig.update_xaxes(range=[0, x_max])
    fig.update_layout(
        xaxis_title="Total Entry Value",
        yaxis_title="",
        showlegend=False,
    )
    return charting.polish_figure(
        fig,
        title,
        height=max(440, 32 * len(plot_df) + 145),
    )


ta.page_header(
    "Final Third Analysis",
    "Analyse final-third entry quantity, quality, location and player contribution from event-level data.",
    FINAL_THIRD_SOURCE,
)

with st.expander("Final Third Analysis Controls", expanded=True):
    season = ma.select_match_season(key="final_third_analysis_season")
    matches = ma.load_matches(season)
    if matches.empty:
        st.warning("No match data is available for this season.")
        st.stop()

    teams = _team_options(matches)
    if not teams:
        st.warning("No teams are available from the selected match data.")
        st.stop()

    control_cols = st.columns([1.1, 1.2, 1.2])
    with control_cols[0]:
        team_name = st.selectbox("Team", teams, index=_team_default_index(teams), key="final_third_analysis_team")

    team_matches_all = _team_matches(matches, team_name)
    if team_matches_all.empty:
        st.warning("No matches are available for the selected team.")
        st.stop()

    match_label_lookup = {str(row["MatchId"]): ma.match_label(row) for _, row in team_matches_all.iterrows() if pd.notna(row.get("MatchId"))}
    match_options = list(match_label_lookup.keys())

    with control_cols[1]:
        zone = st.selectbox("Entry Zone", ft.ZONE_OPTIONS, key="final_third_analysis_zone")
        zone_label = ft.zone_title(zone)

    with control_cols[2]:
        selected_match_ids = st.multiselect(
            "Matches",
            match_options,
            default=match_options,
            format_func=lambda match_id: match_label_lookup.get(str(match_id), str(match_id)),
            key="final_third_analysis_matches",
        )

    if not selected_match_ids:
        st.info("Select at least one match to analyse final-third entries.")
        st.stop()

    # Move data loading outside to keep it from re-running unnecessarily if possible, 
    # though in Streamlit it will run anyway. Let's keep it inside the expander logic flow for clarity.
    
    raw_events = data.load_match_events(
        season=season,
        team=team_name,
        match_ids=selected_match_ids,
        limit=120000,
    )
    if raw_events.empty:
        st.info("No CAFC_DB Impect event rows are available for this selected season and team.")
        st.stop()

    if len(raw_events) >= 120000:
        st.warning("The selected-match event pull reached the 120,000-row cap. Results may be incomplete.")

    selected_match_id_set = {str(match_id) for match_id in selected_match_ids}
    events = raw_events[raw_events["MatchId"].astype(str).isin(selected_match_id_set)].copy()
    team_matches = team_matches_all[team_matches_all["MatchId"].astype(str).isin(selected_match_id_set)].copy()

    spatial = events.dropna(subset=["Start X", "Start Y", "End X", "End Y"]).copy()
    base_entries = ft.prepare_entries(spatial, zone_label)

    entry_types = sorted(base_entries["Action Type"].dropna().astype(str).unique().tolist()) if "Action Type" in base_entries else []
    entry_results = sorted(base_entries["Result"].dropna().astype(str).unique().tolist()) if "Result" in base_entries else []

    filter_cols = st.columns([1.3, 1.3, 1, 1])
    with filter_cols[0]:
        selected_entry_types = st.multiselect(
            "Action Types",
            entry_types,
            default=entry_types,
            key="final_third_analysis_action_types",
        )
    with filter_cols[1]:
        selected_entry_results = st.multiselect(
            "Entry Outcomes",
            entry_results,
            default=entry_results,
            key="final_third_analysis_results",
        )
    with filter_cols[2]:
        min_value = st.number_input("Minimum Entry Value", min_value=0.0, value=0.0, step=0.01)
    with filter_cols[3]:
        map_limit = st.slider("Map Entries", min_value=25, max_value=300, value=150, step=25)

    entries = base_entries.copy()
    if entry_types:
        entries = entries[entries["Action Type"].astype(str).isin(selected_entry_types)]
    if entry_results:
        entries = entries[entries["Result"].astype(str).isin(selected_entry_results)]
    entries = entries[pd.to_numeric(entries["_Entry Value"], errors="coerce").fillna(0) >= min_value].copy()

linked_shots = _linked_entry_sequence_shots(events, entries)
entry_value_total = pd.to_numeric(entries.get("_Entry Value", pd.Series(dtype="float64")), errors="coerce").fillna(0).sum()
entry_value_avg = pd.to_numeric(entries.get("_Entry Value", pd.Series(dtype="float64")), errors="coerce").fillna(0).mean() if not entries.empty else 0
successful_entries = int(entries["_Outcome"].astype(str).eq("Successful").sum()) if "_Outcome" in entries else 0
success_rate = successful_entries / len(entries) * 100 if len(entries) else 0
shots_xg = pd.to_numeric(linked_shots.get("Shot xG", pd.Series(dtype="float64")), errors="coerce").fillna(0).sum()
entries_per_match = len(entries) / max(len(selected_match_ids), 1)

ta.section_heading("Selected Team Entry Snapshot")
summary_cols = st.columns(6)
summary_cols[0].metric("Entries", f"{len(entries):,}", f"{entries_per_match:.1f} Per Match")
summary_cols[1].metric("Successful", f"{successful_entries:,}", f"{success_rate:.1f}%")
summary_cols[2].metric("Entry Value", f"{entry_value_total:.2f}")
summary_cols[3].metric("Avg Entry Value", f"{entry_value_avg:.3f}")
summary_cols[4].metric("Shots From Entry Sequences", f"{len(linked_shots):,}")
summary_cols[5].metric("xG From Entry Sequences", f"{shots_xg:.2f}", f"{_goal_count(linked_shots)} Goals")

st.caption(
    "Shots and xG are linked by MatchId and Sequence Index, so they are a sequence-based proxy for what happened after entries."
)

match_summary = _match_entry_summary(entries, team_matches)
outcomes = ft.outcome_summary(entries)

ta.section_heading("Quantity vs Quality")

_visual_key(
    [
        ("dash", "#667085", "Average Line"),
        ("circle", "#16a34a", "High Success %"),
        ("circle", "#f59e0b", "Mid Success %"),
        ("circle", "#dc2626", "Low Success %"),
    ]
)

st.caption(
    "Each bubble is one match. Further right means more entries. Higher up means better entry quality "
    "(Quality Score = Average Entry Value x 1,000). Larger bubbles show greater total Entry Value. "
    "Success colour uses a fixed 25-65% scale so match colours are comparable."
)

st.plotly_chart(
    _quantity_quality_chart(match_summary, team_name, zone_label),
    width="stretch",
)

with st.expander("Show Match-by-Match Entry Summary"):
    match_table = match_summary.copy()
    for column in ["Success %", "Entry Value", "Avg Entry Value"]:
        if column in match_table:
            match_table[column] = pd.to_numeric(match_table[column], errors="coerce").round(2)
    st.dataframe(match_table, width="stretch", hide_index=True)

ta.section_heading("Entry Outcomes")
_visual_key(
    [
        ("circle", OUTCOME_COLORS["Successful"], "Successful"),
        ("circle", OUTCOME_COLORS["Unsuccessful"], "Unsuccessful"),
        ("circle", OUTCOME_COLORS["Other"], "Other"),
    ]
)

st.plotly_chart(_outcome_chart(outcomes, f"{zone_label} Outcomes"), width="stretch")

ta.section_heading("Entry Conversion Funnel")
st.caption("This shows how many entries remain once linked to success, shots and goals in the same attacking sequence.")
st.plotly_chart(_funnel_chart(entries, linked_shots, "Entry Outcome Funnel"), width="stretch")

ta.section_heading("Entry Map")
if entries.empty:
    st.info(f"No {zone_label.lower()} entries match the current filters.")
else:
    map_entries = entries.sort_values("_Entry Value", ascending=False).head(map_limit).copy()
    _visual_key(
        [
            ("circle", pitch.GREEN, "Successful Entry"),
            ("circle", pitch.RED, "Unsuccessful Entry"),
            ("circle", "#98a2b3", "Other Entry"),
        ]
    )
    st.caption(f"Showing the top {len(map_entries)} entries by Entry Value to keep the season map readable.")
    st.plotly_chart(
        pitch.entry_zone_map(map_entries, team_name, f"{team_name}: {zone_label} Entries", zone=zone_label),
        width="stretch",
    )

ta.section_heading("Lane Profile")

lane_view = st.radio(
    "Lane View",
    ["End Lane", "Start Lane"],
    horizontal=True,
    key="final_third_analysis_lane_view",
)

lane_effectiveness = _lane_effectiveness_summary(entries, lane_view)

st.caption(
    f"This shows {lane_view.lower()} effectiveness. Colour is based on success rate; "
    "the numbers show volume, success rate and average Entry Value."
)
st.plotly_chart(
    _lane_pitch_map(lane_effectiveness, f"{team_name}: {zone_label} Lane Effectiveness", zone_label),
    width="stretch",
)

with st.expander("Show Start-to-End Lane Route Matrix"):
    lane_metric = st.selectbox(
        "Route Matrix Metric",
        ["Entries", "Entry Value", "Avg Entry Value"],
        key="final_third_analysis_lane_metric",
    )
    lane_rows = ft.lane_summary(entries)

    st.caption(f"Darker cells show higher {lane_metric} for that start-lane to end-lane route.")
    st.plotly_chart(_lane_heatmap(lane_rows, lane_metric, f"{zone_label} Start Lane to End Lane"), width="stretch")

ta.section_heading("Player Entry Contributors")
player_summary = ft.player_entry_summary(entries)
st.caption(
    "Bar length shows total Entry Value and the dashed line marks the squad average for players with "
    "10+ entries. Colour shows value per entry: red means lower efficiency and green means higher "
    "efficiency, so total contribution and colour will not always agree."
)
st.plotly_chart(_player_value_chart(player_summary, f"{team_name}: Player Entry Value"), width="stretch")

with st.expander("Show Player Entry Contributor Table"):
    st.dataframe(player_summary, width="stretch", hide_index=True)


with st.expander("Terminology Key"):
    st.markdown(
        """
        - **Entry**: An action that starts outside the selected zone and ends inside it. Shots are excluded.
        - **Entry Value**: The larger positive value from PXT Pass and Team xT.
        - **Average Entry Value**: Entry Value divided by number of entries. This is the quality side of the page.
        - **Quantity vs Quality**: Match-by-match comparison of how often the team enters and how valuable those entries are.
        - **Shots From Entry Sequences**: Shots sharing MatchId and Sequence Index with an entry.
        - **Lane Profile**: Where entries start and where they end by pitch lane.
        - **Data Source**: CAFC_DB Impect provider events, not aggregated player/team average tables.
        """
    )
