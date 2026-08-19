# =============================================================================
# PLAYER SCATTER GRAPHS - compare players on two metrics
# =============================================================================
import importlib
import inspect

import pandas as pd
import streamlit as st

from utils import player_analysis as pa


if "highlight_players" not in inspect.signature(pa.metric_scatter).parameters:
    pa = importlib.reload(pa)


def _with_age(players: pd.DataFrame) -> pd.DataFrame:
    out = players.copy()
    if "Birthdate" not in out.columns:
        return out
    birthdates = pd.to_datetime(out["Birthdate"], errors="coerce")
    ages = ((pd.Timestamp.today().normalize() - birthdates).dt.days / 365.25).round(1)
    out["Age"] = ages
    out["_Age"] = ages
    return out


def _sorted_unique(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).unique().tolist(), key=str.casefold)


def _range_label(label: str, value_range: tuple[int, int] | None) -> str | None:
    if value_range is None:
        return None
    return f"{label}: {value_range[0]:,}-{value_range[1]:,}"


pa.page_header(
    "Player Scatter Graphs",
    "Plot players across two selected metrics to spot profiles, outliers and clusters.",
)

season = pa.select_season(key="player_scatter_season")
players = _with_age(pa.load_player_data(season))
if players.empty:
    st.warning("No players are available for the selected season.")
    st.stop()

metrics = pa.scatter_metric_columns(players)
if len(metrics) < 2:
    st.warning("At least two numeric player metrics are required for the scatter graph.")
    st.stop()

default_x = metrics.index("xG /90") if "xG /90" in metrics else 0
default_y = metrics.index("Assists /90") if "Assists /90" in metrics else min(1, len(metrics) - 1)

pa.section_heading("Scatter controls")
st.caption("Filter the player pool first, then choose how to highlight players inside that cohort. Leave team or position filters blank to include all.")

with st.expander("Filters", expanded=True):
    metric_cols = st.columns(3)
    metric_view = metric_cols[0].selectbox(
        "Metric view",
        ["Metric values", "Percentile ranks"],
        key="player_scatter_metric_view",
    )
    x_metric = metric_cols[1].selectbox("X axis metric", metrics, index=default_x, key="player_scatter_x")
    y_metric = metric_cols[2].selectbox("Y axis metric", metrics, index=default_y, key="player_scatter_y")

    filter_cols = st.columns(3)
    team_options = _sorted_unique(players["Team"]) if "Team" in players.columns else []
    selected_teams = filter_cols[0].multiselect("Teams", team_options, default=[], key="player_scatter_team_filter")

    position_options = _sorted_unique(players["_Position Display"]) if "_Position Display" in players.columns else []
    selected_positions = filter_cols[1].multiselect("Positions", position_options, default=[], key="player_scatter_position_filter")

    color_options = ["None"]
    for option in ["Minutes", "Age", *metrics]:
        if option in players.columns and option not in color_options:
            color_options.append(option)
    default_color = color_options.index("Minutes") if "Minutes" in color_options else 0
    color_metric_label = filter_cols[2].selectbox("Colour by", color_options, index=default_color, key="player_scatter_color")
    color_metric = None if color_metric_label == "None" else color_metric_label

    range_cols = st.columns(3)
    minutes_range = None
    if "Minutes" in players.columns and pd.to_numeric(players["Minutes"], errors="coerce").notna().any():
        minutes_values = pd.to_numeric(players["Minutes"], errors="coerce").dropna()
        min_minutes = int(max(0, minutes_values.min()))
        max_minutes = int(minutes_values.max())
        if min_minutes < max_minutes:
            minutes_range = range_cols[0].slider(
                "Minutes range",
                min_value=min_minutes,
                max_value=max_minutes,
                value=(min_minutes, max_minutes),
                step=50,
                key="player_scatter_minutes_range",
            )
        else:
            range_cols[0].caption(f"Minutes: {min_minutes:,}")

    age_range = None
    if "Age" in players.columns and pd.to_numeric(players["Age"], errors="coerce").notna().any():
        age_values = pd.to_numeric(players["Age"], errors="coerce").dropna()
        min_age = int(age_values.min())
        max_age = int(age_values.max() + 0.999)
        if min_age < max_age:
            age_range = range_cols[1].slider(
                "Age range",
                min_value=min_age,
                max_value=max_age,
                value=(min_age, max_age),
                step=1,
                key="player_scatter_age_range",
            )
        else:
            range_cols[1].caption(f"Age: {min_age}")

    size_options = ["None"]
    for option in ["Minutes", *metrics]:
        if option in players.columns and option not in size_options:
            size_options.append(option)
    size_metric = range_cols[2].selectbox("Bubble size", size_options, index=0, key="player_scatter_size")

filtered = players.copy()
if selected_teams and "Team" in filtered.columns:
    filtered = filtered[filtered["Team"].astype(str).isin(selected_teams)]
if selected_positions and "_Position Display" in filtered.columns:
    filtered = filtered[filtered["_Position Display"].astype(str).isin(selected_positions)]
if minutes_range is not None and "Minutes" in filtered.columns:
    minutes_values = pd.to_numeric(filtered["Minutes"], errors="coerce")
    filtered = filtered[minutes_values.between(minutes_range[0], minutes_range[1], inclusive="both")]
if age_range is not None and "Age" in filtered.columns:
    age_values = pd.to_numeric(filtered["Age"], errors="coerce")
    filtered = filtered[age_values.between(age_range[0], age_range[1], inclusive="both")]

if filtered.empty:
    st.warning("No players match the selected filters.")
    st.stop()

with st.expander("Highlighting and display", expanded=True):
    top_cols = st.columns(5)
    top_n = top_cols[0].slider("Top N", min_value=3, max_value=25, value=10, step=1, key="player_scatter_top_n")
    highlight_top_x = top_cols[1].checkbox("Top X axis", key="player_scatter_top_x")
    highlight_top_y = top_cols[2].checkbox("Top Y axis", key="player_scatter_top_y")
    has_age = "Age" in filtered.columns and pd.to_numeric(filtered["Age"], errors="coerce").notna().any()
    highlight_u21 = top_cols[3].checkbox("U21 players", disabled=not has_age, key="player_scatter_u21")
    highlight_u19 = top_cols[4].checkbox("U19 players", disabled=not has_age, key="player_scatter_u19")

    highlight_cols = st.columns(2)
    player_options = _sorted_unique(filtered["Player"]) if "Player" in filtered.columns else []
    previous_player = st.session_state.get("selected_player")
    default_players = [previous_player] if previous_player in player_options else []
    highlight_players = highlight_cols[0].multiselect(
        "Highlight player(s)",
        player_options,
        default=default_players,
        key="player_scatter_highlight_players",
    )
    if highlight_players:
        st.session_state["selected_player"] = highlight_players[0]

    filtered_team_options = _sorted_unique(filtered["Team"]) if "Team" in filtered.columns else []
    highlight_teams = highlight_cols[1].multiselect(
        "Highlight team(s)",
        filtered_team_options,
        default=[],
        key="player_scatter_highlight_teams",
    )

    display_cols = st.columns(2)
    label_highlights = display_cols[0].checkbox("Label highlighted players", value=True, key="player_scatter_label_highlights")
    show_median_lines = display_cols[1].checkbox("Show median lines", value=True, key="player_scatter_median_lines")

chart_players = filtered.copy()
x_axis = x_metric
y_axis = y_metric
title_x = x_metric
title_y = y_metric
if metric_view == "Percentile ranks":
    x_axis = f"{x_metric} Percentile"
    y_axis = f"{y_metric} Percentile"
    x_higher = pa._metric_meta(x_metric)[1]
    y_higher = pa._metric_meta(y_metric)[1]
    chart_players[x_axis] = pa.percentile(chart_players[x_metric], higher_is_better=x_higher)
    chart_players[y_axis] = pa.percentile(chart_players[y_metric], higher_is_better=y_higher)
    title_x = f"{x_metric} percentile"
    title_y = f"{y_metric} percentile"

chart_players = chart_players.dropna(subset=[x_axis, y_axis])
if chart_players.empty:
    st.warning("No players have values for both selected axis metrics after filtering.")
    st.stop()

summary_parts = [f"{len(chart_players):,} players shown"]
if selected_teams:
    summary_parts.append(f"Teams: {', '.join(selected_teams[:4])}{' +' + str(len(selected_teams) - 4) if len(selected_teams) > 4 else ''}")
if selected_positions:
    summary_parts.append(f"Positions: {', '.join(selected_positions[:5])}{' +' + str(len(selected_positions) - 5) if len(selected_positions) > 5 else ''}")
for value in [_range_label("Minutes", minutes_range), _range_label("Age", age_range)]:
    if value:
        summary_parts.append(value)
summary_parts.append(metric_view.lower())

pa.section_heading("Metric scatter")
pa.chart_title(f"{title_x} vs {title_y}")
st.caption(" | ".join(summary_parts))
st.plotly_chart(
    pa.metric_scatter(
        chart_players,
        x=x_axis,
        y=y_axis,
        selected=highlight_players[0] if highlight_players else None,
        size=None if size_metric == "None" else size_metric,
        highlight_players=highlight_players,
        highlight_teams=highlight_teams,
        highlight_top_x=highlight_top_x,
        highlight_top_y=highlight_top_y,
        top_n=top_n,
        highlight_u21=highlight_u21,
        highlight_u19=highlight_u19,
        color_metric=color_metric,
        label_highlights=label_highlights,
        show_median_lines=show_median_lines,
        show_title=False,
    ),
    width="stretch",
)

st.caption("Hover a marker for player, team, position, minutes and exact metric values. Highlighting changes labels and marker emphasis without removing the rest of the cohort.")
