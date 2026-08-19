# =============================================================================
# PERCENTILE RANKINGS - league-wide individual style-of-play rankings
# =============================================================================
# The player-equivalent of the team League Rankings page: every player across
# every club in the selected season, ranked on any style metric, with
# percentiles computed either league-wide or within the player's own position
# group so a centre-back isn't judged on a winger's scale.
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charting, ui
from utils import player_analysis as pa


def _profile_figure(profile: pd.DataFrame, player_name: str, basis_label: str) -> go.Figure:
    if profile.empty:
        return charting.polish_figure(go.Figure(), f"{player_name}: style profile", height=340)
    category_order = {category: index for index, category in enumerate(pa.STYLE_METRIC_CATEGORY_ORDER)}
    rows = profile.copy()
    rows["_Category Order"] = rows["Category"].map(category_order).fillna(len(category_order))
    rows = rows.sort_values(["_Category Order", "Percentile"], ascending=[True, False]).reset_index(drop=True)
    rows["Label"] = [
        f"{str(category).upper()} · {charting.wrap_label(metric, width=27, max_lines=2)}"
        for category, metric in zip(rows["Category"], rows["Metric"], strict=False)
    ]
    rows["Value Label"] = [
        f"{pa.metric_value(value, metric)} · #{int(rank)}/{int(pool_size)}"
        for metric, value, rank, pool_size in zip(rows["Metric"], rows["Value"], rows["Rank"], rows["Pool Size"], strict=False)
    ]
    rows["Colour"] = np.select(
        [rows["Percentile"].ge(67), rows["Percentile"].ge(33)],
        ["#16803c", "#d89216"],
        default="#c30017",
    )
    customdata = np.column_stack(
        [
            rows["Metric"],
            [pa.metric_value(value, metric) for metric, value in zip(rows["Metric"], rows["Value"], strict=False)],
            rows["Rank"],
            rows["Pool Size"],
            rows["Direction"],
        ]
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=[100] * len(rows), y=rows["Label"], orientation="h", marker_color="#eef2f6", hoverinfo="skip", showlegend=False)
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
                "Rank: %{customdata[2]} of %{customdata[3]}<br>%{customdata[4]}<extra></extra>"
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
            textfont=dict(color=ui.CHARLTON_BLACK, size=10),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_vline(x=50, line=dict(color="#667085", width=1.3, dash="dash"))
    fig.update_layout(barmode="overlay", bargap=0.32)
    fig.update_xaxes(range=[0, 122], tickvals=[0, 25, 50, 75, 100], title=f"Percentile ({basis_label})")
    fig.update_yaxes(autorange="reversed", title="")
    fig = charting.polish_figure(fig, f"{player_name}: all-metric style profile", height=max(480, len(rows) * 46 + 140))
    fig.update_layout(margin=dict(l=42, r=112, t=78, b=64))
    return fig


def _player_style_profile(pool: pd.DataFrame, player_name: str, groups: dict[str, list[str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if player_name not in pool["Player"].astype(str).unique():
        return pd.DataFrame()
    for category, metrics in groups.items():
        for metric in metrics:
            category_name, higher_is_better, _label = pa.metric_meta(metric)
            values = pd.to_numeric(pool[metric], errors="coerce")
            player_rows = pool[pool["Player"].astype(str) == str(player_name)]
            if player_rows.empty:
                continue
            target_index = player_rows.index[0]
            if pd.isna(values.loc[target_index]):
                continue
            percentile_value = float(values.rank(pct=True, ascending=higher_is_better).loc[target_index] * 100)
            rank_value = int(values.rank(method="min", ascending=not higher_is_better).loc[target_index])
            rows.append(
                {
                    "Category": category_name,
                    "Metric": metric,
                    "Value": float(values.loc[target_index]),
                    "Rank": rank_value,
                    "Pool Size": int(values.notna().sum()),
                    "Percentile": round(percentile_value, 1),
                    "Direction": "Higher is better" if higher_is_better else "Lower is better",
                }
            )
    return pd.DataFrame(rows)


pa.page_header(
    "League Ranking - Individual",
    "Rank every league player on any style metric, then read one player's full style profile against their peers.",
)

season = pa.select_season(key="percentile_rankings_season")
all_players = pa.add_position_groups(pa.load_player_data(season))
if all_players.empty:
    st.warning("No players are available for the selected season.")
    st.stop()

pa.section_heading("Ranking pool")
st.caption("Filters shape both the ranking chart below and the percentile basis for the style profile.")
pool_cols = st.columns([1.0, 1.1, 0.9])
position_options = ["All positions", *sorted(all_players["Role Group"].dropna().astype(str).unique().tolist())]
with pool_cols[0]:
    position_filter = st.selectbox("Position group", position_options, key="percentile_rankings_position")
team_options = sorted(all_players["Team"].dropna().astype(str).unique().tolist()) if "Team" in all_players else []
with pool_cols[1]:
    team_filter = st.selectbox("Team", ["All teams", *team_options], key="percentile_rankings_team")
with pool_cols[2]:
    max_minutes = int(pd.to_numeric(all_players.get("Minutes", pd.Series([0])), errors="coerce").fillna(0).max())
    minimum_minutes = st.slider("Minimum minutes", 0, max(max_minutes, 1), min(270, max_minutes), step=45, key="percentile_rankings_minutes")

pool = all_players.copy()
if position_filter != "All positions":
    pool = pool[pool["Role Group"].astype(str).eq(position_filter)].copy()
if team_filter != "All teams":
    pool = pool[pool["Team"].astype(str).eq(team_filter)].copy()
pool = pool[pd.to_numeric(pool["Minutes"], errors="coerce").fillna(0).ge(minimum_minutes)].copy()
if pool.empty:
    st.warning("No players match the current pool filters. Widen the position, team or minutes filter.")
    st.stop()

basis_label = "within selected pool" if (position_filter != "All positions" or team_filter != "All teams") else "league-wide"
st.caption(f"{len(pool):,} players in the current pool · percentiles computed {basis_label}.")

style_groups = pa.style_metric_groups(pool)
if not style_groups:
    st.warning("No style metrics have data for the current pool.")
    st.stop()

player_name = pa.player_selector(pool, key="percentile_rankings_player", label="Highlight player")

ranking_tab, profile_tab = st.tabs(["Metric Ranking", "Style Profile"])

with ranking_tab:
    pa.section_heading("Ranking controls")
    control_cols = st.columns([1.0, 1.4, 0.8])
    with control_cols[0]:
        metric_category = st.selectbox("Metric category", list(style_groups), key="percentile_rankings_category")
    with control_cols[1]:
        metric = st.selectbox("Metric", style_groups[metric_category], key="percentile_rankings_metric")
    with control_cols[2]:
        slider_min = min(5, len(pool))
        slider_max = max(len(pool), slider_min + 1)
        top_n = st.slider(
            "Players shown", min_value=slider_min, max_value=slider_max, value=min(20, len(pool)), key="percentile_rankings_count"
        )

    _category, higher_is_better, short_label = pa.metric_meta(metric)
    ranked = pool.copy()
    ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
    ranked = ranked.dropna(subset=[metric])
    if player_name not in ranked["Player"].astype(str).unique():
        st.info(f"{player_name} has no value for {metric} in the current pool.")
    else:
        ranked["_Rank"] = ranked[metric].rank(ascending=not higher_is_better, method="min").astype(int)
        ranked["_Percentile"] = pa.percentile(ranked[metric], higher_is_better=higher_is_better)
        row = ranked[ranked["Player"].astype(str) == str(player_name)].iloc[0]
        metric_cols = st.columns(4)
        metric_cols[0].metric(f"{player_name} rank", f"{int(row['_Rank'])} / {len(ranked)}", pa.metric_value(row.get(metric), metric))
        metric_cols[1].metric("Percentile", f"{row['_Percentile']:.0f}th")
        metric_cols[2].metric("League average", pa.metric_value(ranked[metric].mean(), metric))
        metric_cols[3].metric("Direction", "Higher is better" if higher_is_better else "Lower is better")

        st.plotly_chart(pa.ranked_bar(pool, metric, selected=player_name, top_n=top_n, higher_is_better=higher_is_better), width="stretch")
        st.caption(f"{short_label}: ranked across {basis_label} players who meet the minutes filter.")

        pa.section_heading("Ranking table")
        table = ranked.sort_values("_Rank")
        table_cols = [column for column in ["Player", "Team", "_Position Display", "Minutes", metric, "_Rank", "_Percentile"] if column in table]
        display_table = table[table_cols].rename(
            columns={"_Position Display": "Position", "_Rank": "Rank", "_Percentile": "Percentile"}
        )
        st.dataframe(
            display_table,
            width="stretch",
            hide_index=True,
            column_config={
                "Percentile": st.column_config.ProgressColumn("Percentile", min_value=0, max_value=100, format="%.0f"),
            },
        )

with profile_tab:
    pa.section_heading(f"{player_name}: Style Profile")
    profile = _player_style_profile(pool, player_name, style_groups)
    if profile.empty:
        st.info(f"{player_name} has no style metrics available in the current pool.")
    else:
        top_third = int(profile["Percentile"].ge(67).sum())
        median_percentile = float(profile["Percentile"].median())
        st.caption(
            f"{player_name} sits in the top third on {top_third} of {len(profile)} available style metrics "
            f"(median percentile {median_percentile:.0f}), computed {basis_label}."
        )
        st.plotly_chart(_profile_figure(profile, player_name, basis_label), width="stretch")
        st.caption(
            "Green marks the top third, amber the middle third and red the bottom third of the current pool for each "
            "metric. Higher percentile always means a stronger standing after each metric's direction is applied."
        )
        with st.expander("Open the style profile table"):
            profile_display = profile.copy()
            profile_display["Value"] = [
                pa.metric_value(value, metric) for metric, value in zip(profile_display["Metric"], profile_display["Value"], strict=False)
            ]
            profile_display["Rank"] = [
                f"{int(rank)} of {int(pool_size)}" for rank, pool_size in zip(profile_display["Rank"], profile_display["Pool Size"], strict=False)
            ]
            st.dataframe(
                profile_display[["Category", "Metric", "Value", "Rank", "Percentile", "Direction"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "Percentile": st.column_config.ProgressColumn("Percentile", min_value=0, max_value=100, format="%.0f"),
                },
            )
