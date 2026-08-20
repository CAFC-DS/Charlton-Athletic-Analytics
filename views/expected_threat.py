# =============================================================================
# EXPECTED THREAT (xT) - team trend across games and player contribution
# =============================================================================
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import data
from utils import match_analysis as ma
from utils import pitch


XT_BASIS = (
    "Expected threat is built from Impect's per-event packing-xT values -- PXT Pass (passes and clearances) "
    "and PXT Shot (shots) -- summed as signed values, so a misplaced pass nets against the total rather than "
    "being ignored. This differs from the 'event threat' shown on Game Control / Momentum, which blends in the "
    "always-positive Team xT positional value; this page isolates the two fields that are directly attributable "
    "to the player who made the action."
)


def _team_xt_by_match(season: str, team_name: str, match_ids: list[object]) -> pd.DataFrame:
    columns = ["MatchId", "xT"]
    if not match_ids:
        return pd.DataFrame(columns=columns)
    events = data.load_match_events(
        season=season, team=team_name, match_ids=match_ids, action_types=data.XT_ACTION_TYPES, limit=120000,
    )
    if events.empty:
        return pd.DataFrame(columns=columns)
    events["xT Value"] = data.event_xt_value(events)
    grouped = events.groupby("MatchId", as_index=False)["xT Value"].sum().rename(columns={"xT Value": "xT"})
    return grouped[columns]


def _player_xt_contribution(events: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    columns = ["Player", "xT", "Actions", "xT / Action"]
    if events.empty or "Player" not in events:
        return pd.DataFrame(columns=columns)
    working = events.copy()
    working["xT Value"] = data.event_xt_value(working)
    working = working[working["xT Value"] != 0]
    if working.empty:
        return pd.DataFrame(columns=columns)
    grouped = working.groupby("Player", as_index=False).agg(xT=("xT Value", "sum"), Actions=("xT Value", "size"))
    grouped["xT / Action"] = (grouped["xT"] / grouped["Actions"]).round(4)
    grouped["xT"] = grouped["xT"].round(3)
    return grouped.sort_values("xT", ascending=False).head(top_n)[columns]


def _player_xt_chart(contribution: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if contribution.empty:
        fig.add_annotation(text="No Player Expected-Threat Contribution", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return ma.polish_figure(fig, title)
    plot_df = contribution.sort_values("xT")
    colors = np.where(plot_df["xT"] >= 0, ma.RED, ma.DARK)
    fig.add_trace(
        go.Bar(
            x=plot_df["xT"],
            y=plot_df["Player"],
            orientation="h",
            marker_color=colors,
            text=[f"{value:+.2f}" for value in plot_df["xT"]],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([plot_df["Actions"], plot_df["xT / Action"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>Total xT: %{x:.3f}<br>Actions: %{customdata[0]:.0f}<br>xT/Action: %{customdata[1]:.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color="#7a7f87", width=1.2, dash="dash"))
    fig.update_layout(height=max(360, 26 * len(plot_df) + 120), xaxis_title="Total Expected Threat (PXT)", yaxis_title="")
    return ma.polish_figure(fig, title)


def _season_xt_trend_chart(team_matches: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if team_matches.empty:
        fig.add_annotation(text="No Season Expected-Threat Data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return ma.polish_figure(fig, title)
    plot_df = team_matches.copy().reset_index(drop=True)
    plot_df["Fixture Number"] = np.arange(1, len(plot_df) + 1)
    plot_df["Rolling xT"] = plot_df["xT"].rolling(5, min_periods=1).mean().round(3)
    date_text = pd.to_datetime(plot_df["Date"], errors="coerce").dt.strftime("%d %b %Y").fillna("") if "Date" in plot_df else pd.Series("", index=plot_df.index)
    opponents = plot_df["Opponent"].fillna("Unknown") if "Opponent" in plot_df else pd.Series("Unknown", index=plot_df.index)
    results = plot_df["Team Result"].fillna("") if "Team Result" in plot_df else pd.Series("", index=plot_df.index)
    customdata = np.stack([date_text, opponents, results], axis=-1)
    bar_colors = np.where(plot_df["xT"] >= 0, "rgba(195, 0, 23, 0.78)", "rgba(17, 17, 17, 0.42)")
    fig.add_trace(
        go.Bar(
            x=plot_df["Fixture Number"],
            y=plot_df["xT"],
            name="Match Expected Threat",
            marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.78)", width=0.8)),
            customdata=customdata,
            hovertemplate=(
                "<b>Gameweek %{x:.0f}</b><br>Date: %{customdata[0]}<br>Opponent: %{customdata[1]}"
                "<br>Result: %{customdata[2]}<br>Match xT: %{y:.3f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["Fixture Number"],
            y=plot_df["Rolling xT"],
            name="5-Match Rolling Average",
            mode="lines",
            line=dict(color=ma.DARK, width=2.6),
            hovertemplate="Rolling avg xT: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color="#7a7f87", width=1.2, dash="dash"))
    fig.update_layout(height=460, xaxis_title="Gameweek (chronological)", yaxis_title="Expected Threat (PXT)", legend=dict(orientation="h", y=1.1))
    return ma.polish_figure(fig, title)


ma.page_header(
    "Expected Threat",
    "Track expected threat (xT) generated match by match across the season, and which players are driving it.",
    XT_BASIS,
)

season = ma.select_match_season(key="xt_season")
matches = ma.load_matches(season)
if matches.empty:
    st.warning("No match data is available for this season.")
    st.stop()

match_row = ma.match_selector(matches, key="xt_match")
team_name = ma.team_selector_for_match(match_row, key="xt_team")

team_matches = ma.team_match_rows(matches, team_name)
if team_matches.empty:
    st.warning(f"No fixtures are available for {team_name} in {season}.")
    st.stop()
match_ids = team_matches["MatchId"].dropna().tolist() if "MatchId" in team_matches else []

with st.spinner(f"Building {team_name}'s {season} expected-threat picture..."):
    season_events = data.load_match_events(
        season=season, team=team_name, match_ids=match_ids, action_types=data.XT_ACTION_TYPES, limit=120000,
    )
    match_xt = _team_xt_by_match(season, team_name, match_ids)

team_matches_with_xt = team_matches.merge(match_xt, on="MatchId", how="left")
team_matches_with_xt["xT"] = pd.to_numeric(team_matches_with_xt["xT"], errors="coerce").fillna(0.0)
season_total_xt = float(team_matches_with_xt["xT"].sum())
season_xt_per_90 = season_total_xt / len(team_matches_with_xt) if len(team_matches_with_xt) else 0.0

ma.section_heading("Season Expected-Threat Summary")
summary_cols = st.columns(4)
summary_cols[0].metric("Matches Covered", len(team_matches_with_xt))
summary_cols[1].metric("Total Expected Threat", f"{season_total_xt:+.2f}")
summary_cols[2].metric("Expected Threat / Match", f"{season_xt_per_90:+.3f}")
best_match = team_matches_with_xt.sort_values("xT", ascending=False).iloc[0] if not team_matches_with_xt.empty else None
summary_cols[3].metric("Best Match", str(best_match.get("Opponent", "N/A")) if best_match is not None else "N/A", f"{best_match['xT']:+.2f}" if best_match is not None else None)

ma.section_heading("Expected Threat Across the Season")
st.plotly_chart(_season_xt_trend_chart(team_matches_with_xt, f"{team_name}: Expected Threat by Gameweek"), width="stretch")
st.caption(
    "Bars show each match's net expected threat (PXT Pass + PXT Shot, summed as signed values across every "
    f"{team_name} action). The line is a 5-match rolling average. A negative bar means the team's own actions "
    "reduced their possession value more than they added to it that match -- not the same as losing, since goals "
    "conceded from set pieces or opponent errors aren't captured here."
)

ma.section_heading("Players Contributing to Expected Threat (Season)")
season_contribution = _player_xt_contribution(season_events, top_n=15)
if season_contribution.empty:
    st.info("No player-level expected-threat contributions are available for this selection.")
else:
    st.plotly_chart(_player_xt_chart(season_contribution, f"{team_name}: Top Expected-Threat Contributors -- {season}"), width="stretch")
    st.dataframe(season_contribution, width="stretch", hide_index=True)

ma.section_heading("Selected Match Detail")
metric_cols = st.columns(4)
selected_summary = ma.team_match_summary(match_row, team_name)
metric_cols[0].metric("Fixture", str(match_row.get("Match", "Unknown")))
metric_cols[1].metric("Result", selected_summary["Result"])
selected_match_xt = float(team_matches_with_xt.loc[team_matches_with_xt["MatchId"] == match_row.get("MatchId"), "xT"].sum())
metric_cols[2].metric("Match Expected Threat", f"{selected_match_xt:+.3f}")
match_all_teams_events = data.load_match_events(season=season, match_id=match_row.get("MatchId"), action_types=data.XT_ACTION_TYPES, limit=9000)
fixture_id = data.opta_fixture_id_for_match(match_row)
match_all_teams_events = data.append_opta_card_events(match_all_teams_events, fixture_id)
metric_cols[3].metric("Event Rows", len(match_all_teams_events))

st.plotly_chart(pitch.expected_threat_timeline(match_all_teams_events, f"{match_row.get('Match', 'Selected Match')}: Cumulative Expected Threat"), width="stretch")

match_team_events = match_all_teams_events[match_all_teams_events["Team"].astype(str) == str(team_name)].copy() if not match_all_teams_events.empty else match_all_teams_events
match_contribution = _player_xt_contribution(match_team_events, top_n=15)
if match_contribution.empty:
    st.info("No player-level expected-threat contributions are available for this selected match.")
else:
    st.plotly_chart(_player_xt_chart(match_contribution, f"{team_name}: Expected-Threat Contributors -- Selected Match"), width="stretch")

ma.section_heading("Selected Match Event Table")
display_events = match_team_events.copy()
if not display_events.empty:
    display_events["xT Value"] = data.event_xt_value(display_events)
    display_events = display_events[display_events["xT Value"] != 0]
display_cols = ma.available_columns(display_events, ["Minute", "Player", "Action", "Action Type", "Result", "xT Value"])
if display_events.empty:
    st.caption("No expected-threat event rows are available for this selected match.")
else:
    st.dataframe(display_events[display_cols].sort_values("Minute"), width="stretch", hide_index=True)
