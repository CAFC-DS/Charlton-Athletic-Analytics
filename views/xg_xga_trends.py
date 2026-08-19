# =============================================================================
# XG / XGA TRENDS - real event xG where available
# =============================================================================
import plotly.graph_objects as go
import streamlit as st

from utils import charting, data
from utils import team_analysis as ta


ta.page_header(
    "xG / xGA Trends",
    "Track expected goals for and against over time for a selected team.",
    "CAFC_DB Impect provider events supply shot xG for event-covered seasons. Seasons without event xG fall back to actual goals.",
)

team_season = ta.select_season("teams", key="xg_team_season")
teams = ta.load_team_data(team_season)
if teams.empty:
    st.warning("No team data is available for team selection.")
    st.stop()
team_name = ta.team_selector(teams, key="xg_team")

match_season = ta.select_season("matches", key="xg_match_season")
matches = data.load_matches(season=match_season)
team_matches = ta.match_rows_for_team(matches, team_name)

if team_matches.empty:
    st.info("No match rows are available for the selected team and match season.")
    st.stop()

xg_rows = data.load_team_match_shot_xg(season=match_season)
team_xg = xg_rows.rename(columns={"Team": "Team For", "xG": "xG For", "Post-Shot xG": "Post-Shot xG For", "Shots": "Shots For"})
opp_xg = xg_rows.rename(columns={"Team": "Opponent", "xG": "xG Against", "Post-Shot xG": "Post-Shot xG Against", "Shots": "Shots Against"})
trend = team_matches.merge(
    team_xg[["MatchId", "Team For", "xG For", "Post-Shot xG For", "Shots For"]],
    on="MatchId",
    how="left",
)
trend = trend[trend["Team For"].isna() | (trend["Team For"].astype(str) == str(team_name))].copy()
trend = trend.drop_duplicates(subset=["MatchId"])
trend = trend.merge(
    opp_xg[["MatchId", "Opponent", "xG Against", "Post-Shot xG Against", "Shots Against"]],
    on=["MatchId", "Opponent"],
    how="left",
)

has_xg = trend["xG For"].notna().any() or trend["xG Against"].notna().any()

if has_xg:
    ta.section_heading("Expected goals for and against")
    x = trend["Date"] if "Date" in trend else trend.index + 1
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=trend["xG For"],
            mode="lines+markers",
            name="xG For",
            line=dict(color=ta.RED, width=3),
            marker=dict(size=8),
            hovertemplate="xG for: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=trend["xG Against"],
            mode="lines+markers",
            name="xG Against",
            line=dict(color=ta.DARK, width=3),
            marker=dict(size=8),
            hovertemplate="xG against: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(height=500, yaxis_title="xG", xaxis_title="Match date")
    fig.update_yaxes(tickformat=".2f", rangemode="tozero")
    st.plotly_chart(ta.polish_figure(fig, f"{team_name}: xG for and against"), width="stretch")

    ta.section_heading("Cumulative xG trend")
    cumulative = trend.copy()
    cumulative["Cumulative xG For"] = cumulative["xG For"].fillna(0).cumsum()
    cumulative["Cumulative xG Against"] = cumulative["xG Against"].fillna(0).cumsum()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=cumulative["Cumulative xG For"],
            mode="lines+markers",
            name="Cumulative xG For",
            line=dict(color=ta.RED, width=3),
            marker=dict(size=8),
            hovertemplate="Cumulative xG for: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=cumulative["Cumulative xG Against"],
            mode="lines+markers",
            name="Cumulative xG Against",
            line=dict(color=ta.DARK, width=3),
            marker=dict(size=8),
            hovertemplate="Cumulative xG against: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(height=500, yaxis_title="Cumulative xG", xaxis_title="Match date")
    fig.update_yaxes(tickformat=".2f", rangemode="tozero")
    st.plotly_chart(charting.polish_figure(fig, "Cumulative xG trend"), width="stretch")

    ta.section_heading("Fixture xG detail")
    display_cols = [
        "Date",
        "Opponent",
        "Goals For",
        "Goals Against",
        "xG For",
        "xG Against",
        "Shots For",
        "Shots Against",
        "Team Result",
    ]
    st.dataframe(trend[[col for col in display_cols if col in trend]], width="stretch", hide_index=True)
else:
    st.info("No event-level shot xG rows are available for this selected match season. Showing actual goals as a fallback.")
    ta.section_heading("Goals for and against")
    st.plotly_chart(ta.match_trend_chart(team_matches, f"{team_name}: goals for and against"), width="stretch")

    ta.section_heading("Fixture detail")
    st.dataframe(
        team_matches[["Date", "Opponent", "Goals For", "Goals Against", "Goal Difference", "Team Result"]],
        width="stretch",
        hide_index=True,
    )
