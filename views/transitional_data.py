# =============================================================================
# TRANSITIONAL DATA - attacking-transition output, team and player
# =============================================================================
# Impect tags each event with a Phase; ATTACKING_TRANSITION marks actions taken
# in the moments right after winning the ball back, before the game settles
# into a structured possession. This page isolates that phase to answer: how
# often does the team transition, how fast, and does it produce shots -- both
# for the team as a whole and for the players driving it.
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charting, data, pitch, ui
from utils import match_analysis as ma


TRANSITION_PHASE = "ATTACKING_TRANSITION"
RED = ui.CHARLTON_RED
DARK = ui.CHARLTON_BLACK
GREEN = "#16803c"
GREY = "#7a7f87"


def _transitional_css() -> None:
    st.markdown(
        """
        <style>
        .td-summary-card {
            background: var(--ss-panel);
            border: 1px solid var(--ss-border);
            border-radius: 8px;
            border-top: 3px solid var(--ss-accent);
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-height: 102px;
            padding: 14px 16px;
        }
        .td-summary-label {
            color: var(--ss-muted);
            font-size: 0.875rem;
            font-weight: 650;
            line-height: 1.25;
            margin-bottom: 14px;
        }
        .td-summary-value {
            color: var(--ss-ink);
            display: block;
            font-size: clamp(1.35rem, 1.75vw, 1.75rem);
            font-weight: 400;
            letter-spacing: -0.03em;
            line-height: 1.1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _summary_card(label: str, value: object) -> None:
    st.markdown(
        f"""
        <div class="td-summary-card">
            <div class="td-summary-label">{ui.esc(label)}</div>
            <div class="td-summary-value">{ui.esc(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _numeric(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def _team_options(matches: pd.DataFrame) -> list[str]:
    values = pd.concat([matches.get("Home", pd.Series(dtype=str)), matches.get("Away", pd.Series(dtype=str))])
    return sorted(values.dropna().astype(str).loc[lambda s: s.str.strip().ne("")].unique().tolist())


def _default_team_index(teams: list[str]) -> int:
    for index, team in enumerate(teams):
        if "charlton" in team.lower():
            return index
    return 0


def _team_fixture_rows(matches: pd.DataFrame, team_name: str) -> pd.DataFrame:
    rows = matches[matches["Home"].astype(str).eq(str(team_name)) | matches["Away"].astype(str).eq(str(team_name))].copy()
    rows["Date"] = pd.to_datetime(rows["Date"], errors="coerce", utc=True).dt.tz_convert(None)
    rows["Venue"] = np.where(rows["Home"].astype(str).eq(str(team_name)), "Home", "Away")
    rows["Opponent"] = np.where(rows["Home"].astype(str).eq(str(team_name)), rows["Away"].astype(str), rows["Home"].astype(str))
    return rows.sort_values(["Date", "MatchId"]).reset_index(drop=True)


def _transition_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty or "Phase" not in events:
        return events.iloc[0:0].copy()
    out = events[events["Phase"].astype(str).eq(TRANSITION_PHASE)].copy()
    for col in ["PXT Pass", "PXT Shot", "Shot xG", "Bypassed Opponents", "Bypassed Defenders", "Second"]:
        out[col] = _numeric(out, col)
    out["_Threat"] = out[["PXT Pass", "PXT Shot"]].clip(lower=0).sum(axis=1)
    return out


def _sequence_outcomes(transition_events: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, sequence) transition, with duration and shot outcome."""
    if transition_events.empty:
        return pd.DataFrame(columns=["MatchId", "Sequence Index", "Duration Seconds", "Shot", "Goal", "Shot xG"])
    working = transition_events.dropna(subset=["MatchId", "Sequence Index"]).copy()
    if working.empty:
        return pd.DataFrame(columns=["MatchId", "Sequence Index", "Duration Seconds", "Shot", "Goal", "Shot xG"])
    action_type = working["Action Type"].fillna("").astype(str).str.upper()
    action = working["Action"].fillna("").astype(str).str.upper()
    working["_Shot"] = action_type.eq("SHOT")
    working["_Goal"] = action.eq("GOAL")
    grouped = working.groupby(["MatchId", "Sequence Index"], as_index=False).agg(
        **{
            "Duration Seconds": ("Second", lambda s: float(s.max() - s.min()) if len(s) > 1 else 0.0),
            "Events": ("Second", "size"),
            "Shot": ("_Shot", "max"),
            "Goal": ("_Goal", "max"),
            "Shot xG": ("Shot xG", "sum"),
            "Threat": ("_Threat", "sum"),
            "Bypassed Opponents": ("Bypassed Opponents", "sum"),
        }
    )
    return grouped


def _match_transition_summary(events_by_match: pd.DataFrame, fixture_rows: pd.DataFrame, team_name: str) -> pd.DataFrame:
    if events_by_match.empty:
        return pd.DataFrame()
    transitions = _transition_events(events_by_match)
    sequences = _sequence_outcomes(transitions)
    if sequences.empty:
        return pd.DataFrame()
    per_match = sequences.groupby("MatchId", as_index=False).agg(
        **{
            "Transition Sequences": ("Sequence Index", "size"),
            "Shots From Transition": ("Shot", "sum"),
            "Goals From Transition": ("Goal", "sum"),
            "xG From Transition": ("Shot xG", "sum"),
            "Threat From Transition": ("Threat", "sum"),
            "Avg Transition Duration": ("Duration Seconds", "mean"),
        }
    )
    per_match["Shots From Transition %"] = (
        per_match["Shots From Transition"] / per_match["Transition Sequences"].replace(0, np.nan) * 100
    )
    per_match["MatchId"] = per_match["MatchId"].astype(str)
    context = fixture_rows.copy()
    context["MatchId"] = context["MatchId"].astype(str)
    merged = per_match.merge(context[["MatchId", "Date", "Opponent", "Venue"]], on="MatchId", how="left")
    merged["Match Label"] = (
        pd.to_datetime(merged["Date"], errors="coerce").dt.strftime("%d %b %Y").fillna("Date unknown")
        + " · " + merged["Venue"].fillna("") + " vs " + merged["Opponent"].fillna("Unknown")
    )
    return merged.sort_values("Date").reset_index(drop=True)


def _trend_chart(match_summary: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = go.Figure()
    if match_summary.empty or metric not in match_summary:
        return charting.polish_figure(fig, title, height=440)
    plot_df = match_summary.copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])
    if plot_df.empty:
        return charting.polish_figure(fig, title, height=440)
    average = float(plot_df[metric].mean())
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(plot_df["Date"], errors="coerce"),
            y=plot_df[metric],
            mode="lines+markers",
            line=dict(color=RED, width=2.5),
            marker=dict(size=9, color=RED, line=dict(color="#ffffff", width=1.2)),
            customdata=np.stack([plot_df["Match Label"]], axis=-1),
            hovertemplate="%{customdata[0]}<br>" + metric + ": %{y:.2f}<extra></extra>",
            name=metric,
        )
    )
    if len(plot_df) > 1:
        fig.add_hline(y=average, line=dict(color=GREY, width=1.5, dash="dash"), annotation_text=f"Average: {average:.2f}")
    fig.update_xaxes(title="Match")
    fig.update_yaxes(title=metric, rangemode="tozero")
    return charting.polish_figure(fig, title, height=460)


def _transition_map(transition_events: pd.DataFrame, team_name: str, title: str) -> go.Figure:
    fig = pitch.pitch_figure(title, height=620, legend=True)
    spatial = transition_events.dropna(subset=["Start X", "Start Y"]) if not transition_events.empty else transition_events
    if spatial.empty:
        fig.add_annotation(x=0, y=0, text="No transition-phase locations", showarrow=False, font=dict(size=16, color=GREY))
        return fig

    spatial = spatial.copy()
    action_type = spatial["Action Type"].fillna("").astype(str).str.upper()
    action = spatial["Action"].fillna("").astype(str).str.upper()
    outcome = np.select([action.eq("GOAL"), action_type.eq("SHOT")], ["Goal", "Shot"], default="No Shot")
    spatial["_Outcome"] = outcome
    colour_map = {"Goal": GREEN, "Shot": "#c69214", "No Shot": "rgba(52,64,84,0.35)"}
    for group_name, colour in colour_map.items():
        group = spatial[spatial["_Outcome"].eq(group_name)]
        if group.empty:
            continue
        customdata = np.stack(
            [
                group["Player"].fillna("Unknown"),
                group["Action"].fillna(group["Action Type"]),
                group["Minute"].fillna(0),
                group["_Threat"].fillna(0),
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=group["Start X"],
                y=group["Start Y"],
                mode="markers",
                name=group_name,
                marker=dict(
                    size=9 if group_name == "No Shot" else 14,
                    color=colour,
                    opacity=0.85 if group_name == "No Shot" else 0.95,
                    line=dict(color="#ffffff", width=1),
                ),
                customdata=customdata,
                hovertemplate=(
                    "%{customdata[0]} - %{customdata[1]}"
                    "<br>Minute: %{customdata[2]:.0f}"
                    "<br>Threat: %{customdata[3]:.3f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(margin=dict(l=28, r=94, t=104, b=42))
    return fig


def _player_leaderboard(transition_events: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    if transition_events.empty or "Player" not in transition_events:
        return pd.DataFrame()
    grouped = transition_events.dropna(subset=["Player"]).groupby(["PlayerId", "Player"], as_index=False).agg(
        **{
            "Transition Actions": ("Player", "size"),
            "Bypassed Opponents": ("Bypassed Opponents", "sum"),
            "Threat": ("_Threat", "sum"),
            "Shot xG": ("Shot xG", "sum"),
        }
    )
    return grouped.sort_values("Threat", ascending=False).head(top_n)


def _leaderboard_chart(leaderboard: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if leaderboard.empty:
        return charting.polish_figure(fig, title, height=420)
    plot_df = leaderboard.sort_values("Threat", ascending=True)
    customdata = np.stack([plot_df["Transition Actions"], plot_df["Bypassed Opponents"], plot_df["Shot xG"]], axis=-1)
    fig.add_trace(
        go.Bar(
            x=plot_df["Threat"],
            y=plot_df["Player"],
            orientation="h",
            marker=dict(color=RED, line=dict(color="#ffffff", width=1)),
            text=[f"{value:.2f}" for value in plot_df["Threat"]],
            textposition="outside",
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                "<b>%{y}</b><br>Transition threat: %{x:.2f}"
                "<br>Transition actions: %{customdata[0]:.0f}"
                "<br>Bypassed opponents: %{customdata[1]:.1f}"
                "<br>Shot xG: %{customdata[2]:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="Transition threat (summed positive PXT pass + PXT shot)")
    fig.update_yaxes(title="")
    fig.update_layout(showlegend=False)
    return charting.polish_figure(
        fig, title, height=charting.horizontal_bar_height(len(plot_df), min_height=380, row_height=34, max_height=720)
    )


ma.page_header(
    "Transitional Data",
    "Isolate Impect's ATTACKING_TRANSITION phase to measure how often the team transitions after winning the ball, "
    "how quickly, and whether it produces shots -- team trend, a match map and a player leaderboard.",
    "CAFC_DB Impect provider events, filtered to the ATTACKING_TRANSITION phase label.",
    (
        "This uses provider phase tags on this team's own events only; it does not benchmark against the rest of the "
        "league (that would require pulling full-season events for every club) and it does not cover the mirror case of "
        "defensive transition risk after losing the ball. For our own regain-to-attack funnel, see Defensive Actions."
    ),
)
_transitional_css()

control_cols = st.columns([1.0, 1.3, 0.9, 0.9])
with control_cols[0]:
    seasons = data.list_seasons().get("matches", [])
    if not seasons:
        st.warning("No match seasons are available.")
        st.stop()
    preferred_season = "25/26" if "25/26" in seasons else seasons[-1]
    season = st.selectbox("Season", seasons, index=seasons.index(preferred_season), key="transitions_season")

matches = ma.load_matches(season)
if matches.empty:
    st.warning("No match data is available for this season.")
    st.stop()

teams = _team_options(matches)
if not teams:
    st.warning("No teams are available from the selected match data.")
    st.stop()

with control_cols[1]:
    team_name = st.selectbox("Team", teams, index=_default_team_index(teams), key="transitions_team")

all_fixtures = _team_fixture_rows(matches, team_name)
if all_fixtures.empty:
    st.warning("No fixtures are available for the selected team.")
    st.stop()

with control_cols[2]:
    venue = st.selectbox("Venue", ["All", "Home", "Away"], key="transitions_venue")
with control_cols[3]:
    window = st.selectbox("Match window", ["Full Season", "Last 5", "Last 10"], key="transitions_window")

venue_fixtures = all_fixtures if venue == "All" else all_fixtures[all_fixtures["Venue"].eq(venue)].copy()
if venue_fixtures.empty:
    st.info(f"No {venue.lower()} fixtures are available for this team and season.")
    st.stop()
if window == "Last 5":
    venue_fixtures = venue_fixtures.tail(5)
elif window == "Last 10":
    venue_fixtures = venue_fixtures.tail(10)

selected_match_ids = venue_fixtures["MatchId"].astype(str).tolist()
st.caption(f"{len(venue_fixtures)} matches · {venue} venue · {window} window.")

with st.spinner("Loading transition-phase events..."):
    events = data.load_match_events(season=season, team=team_name, match_ids=selected_match_ids, limit=120000)
if len(events) >= 120000:
    st.warning("The selected-window event pull reached its 120,000-row cap; transition totals may be incomplete.")
if events.empty:
    st.info("No event-level rows are available for the selected fixtures.")
    st.stop()

transitions = _transition_events(events)
if transitions.empty:
    st.info("No ATTACKING_TRANSITION-phase events are available for the selected fixtures.")
    st.stop()

match_summary = _match_transition_summary(events, venue_fixtures, team_name)

ma.section_heading("Transition Snapshot")
snapshot_cols = st.columns(4)
matches_covered = max(len(venue_fixtures), 1)
transitions_per_match = match_summary["Transition Sequences"].sum() / matches_covered if not match_summary.empty else 0.0
with snapshot_cols[0]:
    _summary_card("Transitions / Match", f"{transitions_per_match:.1f}")
with snapshot_cols[1]:
    shot_pct = (
        match_summary["Shots From Transition"].sum() / max(match_summary["Transition Sequences"].sum(), 1) * 100
        if not match_summary.empty else 0.0
    )
    _summary_card("Shots From Transition %", f"{shot_pct:.1f}%")
with snapshot_cols[2]:
    xg_per_match = match_summary["xG From Transition"].sum() / matches_covered if not match_summary.empty else 0.0
    _summary_card("xG From Transition / Match", f"{xg_per_match:.2f}")
with snapshot_cols[3]:
    avg_duration = pd.to_numeric(match_summary.get("Avg Transition Duration", pd.Series(dtype=float)), errors="coerce").mean()
    _summary_card("Avg Transition Duration", f"{avg_duration:.1f}s" if pd.notna(avg_duration) else "N/A")
st.caption(
    "A transition sequence is one possession-sequence id containing at least one ATTACKING_TRANSITION-phase event for "
    "this team. Duration is the span between the first and last transition-phase event timestamp in that sequence."
)

ma.section_heading("Transition Output Trend")
if match_summary.empty:
    st.info("No match-by-match transition summary is available for this selection.")
else:
    trend_metric = st.selectbox(
        "Trend metric",
        ["Transition Sequences", "Shots From Transition", "xG From Transition", "Threat From Transition", "Avg Transition Duration"],
        key="transitions_trend_metric",
    )
    st.plotly_chart(_trend_chart(match_summary, trend_metric, f"{team_name}: {trend_metric} by Match"), width="stretch")

ma.section_heading("Transition Map")
map_match_options = {str(row["MatchId"]): row.get("Match Label", str(row["MatchId"])) for _, row in match_summary.iterrows()} if not match_summary.empty else {}
if map_match_options:
    map_match_id = st.selectbox(
        "Match", list(map_match_options), format_func=lambda mid: map_match_options.get(mid, mid), key="transitions_map_match"
    )
    map_events = transitions[transitions["MatchId"].astype(str).eq(map_match_id)]
    st.plotly_chart(
        _transition_map(map_events, team_name, f"{team_name}: Transition Actions vs Outcome"),
        width="stretch",
    )
    st.caption(
        "Each marker is a transition-phase action's start location. Gold marks a shot-producing sequence, green a "
        "goal-producing sequence, and grey shows transition play that did not reach a shot."
    )
else:
    st.info("No matches with transition sequences are available to map.")

ma.section_heading("Player Involvement in Transitions")
leaderboard = _player_leaderboard(transitions)
if leaderboard.empty:
    st.info("No player-level transition data is available for the selected fixtures.")
else:
    st.plotly_chart(_leaderboard_chart(leaderboard, f"{team_name}: Transition Threat by Player"), width="stretch")
    st.caption(
        "Threat sums each player's positive PXT pass and PXT shot values recorded during ATTACKING_TRANSITION-phase "
        "actions -- it credits the players driving the team's transition play, not just the eventual shot-taker."
    )
    st.dataframe(leaderboard, width="stretch", hide_index=True)

with st.expander("Terminology"):
    st.markdown(
        """
        - **ATTACKING_TRANSITION phase**: Impect's own event-level tag for the moments right after a team regains
          possession, before play settles into a structured attack.
        - **Transition sequence**: all events sharing one Sequence Index that include at least one transition-phase action.
        - **Threat**: the positive parts of PXT Pass and PXT Shot summed across an action or player; a proxy for danger
          created, not an official Impect KPI.
        - **Duration**: seconds between the first and last transition-phase event timestamp within a sequence; an
          approximation of how long the transition window lasted, not a provider-defined transition-speed metric.
        """
    )
