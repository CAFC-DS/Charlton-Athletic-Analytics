# =============================================================================
# BUILD-UP INVOLVEMENT - who touches the ball as the team builds from the back
# =============================================================================
# A build-up sequence is defined here as a possession-sequence (one Sequence
# Index) whose first in-possession action starts outside the final third --
# i.e. the team has to build the attack rather than already being there.
# Involvement credits every player who touches the ball in that sequence,
# either as the acting player or as a pass receiver, not just the passer.
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charting, data, pitch, ui
from utils import match_analysis as ma


BUILD_UP_START_X = pitch.FINAL_THIRD_X  # sequences starting beyond this are already in the final third
RED = ui.CHARLTON_RED
DARK = ui.CHARLTON_BLACK
GREEN = "#16803c"
BLUE = "#344054"
GREY = "#7a7f87"
LIGHT_GREY = "#d0d5dd"


def _buildup_css() -> None:
    st.markdown(
        """
        <style>
        .bu-summary-card {
            background: var(--ss-panel);
            border: 1px solid var(--ss-border);
            border-radius: 8px;
            border-top: 3px solid var(--ss-accent);
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-height: 102px;
            padding: 14px 16px;
        }
        .bu-summary-label {
            color: var(--ss-muted);
            font-size: 0.875rem;
            font-weight: 650;
            line-height: 1.25;
            margin-bottom: 14px;
        }
        .bu-summary-value {
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
        <div class="bu-summary-card">
            <div class="bu-summary-label">{ui.esc(label)}</div>
            <div class="bu-summary-value">{ui.esc(value)}</div>
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


def _sequence_keys(events: pd.DataFrame) -> pd.DataFrame:
    """One row per (MatchId, Sequence Index) with ordering, start location and phase of the first event."""
    working = events.dropna(subset=["MatchId", "Sequence Index"]).copy()
    if working.empty:
        return pd.DataFrame(columns=["MatchId", "Sequence Index", "_Order"])
    working["_Order"] = _numeric(working, "Event Number")
    first = (
        working.sort_values("_Order")
        .groupby(["MatchId", "Sequence Index"], as_index=False)
        .first()
    )
    return first


def _buildup_sequence_keys(events: pd.DataFrame) -> pd.DataFrame:
    """Sequences whose first event is an in-possession action starting outside the final third."""
    first = _sequence_keys(events)
    if first.empty:
        return first
    first["Start X"] = _numeric(first, "Start X", np.nan)
    starts_deep = pd.to_numeric(first["Start X"], errors="coerce").lt(BUILD_UP_START_X)
    in_possession = first["Phase"].fillna("").astype(str).eq("IN_POSSESSION")
    return first[starts_deep & in_possession].copy()


def _sequence_outcomes(events: pd.DataFrame, buildup_keys: pd.DataFrame) -> pd.DataFrame:
    if buildup_keys.empty:
        return pd.DataFrame(columns=["Stage", "Sequences", "Conversion %"])
    keys = buildup_keys[["MatchId", "Sequence Index"]].drop_duplicates()
    working = events.dropna(subset=["MatchId", "Sequence Index"]).merge(keys, on=["MatchId", "Sequence Index"], how="inner")
    action_type = working["Action Type"].fillna("").astype(str).str.upper()
    action = working["Action"].fillna("").astype(str).str.upper()
    start_x = _numeric(working, "Start X", np.nan)
    end_x = _numeric(working, "End X", np.nan)
    working["_Reached Final Third"] = start_x.ge(BUILD_UP_START_X) | end_x.ge(BUILD_UP_START_X)
    working["_Shot"] = action_type.eq("SHOT")
    working["_Goal"] = action.eq("GOAL")
    working["_Shot xG"] = _numeric(working, "Shot xG").where(working["_Shot"], 0)
    per_sequence = working.groupby(["MatchId", "Sequence Index"], as_index=False).agg(
        **{
            "Reached Final Third": ("_Reached Final Third", "max"),
            "Shot": ("_Shot", "max"),
            "Goal": ("_Goal", "max"),
            "Shot xG": ("_Shot xG", "sum"),
        }
    )
    per_sequence["Reached Final Third"] = per_sequence[["Reached Final Third", "Shot"]].max(axis=1)
    per_sequence["Shot"] = per_sequence[["Shot", "Goal"]].max(axis=1)
    total = len(per_sequence)
    stages = ["Build-Up Sequences", "Reached Final Third", "Produced a Shot", "Produced a Goal"]
    counts = [
        total,
        int(per_sequence["Reached Final Third"].sum()),
        int(per_sequence["Shot"].sum()),
        int(per_sequence["Goal"].sum()),
    ]
    summary = pd.DataFrame({"Stage": stages, "Sequences": counts})
    summary["Conversion %"] = summary["Sequences"].div(max(total, 1)).mul(100)
    return summary


def _funnel_chart(summary: pd.DataFrame, title: str) -> go.Figure:
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
            hovertemplate="%{y}: %{x:.0f} sequences<br>%{customdata[0]:.1f}% of build-up sequences<extra></extra>",
        )
    )
    return charting.polish_figure(fig, title, height=440)


def _match_buildup_summary(events: pd.DataFrame, fixture_rows: pd.DataFrame) -> pd.DataFrame:
    buildup_keys = _buildup_sequence_keys(events)
    if buildup_keys.empty:
        return pd.DataFrame()
    per_match = buildup_keys.groupby("MatchId", as_index=False).agg(**{"Build-Up Sequences": ("Sequence Index", "size")})
    per_match["MatchId"] = per_match["MatchId"].astype(str)
    context = fixture_rows.copy()
    context["MatchId"] = context["MatchId"].astype(str)
    merged = per_match.merge(context[["MatchId", "Date", "Opponent", "Venue"]], on="MatchId", how="left")
    merged["Match Label"] = (
        pd.to_datetime(merged["Date"], errors="coerce").dt.strftime("%d %b %Y").fillna("Date unknown")
        + " · " + merged["Venue"].fillna("") + " vs " + merged["Opponent"].fillna("Unknown")
    )
    return merged.sort_values("Date").reset_index(drop=True)


def _trend_chart(match_summary: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if match_summary.empty:
        return charting.polish_figure(fig, title, height=440)
    plot_df = match_summary.copy()
    average = float(plot_df["Build-Up Sequences"].mean())
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(plot_df["Date"], errors="coerce"),
            y=plot_df["Build-Up Sequences"],
            mode="lines+markers",
            line=dict(color=RED, width=2.5),
            marker=dict(size=9, color=RED, line=dict(color="#ffffff", width=1.2)),
            customdata=np.stack([plot_df["Match Label"]], axis=-1),
            hovertemplate="%{customdata[0]}<br>Build-Up Sequences: %{y:.0f}<extra></extra>",
            name="Build-Up Sequences",
        )
    )
    if len(plot_df) > 1:
        fig.add_hline(y=average, line=dict(color=GREY, width=1.5, dash="dash"), annotation_text=f"Average: {average:.1f}")
    fig.update_xaxes(title="Match")
    fig.update_yaxes(title="Build-Up Sequences", rangemode="tozero")
    return charting.polish_figure(fig, title, height=460)


def _player_involvement(events: pd.DataFrame, buildup_keys: pd.DataFrame, minutes_lookup: pd.DataFrame) -> pd.DataFrame:
    if buildup_keys.empty:
        return pd.DataFrame()
    keys = buildup_keys[["MatchId", "Sequence Index"]].drop_duplicates()
    working = events.dropna(subset=["MatchId", "Sequence Index"]).merge(keys, on=["MatchId", "Sequence Index"], how="inner")
    total_sequences = len(keys)
    if total_sequences == 0:
        return pd.DataFrame()

    actor = working[["MatchId", "Sequence Index", "PlayerId", "Player"]].dropna(subset=["Player"])
    receiver = working[["MatchId", "Sequence Index", "ReceiverId", "Receiver"]].dropna(subset=["Receiver"]).rename(
        columns={"ReceiverId": "PlayerId", "Receiver": "Player"}
    )
    touches = pd.concat([actor, receiver], ignore_index=True).drop_duplicates(["MatchId", "Sequence Index", "Player"])
    touches["Touches"] = 1
    per_player = touches.groupby(["PlayerId", "Player"], as_index=False).agg(
        **{"Sequences Touched": ("MatchId", "size")}
    )
    per_player["Build-Up Involvement %"] = per_player["Sequences Touched"] / total_sequences * 100

    if not minutes_lookup.empty:
        lookup = minutes_lookup.copy()
        lookup["PlayerId"] = lookup["PlayerId"].astype(str)
        per_player["PlayerId"] = per_player["PlayerId"].astype(str)
        per_player = per_player.merge(lookup[["PlayerId", "Minutes"]], on="PlayerId", how="left")
    return per_player.sort_values("Build-Up Involvement %", ascending=False).reset_index(drop=True)


def _involvement_chart(players: pd.DataFrame, title: str, top_n: int = 15) -> go.Figure:
    fig = go.Figure()
    if players.empty:
        return charting.polish_figure(fig, title, height=420)
    plot_df = players.nlargest(top_n, "Build-Up Involvement %").sort_values("Build-Up Involvement %", ascending=True)
    minutes = plot_df["Minutes"] if "Minutes" in plot_df else pd.Series(np.nan, index=plot_df.index)
    customdata = np.stack([plot_df["Sequences Touched"], minutes.fillna(-1)], axis=-1)
    fig.add_trace(
        go.Bar(
            x=plot_df["Build-Up Involvement %"],
            y=plot_df["Player"],
            orientation="h",
            marker=dict(color=RED, line=dict(color="#ffffff", width=1)),
            text=[f"{value:.0f}%" for value in plot_df["Build-Up Involvement %"]],
            textposition="outside",
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                "<b>%{y}</b><br>Build-up involvement: %{x:.1f}%"
                "<br>Sequences touched: %{customdata[0]:.0f}"
                "<br>Minutes: %{customdata[1]:.0f}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="Build-Up Involvement % (share of build-up sequences touched)", range=[0, 100])
    fig.update_yaxes(title="")
    fig.update_layout(showlegend=False)
    return charting.polish_figure(
        fig, title, height=charting.horizontal_bar_height(len(plot_df), min_height=380, row_height=34, max_height=720)
    )


ma.page_header(
    "Build-Up Involvement",
    "Measure how often each player touches the ball while the team builds an attack from outside the final third, "
    "and whether that build-up play progresses into danger.",
    "CAFC_DB Impect provider events. A build-up sequence is a possession-sequence whose first in-possession action "
    "starts outside the final third.",
    (
        "Involvement credits any touch in a qualifying sequence (acting player or pass receiver), not passing volume "
        "alone. Minutes come from Impect match-player KPI facts where available and may be missing for some players."
    ),
)
_buildup_css()

control_cols = st.columns([1.0, 1.3, 0.9, 0.9])
with control_cols[0]:
    seasons = data.list_seasons().get("matches", [])
    if not seasons:
        st.warning("No match seasons are available.")
        st.stop()
    preferred_season = "25/26" if "25/26" in seasons else seasons[-1]
    season = st.selectbox("Season", seasons, index=seasons.index(preferred_season), key="buildup_season")

matches = ma.load_matches(season)
if matches.empty:
    st.warning("No match data is available for this season.")
    st.stop()

teams = _team_options(matches)
if not teams:
    st.warning("No teams are available from the selected match data.")
    st.stop()

with control_cols[1]:
    team_name = st.selectbox("Team", teams, index=_default_team_index(teams), key="buildup_team")

all_fixtures = _team_fixture_rows(matches, team_name)
if all_fixtures.empty:
    st.warning("No fixtures are available for the selected team.")
    st.stop()

with control_cols[2]:
    venue = st.selectbox("Venue", ["All", "Home", "Away"], key="buildup_venue")
with control_cols[3]:
    window = st.selectbox("Match window", ["Full Season", "Last 5", "Last 10"], key="buildup_window")

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

with st.spinner("Loading build-up sequence events..."):
    events = data.load_match_events(season=season, team=team_name, match_ids=selected_match_ids, limit=120000)
if len(events) >= 120000:
    st.warning("The selected-window event pull reached its 120,000-row cap; build-up totals may be incomplete.")
if events.empty:
    st.info("No event-level rows are available for the selected fixtures.")
    st.stop()

buildup_keys = _buildup_sequence_keys(events)
if buildup_keys.empty:
    st.info("No build-up sequences are available for the selected fixtures.")
    st.stop()

match_summary = _match_buildup_summary(events, venue_fixtures)
outcome_summary = _sequence_outcomes(events, buildup_keys)

ma.section_heading("Build-Up Snapshot")
matches_covered = max(len(venue_fixtures), 1)
snapshot_cols = st.columns(4)
with snapshot_cols[0]:
    _summary_card("Build-Up Sequences / Match", f"{len(buildup_keys) / matches_covered:.1f}")
with snapshot_cols[1]:
    reached_pct = (
        outcome_summary.loc[outcome_summary["Stage"].eq("Reached Final Third"), "Conversion %"].iloc[0]
        if not outcome_summary.empty else 0.0
    )
    _summary_card("Reach Final Third %", f"{reached_pct:.1f}%")
with snapshot_cols[2]:
    shot_pct = (
        outcome_summary.loc[outcome_summary["Stage"].eq("Produced a Shot"), "Conversion %"].iloc[0]
        if not outcome_summary.empty else 0.0
    )
    _summary_card("Produce a Shot %", f"{shot_pct:.1f}%")
with snapshot_cols[3]:
    goal_sequences = outcome_summary.loc[outcome_summary["Stage"].eq("Produced a Goal"), "Sequences"].iloc[0] if not outcome_summary.empty else 0
    _summary_card("Sequences Reaching a Goal", f"{goal_sequences:.0f}")
st.caption(
    "A build-up sequence is one possession-sequence id whose first in-possession action starts outside the final "
    "third for this team. The funnel below tracks the same sequences through to a shot or goal."
)

ma.section_heading("Build-Up Progression")
st.plotly_chart(_funnel_chart(outcome_summary, f"{team_name}: Build-Up-to-Attack Conversion"), width="stretch")
st.caption(
    "A sequence is linked to later events by matching MatchId and Sequence Index; this measures what happens after "
    "the build-up starts, not an Impect-defined possession-chain total."
)

ma.section_heading("Build-Up Volume Trend")
if match_summary.empty:
    st.info("No match-by-match build-up summary is available for this selection.")
else:
    st.plotly_chart(_trend_chart(match_summary, f"{team_name}: Build-Up Sequences by Match"), width="stretch")

ma.section_heading("Player Build-Up Involvement")
minutes_lookup = data.load_match_player_minutes(season=season, team=team_name)
if not minutes_lookup.empty:
    minutes_lookup = minutes_lookup.groupby(["PlayerId"], as_index=False).agg(Minutes=("Minutes", "sum"))
player_involvement = _player_involvement(events, buildup_keys, minutes_lookup)
if player_involvement.empty:
    st.info("No player-level build-up data is available for the selected fixtures.")
else:
    st.plotly_chart(
        _involvement_chart(player_involvement, f"{team_name}: Build-Up Involvement by Player"),
        width="stretch",
    )
    st.caption(
        "Build-up involvement is the share of the team's qualifying build-up sequences in which a player touched the "
        "ball, either by acting on it or receiving a pass -- it is not passing volume alone."
    )
    table_cols = [col for col in ["Player", "Sequences Touched", "Build-Up Involvement %", "Minutes"] if col in player_involvement]
    st.dataframe(
        player_involvement[table_cols],
        width="stretch",
        hide_index=True,
        column_config={
            "Build-Up Involvement %": st.column_config.ProgressColumn(
                "Build-Up Involvement %", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )

with st.expander("Terminology"):
    st.markdown(
        """
        - **Build-up sequence**: a possession-sequence (one Sequence Index) whose first in-possession event starts
          outside the final third for the selected team.
        - **Build-Up Involvement %**: the share of the team's build-up sequences in which a player is recorded as the
          acting player or the pass receiver at least once.
        - **Reach Final Third / Produce a Shot / Produce a Goal**: later events sharing the same match and sequence
          id as the build-up start; an event-based proxy, not an Impect-defined possession chain.
        """
    )
