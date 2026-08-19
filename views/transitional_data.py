# =============================================================================
# TRANSITIONAL DATA - attacking-transition output, both for and against
# =============================================================================
# Impect tags each event with a Phase; ATTACKING_TRANSITION marks actions taken
# in the moments right after winning the ball back, before the game settles
# into a structured possession. This page isolates that phase on both sides of
# the ball: how well the team transitions after winning it back ("for"), and
# how exposed the team is to the opponent doing the same after losing it
# ("against" -- transition risk) -- team trend, zone/duration breakdowns, a
# progression funnel, a match map and a player leaderboard.
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
AMBER = "#d89216"
GREY = "#7a7f87"
LIGHT_GREY = "#d0d5dd"
AGAINST_COLOUR = "#344054"
ZONE_ORDER = ["Defensive Third", "Middle Third", "Final Third"]
ZONE_COLOURS: dict[str, str] = {
    "Defensive Third": "#344054",
    "Middle Third": "#c69214",
    "Final Third": ui.CHARLTON_RED,
}


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
        .td-balance-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin: 8px 0 18px;
        }
        .td-balance-card {
            background: #ffffff;
            border: 1px solid #e2e7ee;
            border-top: 4px solid #98a2b3;
            border-radius: 9px;
            box-shadow: 0 5px 18px rgba(16, 24, 40, 0.055);
            padding: 15px 16px;
        }
        .td-balance-card.favour { border-top-color: #16803c; }
        .td-balance-card.even { border-top-color: #98a2b3; }
        .td-balance-card.risk { border-top-color: #c30017; }
        .td-balance-label {
            color: #667085;
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .td-balance-value {
            color: #101828;
            font-size: 1.5rem;
            font-weight: 850;
            letter-spacing: -0.03em;
            margin: 8px 0 6px;
        }
        .td-balance-note {
            color: #667085;
            font-size: 0.79rem;
            line-height: 1.35;
        }
        .td-callout {
            padding: 14px 16px;
            margin: 4px 0 18px;
            border: 1px solid #dfe5ec;
            border-left: 4px solid #c30017;
            border-radius: 9px;
            background: #f8fafc;
            font-size: 0.92rem;
            color: #253045;
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


def _zone_label(start_x: object) -> str:
    value = pd.to_numeric(pd.Series([start_x]), errors="coerce").iloc[0]
    if pd.isna(value):
        return "Unknown"
    if value < -pitch.FINAL_THIRD_X:
        return "Defensive Third"
    if value < pitch.FINAL_THIRD_X:
        return "Middle Third"
    return "Final Third"


def _transition_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty or "Phase" not in events:
        return events.iloc[0:0].copy()
    out = events[events["Phase"].astype(str).eq(TRANSITION_PHASE)].copy()
    for col in ["PXT Pass", "PXT Shot", "Shot xG", "Bypassed Opponents", "Bypassed Defenders", "Second"]:
        out[col] = _numeric(out, col)
    out["_Threat"] = out[["PXT Pass", "PXT Shot"]].clip(lower=0).sum(axis=1)
    start_x = _numeric(out, "Start X", np.nan)
    out["Zone"] = [_zone_label(value) for value in start_x]
    return out


def _sequence_outcomes(transition_events: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, sequence) transition, with duration, zone and shot outcome."""
    columns = ["MatchId", "Sequence Index", "Duration Seconds", "Shot", "Goal", "Shot xG", "Threat", "Bypassed Opponents", "Zone"]
    if transition_events.empty:
        return pd.DataFrame(columns=columns)
    working = transition_events.dropna(subset=["MatchId", "Sequence Index"]).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
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
            "Zone": ("Zone", "first"),
        }
    )
    return grouped


def _match_transition_summary(events_by_match: pd.DataFrame, fixture_rows: pd.DataFrame) -> pd.DataFrame:
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


def _funnel_data(sequences: pd.DataFrame) -> pd.DataFrame:
    if sequences.empty:
        return pd.DataFrame(columns=["Stage", "Sequences", "Conversion %"])
    total = len(sequences)
    counts = [total, int(sequences["Shot"].sum()), int(sequences["Goal"].sum())]
    stages = ["Transition Sequences", "Produced a Shot", "Produced a Goal"]
    summary = pd.DataFrame({"Stage": stages, "Sequences": counts})
    summary["Conversion %"] = summary["Sequences"].div(max(total, 1)).mul(100)
    return summary


def _funnel_chart(summary: pd.DataFrame, title: str, colours: list[str]) -> go.Figure:
    fig = go.Figure()
    if summary.empty:
        return charting.polish_figure(fig, title, height=380)
    fig.add_trace(
        go.Funnel(
            y=summary["Stage"],
            x=summary["Sequences"],
            textinfo="value+percent initial",
            marker=dict(color=colours),
            connector=dict(line=dict(color=LIGHT_GREY, width=1)),
            customdata=np.stack([summary["Conversion %"]], axis=-1),
            hovertemplate="%{y}: %{x:.0f} sequences<br>%{customdata[0]:.1f}% of transition sequences<extra></extra>",
        )
    )
    return charting.polish_figure(fig, title, height=380)


def _zone_breakdown_figure(sequences: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if sequences.empty:
        return charting.polish_figure(fig, title, height=340)
    summary = sequences[sequences["Zone"].isin(ZONE_ORDER)].groupby("Zone", as_index=False).agg(
        Sequences=("Zone", "size"), **{"Shot xG": ("Shot xG", "sum")}
    )
    if summary.empty:
        return charting.polish_figure(fig, title, height=340)
    summary["_Order"] = summary["Zone"].map({zone: index for index, zone in enumerate(ZONE_ORDER)})
    summary = summary.sort_values("_Order")
    total = max(float(summary["Sequences"].sum()), 1.0)
    summary["Share %"] = summary["Sequences"] / total * 100
    colours = [ZONE_COLOURS.get(zone, "#98a2b3") for zone in summary["Zone"]]
    customdata = np.stack([summary["Share %"], summary["Shot xG"]], axis=-1)
    fig.add_trace(
        go.Bar(
            x=summary["Zone"],
            y=summary["Sequences"],
            marker=dict(color=colours, line=dict(color="#ffffff", width=1)),
            text=[f"{value:.0f} ({share:.0f}%)" for value, share in zip(summary["Sequences"], summary["Share %"], strict=False)],
            textposition="outside",
            customdata=customdata,
            hovertemplate=(
                "<b>%{x}</b><br>Sequences: %{y:.0f}<br>Share: %{customdata[0]:.0f}%<br>xG: %{customdata[1]:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="Zone the transition started in", categoryorder="array", categoryarray=ZONE_ORDER)
    fig.update_yaxes(title="Transition sequences", rangemode="tozero")
    fig.update_layout(showlegend=False)
    return charting.polish_figure(fig, title, height=380)


def _duration_histogram(sequences: pd.DataFrame, title: str, colour: str) -> go.Figure:
    fig = go.Figure()
    if sequences.empty:
        return charting.polish_figure(fig, title, height=340)
    durations = pd.to_numeric(sequences["Duration Seconds"], errors="coerce").dropna()
    durations = durations[durations >= 0]
    if durations.empty:
        return charting.polish_figure(fig, title, height=340)
    fig.add_trace(
        go.Histogram(
            x=durations,
            xbins=dict(start=0, end=max(float(durations.max()), 5) + 2, size=2),
            marker=dict(color=colour, line=dict(color="#ffffff", width=0.6)),
            hovertemplate="Duration bucket: %{x:.0f}s<br>Sequences: %{y}<extra></extra>",
        )
    )
    median = float(durations.median())
    fig.add_vline(x=median, line=dict(color=DARK, width=1.5, dash="dash"), annotation_text=f"Median: {median:.1f}s")
    fig.update_xaxes(title="Transition duration (seconds)")
    fig.update_yaxes(title="Sequences")
    fig.update_layout(showlegend=False)
    return charting.polish_figure(fig, title, height=380)


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


def _combined_trend_chart(summary_for: pd.DataFrame, summary_against: pd.DataFrame, metric: str, title: str, team_name: str) -> go.Figure:
    fig = go.Figure()
    if summary_for.empty and summary_against.empty:
        return charting.polish_figure(fig, title, height=460)
    for label, frame, colour in [(team_name, summary_for, RED), ("Opponents", summary_against, AGAINST_COLOUR)]:
        if frame.empty or metric not in frame:
            continue
        plot_df = frame.copy()
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
        plot_df = plot_df.dropna(subset=[metric])
        if plot_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(plot_df["Date"], errors="coerce"),
                y=plot_df[metric],
                mode="lines+markers",
                name=label,
                line=dict(color=colour, width=2.4),
                marker=dict(size=8, color=colour, line=dict(color="#ffffff", width=1)),
                customdata=np.stack([plot_df["Match Label"]], axis=-1),
                hovertemplate="%{customdata[0]}<br>" + label + " " + metric + ": %{y:.2f}<extra></extra>",
            )
        )
    fig.update_xaxes(title="Match")
    fig.update_yaxes(title=metric, rangemode="tozero")
    fig.update_layout(legend=dict(orientation="h", y=1.05, x=0))
    return charting.polish_figure(fig, title, height=460)


def _transition_map(transition_events: pd.DataFrame, title: str) -> go.Figure:
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


def _matches_summary(match_summary: pd.DataFrame, matches_covered: int) -> dict[str, float]:
    if match_summary.empty:
        return {"per_match": 0.0, "shot_pct": 0.0, "xg_per_match": 0.0, "avg_duration": np.nan, "total": 0.0}
    total = float(match_summary["Transition Sequences"].sum())
    return {
        "per_match": total / max(matches_covered, 1),
        "shot_pct": match_summary["Shots From Transition"].sum() / max(total, 1) * 100,
        "xg_per_match": match_summary["xG From Transition"].sum() / max(matches_covered, 1),
        "avg_duration": pd.to_numeric(match_summary["Avg Transition Duration"], errors="coerce").mean(),
        "total": total,
    }


def _balance_card_status(diff: float, higher_is_better: bool = True) -> str:
    if pd.isna(diff) or abs(diff) < 1e-9:
        return "even"
    is_favourable = diff > 0 if higher_is_better else diff < 0
    return "favour" if is_favourable else "risk"


ma.page_header(
    "Transitional Data",
    "Isolate Impect's ATTACKING_TRANSITION phase on both sides of the ball: how well the team transitions after "
    "winning possession, and how exposed it is when the opponent does the same -- volume, zone, speed, shot "
    "conversion, a match map and player involvement.",
    "CAFC_DB Impect provider events, filtered to the ATTACKING_TRANSITION phase label, for both the selected team "
    "and its opponent within the same fixtures.",
    (
        "This uses provider phase tags from the selected fixtures only; it does not benchmark against the rest of "
        "the league (that would require pulling full-season events for every club). For the regain-to-attack "
        "sequence funnel built from spatial/action proxies instead of phase tags, see Defensive Actions."
    ),
)
_transitional_css()

control_cols = st.columns([1.0, 1.3, 0.9, 0.9])
with control_cols[0]:
    seasons = data.list_seasons().get("matches", [])
    if not seasons:
        st.warning("No match seasons are available.")
        st.stop()
    preferred_season = data.preferred_season(seasons)
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

with st.spinner("Loading transition-phase events for both teams..."):
    all_events = data.load_match_events(season=season, match_ids=selected_match_ids, limit=120000)
if len(all_events) >= 120000:
    st.warning("The selected-window event pull reached its 120,000-row cap; transition totals may be incomplete.")
if all_events.empty:
    st.info("No event-level rows are available for the selected fixtures.")
    st.stop()

team_events = all_events[all_events["Team"].astype(str) == str(team_name)].copy()
opponent_events = all_events[all_events["Team"].astype(str) != str(team_name)].copy()

transitions_for = _transition_events(team_events)
transitions_against = _transition_events(opponent_events)
if transitions_for.empty and transitions_against.empty:
    st.info("No ATTACKING_TRANSITION-phase events are available for the selected fixtures.")
    st.stop()

sequences_for = _sequence_outcomes(transitions_for)
sequences_against = _sequence_outcomes(transitions_against)
match_summary_for = _match_transition_summary(team_events, venue_fixtures)
match_summary_against = _match_transition_summary(opponent_events, venue_fixtures)

matches_covered = max(len(venue_fixtures), 1)
stats_for = _matches_summary(match_summary_for, matches_covered)
stats_against = _matches_summary(match_summary_against, matches_covered)

ma.section_heading("Transition Balance")
diff_per_match = stats_for["per_match"] - stats_against["per_match"]
diff_xg = stats_for["xg_per_match"] - stats_against["xg_per_match"]
balance_cards = [
    (
        "Transitions For vs Against / Match",
        f"{stats_for['per_match']:.1f} — {stats_against['per_match']:.1f}",
        f"Differential {diff_per_match:+.1f} / match",
        _balance_card_status(diff_per_match),
    ),
    (
        "Shot Conversion For vs Against",
        f"{stats_for['shot_pct']:.0f}% — {stats_against['shot_pct']:.0f}%",
        f"Differential {stats_for['shot_pct'] - stats_against['shot_pct']:+.0f} pp",
        _balance_card_status(stats_for["shot_pct"] - stats_against["shot_pct"]),
    ),
    (
        "xG For vs Against / Match",
        f"{stats_for['xg_per_match']:.2f} — {stats_against['xg_per_match']:.2f}",
        f"Differential {diff_xg:+.2f} / match",
        _balance_card_status(diff_xg),
    ),
    (
        "Avg Duration For vs Against",
        f"{stats_for['avg_duration']:.1f}s — {stats_against['avg_duration']:.1f}s"
        if pd.notna(stats_for["avg_duration"]) and pd.notna(stats_against["avg_duration"])
        else "N/A",
        "Faster transitions are harder to defend against",
        "even",
    ),
]
cards_html = "".join(
    f'<div class="td-balance-card {status}"><div class="td-balance-label">{ui.esc(label)}</div>'
    f'<div class="td-balance-value">{ui.esc(value)}</div><div class="td-balance-note">{ui.esc(note)}</div></div>'
    for label, value, note, status in balance_cards
)
st.markdown(f'<div class="td-balance-grid">{cards_html}</div>', unsafe_allow_html=True)
if diff_per_match > 0.5:
    balance_copy = f"{team_name} generates meaningfully more transition volume than it concedes in this window -- a net transition advantage."
elif diff_per_match < -0.5:
    balance_copy = f"{team_name} concedes more transition sequences than it generates in this window -- worth reviewing defensive shape immediately after losing the ball."
else:
    balance_copy = f"{team_name}'s transition volume for and against is roughly balanced in this window."
st.markdown(f'<div class="td-callout">{ui.esc(balance_copy)}</div>', unsafe_allow_html=True)
st.plotly_chart(
    _combined_trend_chart(match_summary_for, match_summary_against, "Transition Sequences", f"{team_name}: Transition Sequences For vs Against by Match", team_name),
    width="stretch",
)
st.caption(
    "For (red) is this team's own transition sequences; Against (dark) is the opponent's transition sequences within "
    "the same fixtures -- i.e. what happened immediately after this team lost the ball. Values are counted from event "
    "phase tags, not an official Impect transition-count KPI."
)

for_tab, against_tab = st.tabs(["Attacking Transitions (For)", "Transition Risk (Against)"])

with for_tab:
    ma.section_heading("Transition Snapshot")
    snapshot_cols = st.columns(4)
    with snapshot_cols[0]:
        _summary_card("Transitions / Match", f"{stats_for['per_match']:.1f}")
    with snapshot_cols[1]:
        _summary_card("Shots From Transition %", f"{stats_for['shot_pct']:.1f}%")
    with snapshot_cols[2]:
        _summary_card("xG From Transition / Match", f"{stats_for['xg_per_match']:.2f}")
    with snapshot_cols[3]:
        _summary_card("Avg Transition Duration", f"{stats_for['avg_duration']:.1f}s" if pd.notna(stats_for["avg_duration"]) else "N/A")
    st.caption(
        "A transition sequence is one possession-sequence id containing at least one ATTACKING_TRANSITION-phase event "
        "for this team. Duration is the span between the first and last transition-phase event timestamp in that sequence."
    )

    detail_cols = st.columns(2)
    with detail_cols[0]:
        ma.section_heading("Progression Funnel")
        st.plotly_chart(_funnel_chart(_funnel_data(sequences_for), f"{team_name}: Transition-to-Shot Conversion", [DARK, RED, GREEN]), width="stretch")
    with detail_cols[1]:
        ma.section_heading("Zone of Origin")
        st.plotly_chart(_zone_breakdown_figure(sequences_for, f"{team_name}: Transitions by Starting Zone"), width="stretch")
        st.caption("Zone is where the transition sequence started, normalised so this team always attacks left to right.")

    ma.section_heading("Transition Duration")
    st.plotly_chart(_duration_histogram(sequences_for, f"{team_name}: Transition Duration Distribution", RED), width="stretch")
    st.caption("Faster transitions (further left) are typically harder for a settled defence to react to.")

    ma.section_heading("Transition Output Trend")
    if match_summary_for.empty:
        st.info("No match-by-match transition summary is available for this selection.")
    else:
        trend_metric = st.selectbox(
            "Trend metric",
            ["Transition Sequences", "Shots From Transition", "xG From Transition", "Threat From Transition", "Avg Transition Duration"],
            key="transitions_trend_metric",
        )
        st.plotly_chart(_trend_chart(match_summary_for, trend_metric, f"{team_name}: {trend_metric} by Match"), width="stretch")

    ma.section_heading("Transition Map")
    map_match_options = (
        {str(row["MatchId"]): row.get("Match Label", str(row["MatchId"])) for _, row in match_summary_for.iterrows()}
        if not match_summary_for.empty else {}
    )
    if map_match_options:
        map_match_id = st.selectbox(
            "Match", list(map_match_options), format_func=lambda mid: map_match_options.get(mid, mid), key="transitions_map_match"
        )
        map_events = transitions_for[transitions_for["MatchId"].astype(str).eq(map_match_id)]
        st.plotly_chart(_transition_map(map_events, f"{team_name}: Transition Actions vs Outcome"), width="stretch")
        st.caption(
            "Each marker is a transition-phase action's start location. Gold marks a shot-producing sequence, green a "
            "goal-producing sequence, and grey shows transition play that did not reach a shot."
        )
    else:
        st.info("No matches with transition sequences are available to map.")

    ma.section_heading("Player Involvement in Transitions")
    leaderboard = _player_leaderboard(transitions_for)
    if leaderboard.empty:
        st.info("No player-level transition data is available for the selected fixtures.")
    else:
        st.plotly_chart(_leaderboard_chart(leaderboard, f"{team_name}: Transition Threat by Player"), width="stretch")
        st.caption(
            "Threat sums each player's positive PXT pass and PXT shot values recorded during ATTACKING_TRANSITION-phase "
            "actions -- it credits the players driving the team's transition play, not just the eventual shot-taker."
        )
        st.dataframe(leaderboard, width="stretch", hide_index=True)

with against_tab:
    st.caption(
        f"Everything below is the opponent's ATTACKING_TRANSITION-phase play inside {team_name}'s own {len(venue_fixtures)} "
        "selected fixtures -- i.e. what happens immediately after this team loses the ball. Read it as transition risk, "
        "not opponent scouting across their whole season."
    )
    ma.section_heading("Transition-Risk Snapshot")
    snapshot_cols = st.columns(4)
    with snapshot_cols[0]:
        _summary_card("Transitions Conceded / Match", f"{stats_against['per_match']:.1f}")
    with snapshot_cols[1]:
        _summary_card("Shots Conceded From Transition %", f"{stats_against['shot_pct']:.1f}%")
    with snapshot_cols[2]:
        _summary_card("xG Conceded From Transition / Match", f"{stats_against['xg_per_match']:.2f}")
    with snapshot_cols[3]:
        _summary_card("Avg Duration Conceded", f"{stats_against['avg_duration']:.1f}s" if pd.notna(stats_against["avg_duration"]) else "N/A")

    detail_cols = st.columns(2)
    with detail_cols[0]:
        ma.section_heading("Progression Funnel (Against)")
        st.plotly_chart(_funnel_chart(_funnel_data(sequences_against), "Opponent: Transition-to-Shot Conversion", [AGAINST_COLOUR, AMBER, "#c30017"]), width="stretch")
    with detail_cols[1]:
        ma.section_heading("Zone Conceded")
        st.plotly_chart(_zone_breakdown_figure(sequences_against, "Opponent Transitions by Starting Zone"), width="stretch")
        st.caption(
            "Zone is from the opponent's own attacking perspective. A high share starting in their Defensive/Middle "
            "Third means this team is regularly caught out by transitions that start deep and travel far."
        )

    ma.section_heading("Duration Conceded")
    st.plotly_chart(_duration_histogram(sequences_against, "Opponent Transition Duration Distribution", AGAINST_COLOUR), width="stretch")
    st.caption("A cluster of very short durations suggests the opponent is punishing quick, direct turnovers rather than building through several passes.")

    ma.section_heading("Transition-Risk Trend")
    if match_summary_against.empty:
        st.info("No match-by-match transition-risk summary is available for this selection.")
    else:
        risk_trend_metric = st.selectbox(
            "Trend metric",
            ["Transition Sequences", "Shots From Transition", "xG From Transition", "Threat From Transition", "Avg Transition Duration"],
            key="transitions_risk_trend_metric",
        )
        st.plotly_chart(
            _trend_chart(match_summary_against, risk_trend_metric, f"Opponent: {risk_trend_metric} by Match"),
            width="stretch",
        )

    ma.section_heading("Worst Transition-Risk Matches")
    if match_summary_against.empty:
        st.caption("No match-level transition-risk data is available for this selection.")
    else:
        worst = match_summary_against.sort_values("xG From Transition", ascending=False).head(5)
        worst_cols = [col for col in ["Match Label", "Transition Sequences", "Shots From Transition", "xG From Transition", "Avg Transition Duration"] if col in worst]
        st.dataframe(worst[worst_cols], width="stretch", hide_index=True)
        st.caption("The fixtures where this team conceded the most transition xG in the selected window -- a starting point for opposition-transition review.")

    ma.section_heading("Transition Map (Against)")
    map_match_options_against = (
        {str(row["MatchId"]): row.get("Match Label", str(row["MatchId"])) for _, row in match_summary_against.iterrows()}
        if not match_summary_against.empty else {}
    )
    if map_match_options_against:
        map_match_id_against = st.selectbox(
            "Match", list(map_match_options_against), format_func=lambda mid: map_match_options_against.get(mid, mid), key="transitions_against_map_match"
        )
        map_events_against = transitions_against[transitions_against["MatchId"].astype(str).eq(map_match_id_against)]
        st.plotly_chart(_transition_map(map_events_against, "Opponent: Transition Actions vs Outcome"), width="stretch")
    else:
        st.info("No matches with conceded transition sequences are available to map.")

with st.expander("Terminology and methodology"):
    st.markdown(
        """
        - **ATTACKING_TRANSITION phase**: Impect's own event-level tag for the moments right after a team regains
          possession, before play settles into a structured attack.
        - **Transition sequence**: all events sharing one Sequence Index that include at least one transition-phase action.
        - **For / Against**: "For" is the selected team's own transition play; "Against" is the opponent's transition
          play inside the same selected fixtures -- what happened immediately after the selected team lost the ball.
          Against is a same-fixtures proxy for transition risk, not a season-wide opponent scouting sample.
        - **Threat**: the positive parts of PXT Pass and PXT Shot summed across an action or player; a proxy for danger
          created, not an official Impect KPI.
        - **Zone of Origin**: the pitch third the transition sequence started in, normalised so the team in possession
          always attacks left to right.
        - **Duration**: seconds between the first and last transition-phase event timestamp within a sequence; an
          approximation of how long the transition window lasted, not a provider-defined transition-speed metric.
        """
    )
