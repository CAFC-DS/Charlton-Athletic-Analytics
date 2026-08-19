# =============================================================================
# PLAYER BUILD-UP INVOLVEMENT - one player's involvement over time and phase
# =============================================================================
# The team-level Build-Up Involvement page (Match Analysis) gives one number
# per player over a chosen match window. This page starts with a squad-wide
# ranking so the shape of the data is visible before picking anyone, then
# takes a single player and asks: where on the pitch are they involved, what
# kind of actions make up that involvement, is it actually productive, has it
# changed match to match across a season, and which part of the match are
# they most involved in.
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charting, data, pitch, ui
from utils import match_analysis as ma
from utils import player_analysis as pa
from utils import possession_analysis as poss


RED = ui.CHARLTON_RED
DARK = ui.CHARLTON_BLACK
GREEN = "#16803c"
AMBER = "#d89216"
GREY = "#7a7f87"
POSITION_COLOURS = {
    "Goalkeeper": "#c69214",
    "Centre Back": DARK,
    "Full Back": "#2563eb",
    "Defensive Midfielder": "#7a4dc4",
    "Central Midfielder": RED,
    "Attacking Midfielder": "#e04f9f",
    "Forward / Winger": GREEN,
    "Outfield": GREY,
}
ZONE_ORDER = ["Defensive Third", "Middle Third", "Final Third"]
ZONE_COLOURS = {"Defensive Third": DARK, "Middle Third": AMBER, "Final Third": RED}
ACTION_COLOURS = {
    "PASS": RED,
    "RECEPTION": "#2563eb",
    "DRIBBLE": GREEN,
    "DUEL": AMBER,
    "INTERCEPTION": "#7a4dc4",
    "CLEARANCE": DARK,
}


def _pbi_css() -> None:
    st.markdown(
        """
        <style>
        .pbi-summary-card {
            background: var(--ss-panel);
            border: 1px solid var(--ss-border);
            border-radius: 8px;
            border-top: 3px solid var(--ss-accent);
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-height: 100px;
            padding: 14px 16px;
        }
        .pbi-summary-label {
            color: var(--ss-muted);
            font-size: 0.85rem;
            font-weight: 650;
            line-height: 1.25;
            margin-bottom: 12px;
        }
        .pbi-summary-value {
            color: var(--ss-ink);
            display: block;
            font-size: clamp(1.25rem, 1.7vw, 1.65rem);
            font-weight: 400;
            letter-spacing: -0.02em;
            line-height: 1.1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _summary_card(label: str, value: object) -> None:
    st.markdown(
        f"""
        <div class="pbi-summary-card">
            <div class="pbi-summary-label">{ui.esc(label)}</div>
            <div class="pbi-summary-value">{ui.esc(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def _squad_ranking_figure(involvement: pd.DataFrame, positions: pd.DataFrame, selected_player: str, title: str) -> go.Figure:
    fig = go.Figure()
    if involvement.empty:
        return charting.polish_figure(fig, title, height=420)
    plot_df = involvement.merge(positions, on=["PlayerId", "Player"], how="left")
    plot_df["Role Group"] = plot_df["Position"].apply(pa.position_group)
    plot_df = plot_df.sort_values("Build-Up Involvement %", ascending=True)
    colours = [
        RED if player == selected_player else POSITION_COLOURS.get(role, GREY)
        for player, role in zip(plot_df["Player"], plot_df["Role Group"], strict=False)
    ]
    line_widths = [2.5 if player == selected_player else 0.8 for player in plot_df["Player"]]
    minutes = plot_df["Minutes"] if "Minutes" in plot_df else pd.Series(np.nan, index=plot_df.index)
    customdata = np.stack([plot_df["Role Group"], plot_df["Sequences Touched"], minutes.fillna(-1)], axis=-1)
    fig.add_trace(
        go.Bar(
            x=plot_df["Build-Up Involvement %"],
            y=plot_df["Player"],
            orientation="h",
            marker=dict(color=colours, line=dict(color=DARK, width=line_widths)),
            text=[f"{value:.0f}%" for value in plot_df["Build-Up Involvement %"]],
            textposition="outside",
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                "<b>%{y}</b> · %{customdata[0]}<br>Involvement: %{x:.1f}%"
                "<br>Sequences touched: %{customdata[1]:.0f}<br>Minutes: %{customdata[2]:.0f}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="Build-Up Involvement %", range=[0, max(float(plot_df["Build-Up Involvement %"].max()) * 1.15, 10)])
    fig.update_yaxes(title="")
    fig.update_layout(showlegend=False)
    return charting.polish_figure(
        fig, title, height=charting.horizontal_bar_height(len(plot_df), min_height=460, row_height=28, max_height=920)
    )


def _touch_map_figure(touches: pd.DataFrame, player_name: str, title: str) -> go.Figure:
    fig = pitch.pitch_figure(title, height=560, legend=True)
    if touches.empty:
        fig.add_annotation(x=0, y=0, text="No build-up touch locations", showarrow=False, font=dict(size=16, color=GREY))
        return fig
    for zone in ZONE_ORDER:
        group = touches[touches["Zone"].eq(zone)]
        if group.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=group["Start X"],
                y=group["Start Y"],
                mode="markers",
                name=zone,
                marker=dict(size=9, color=ZONE_COLOURS.get(zone, GREY), opacity=0.75, line=dict(color="#ffffff", width=0.8)),
                customdata=np.stack([group["Action"].fillna(group["Action Type"])], axis=-1),
                hovertemplate="%{customdata[0]}<extra></extra>",
            )
        )
    fig.update_layout(margin=dict(l=28, r=94, t=104, b=42))
    return fig


def _action_breakdown_figure(touches: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if touches.empty:
        return charting.polish_figure(fig, title, height=360)
    summary = touches.groupby("Action Type", as_index=False).size().rename(columns={"size": "Actions"})
    summary = summary.sort_values("Actions", ascending=True)
    colours = [ACTION_COLOURS.get(str(action).upper(), GREY) for action in summary["Action Type"]]
    fig.add_trace(
        go.Bar(
            x=summary["Actions"],
            y=summary["Action Type"].str.title(),
            orientation="h",
            marker=dict(color=colours, line=dict(color="#ffffff", width=1)),
            text=summary["Actions"],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_xaxes(title="Actions within build-up sequences")
    fig.update_yaxes(title="")
    fig.update_layout(showlegend=False)
    return charting.polish_figure(fig, title, height=charting.horizontal_bar_height(len(summary), min_height=300, row_height=32, max_height=520))


def _productivity_figure(productivity: pd.DataFrame, player_name: str, title: str) -> go.Figure:
    fig = go.Figure()
    if productivity.empty:
        return charting.polish_figure(fig, title, height=380)
    plot_df = productivity[productivity["Stage"].ne("Build-Up Sequences")]
    fig.add_trace(
        go.Bar(y=plot_df["Stage"], x=plot_df["Team Conversion %"], orientation="h", name="Team (all sequences)", marker_color="#98a2b3")
    )
    fig.add_trace(
        go.Bar(y=plot_df["Stage"], x=plot_df["Player Conversion %"], orientation="h", name=f"{player_name} touched", marker_color=RED)
    )
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.08, x=0))
    fig.update_xaxes(title="Conversion %")
    fig.update_yaxes(title="")
    return charting.polish_figure(fig, title, height=380)


def _rolling_average(values: pd.Series, window: int = 5) -> pd.Series:
    return values.rolling(window=window, min_periods=1).mean()


def _match_trend_figure(match_trend: pd.DataFrame, fixture_context: pd.DataFrame, player_name: str, team_average: float, title: str) -> go.Figure:
    fig = go.Figure()
    if match_trend.empty:
        return charting.polish_figure(fig, title, height=440)
    plot_df = match_trend.copy()
    plot_df["MatchId"] = plot_df["MatchId"].astype(str)
    context = fixture_context.copy()
    context["MatchId"] = context["MatchId"].astype(str)
    plot_df = plot_df.merge(context[["MatchId", "Date", "Opponent", "Venue"]], on="MatchId", how="left")
    plot_df = plot_df.sort_values("Date")
    if plot_df.empty:
        return charting.polish_figure(fig, title, height=440)

    plot_df["Match Label"] = (
        pd.to_datetime(plot_df["Date"], errors="coerce").dt.strftime("%d %b %Y").fillna("Date unknown")
        + " · " + plot_df["Venue"].fillna("") + " vs " + plot_df["Opponent"].fillna("Unknown")
    )
    plot_df["Rolling"] = _rolling_average(plot_df["Involvement %"])
    customdata = np.stack([plot_df["Match Label"], plot_df["Sequences Touched"], plot_df["Sequences"]], axis=-1)
    fig.add_trace(
        go.Bar(
            x=plot_df["Date"],
            y=plot_df["Involvement %"],
            name="Match involvement",
            marker=dict(color="#d8dee8"),
            customdata=customdata,
            hovertemplate=(
                "%{customdata[0]}<br>Involvement: %{y:.1f}%"
                "<br>Touched %{customdata[1]:.0f} of %{customdata[2]:.0f} build-up sequences<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["Date"], y=plot_df["Rolling"], mode="lines", name="5-match rolling average",
            line=dict(color=RED, width=2.6),
        )
    )
    if pd.notna(team_average):
        fig.add_hline(
            y=team_average, line=dict(color=GREY, width=1.5, dash="dash"),
            annotation_text=f"Team average: {team_average:.0f}%",
        )
    fig.update_xaxes(title="Match")
    fig.update_yaxes(title="Build-Up Involvement %", range=[0, 100])
    fig.update_layout(legend=dict(orientation="h", y=1.08, x=0))
    return charting.polish_figure(fig, title, height=460)


def _time_window_figure(window_breakdown: pd.DataFrame, player_name: str, title: str) -> go.Figure:
    fig = go.Figure()
    if window_breakdown.empty:
        return charting.polish_figure(fig, title, height=400)
    customdata = np.stack([window_breakdown["Sequences Touched"], window_breakdown["Sequences"]], axis=-1)
    fig.add_trace(
        go.Bar(
            x=window_breakdown["Window"].astype(str),
            y=window_breakdown["Involvement %"],
            marker=dict(color=RED, line=dict(color="#ffffff", width=1)),
            text=[f"{value:.0f}%" for value in window_breakdown["Involvement %"]],
            textposition="outside",
            customdata=customdata,
            hovertemplate=(
                "<b>%{x} min</b><br>Involvement: %{y:.1f}%"
                "<br>Touched %{customdata[0]:.0f} of %{customdata[1]:.0f} build-up sequences<extra></extra>"
            ),
        )
    )
    average = float(window_breakdown["Involvement %"].mean())
    if pd.notna(average):
        fig.add_hline(y=average, line=dict(color=GREY, width=1.5, dash="dash"), annotation_text=f"Player average: {average:.0f}%")
    fig.update_xaxes(title="Match-minute window")
    fig.update_yaxes(title="Build-Up Involvement %", range=[0, 100])
    fig.update_layout(showlegend=False)
    return charting.polish_figure(fig, title, height=420)


pa.page_header(
    "Player Build-Up Involvement",
    "See the whole squad's build-up involvement at a glance, then drill into one player: where on the pitch they're "
    "involved, what kind of actions make it up, whether that involvement is actually productive, how it has moved "
    "match to match across a season, and which phase of the match they're most involved in.",
)
_pbi_css()

control_cols = st.columns([1.0, 1.2, 0.9, 0.9])
with control_cols[0]:
    seasons = data.list_seasons().get("matches", [])
    if not seasons:
        st.warning("No match seasons are available.")
        st.stop()
    preferred_season = data.preferred_season(seasons)
    season = st.selectbox("Season", seasons, index=seasons.index(preferred_season), key="pbi_season")

matches = ma.load_matches(season)
if matches.empty:
    st.warning("No match data is available for this season.")
    st.stop()

teams = _team_options(matches)
if not teams:
    st.warning("No teams are available from the selected match data.")
    st.stop()

with control_cols[1]:
    team_name = st.selectbox("Team", teams, index=_default_team_index(teams), key="pbi_team")

all_fixtures = _team_fixture_rows(matches, team_name)
if all_fixtures.empty:
    st.warning("No fixtures are available for the selected team.")
    st.stop()

with control_cols[2]:
    window = st.selectbox("Match window", ["Full Season", "Last 10", "Last 5"], key="pbi_window")
with control_cols[3]:
    bucket_size = st.selectbox("Time-window size (min)", [10, 15, 30], index=1, key="pbi_bucket_size")

venue_fixtures = all_fixtures
if window == "Last 10":
    venue_fixtures = venue_fixtures.tail(10)
elif window == "Last 5":
    venue_fixtures = venue_fixtures.tail(5)
selected_match_ids = venue_fixtures["MatchId"].astype(str).tolist()

with st.spinner("Loading build-up sequence events..."):
    events = data.load_match_events(season=season, team=team_name, match_ids=selected_match_ids, limit=120000)
if len(events) >= 120000:
    st.warning("The selected-window event pull reached its 120,000-row cap; totals may be incomplete.")
if events.empty:
    st.info("No event-level rows are available for the selected fixtures.")
    st.stop()

buildup_keys = poss.buildup_sequence_keys(events)
if buildup_keys.empty:
    st.info("No build-up sequences are available for the selected fixtures.")
    st.stop()

total_sequences = buildup_keys[["MatchId", "Sequence Index"]].drop_duplicates().shape[0]
if total_sequences < poss.MIN_SEQUENCES_FOR_RANKING:
    st.warning(
        f"Only {total_sequences} build-up sequences are available across {len(venue_fixtures)} match(es) in this "
        "selection -- most of the charts below will look flat or tied, because with this few sequences most players "
        "who touch the ball at all land on the same 1-in-N share (e.g. 1 of 8 sequences is 12.5% no matter who it "
        "is). This isn't a bug -- it's genuinely too small a sample to differentiate players yet. Pick a fuller "
        "season or a wider match window (e.g. Full Season) for a meaningful ranking."
    )

minutes_lookup = data.load_match_player_minutes(season=season, team=team_name)
if not minutes_lookup.empty:
    minutes_lookup = minutes_lookup.groupby(["PlayerId"], as_index=False).agg(Minutes=("Minutes", "sum"))
season_involvement = poss.player_involvement(events, buildup_keys, minutes_lookup)
if season_involvement.empty:
    st.info("No player-level build-up data is available for the selected fixtures.")
    st.stop()
positions = poss.player_positions(events)
team_average = float(season_involvement["Build-Up Involvement %"].mean())
default_player = season_involvement.iloc[0]["Player"]

ma.section_heading(f"{team_name}: Squad Build-Up Ranking")
player_roster = season_involvement["Player"].tolist()
default_index = player_roster.index(default_player) if default_player in player_roster else 0
player_name = st.selectbox("Player (highlighted below in red)", player_roster, index=default_index, key="pbi_player")
st.plotly_chart(
    _squad_ranking_figure(season_involvement, positions, player_name, f"{team_name}: Build-Up Involvement, Every Player · {window}"),
    width="stretch",
)
st.caption(
    "Every player who touched a build-up sequence in the selected window, coloured by position group. The selected "
    "player is highlighted in red with a bold outline; pick anyone from the dropdown above to drill into their detail below."
)

player_id = season_involvement.loc[season_involvement["Player"].eq(player_name), "PlayerId"].iloc[0]
player_position = positions.loc[positions["PlayerId"].eq(player_id), "Position"]
player_role = pa.position_group(player_position.iloc[0]) if not player_position.empty else "Outfield"

match_trend = poss.player_match_involvement(events, buildup_keys, player_id)
match_trend = match_trend[match_trend["Sequences"] > 0]
window_breakdown = poss.player_time_window_involvement(events, buildup_keys, player_id, bucket_size=bucket_size)
touches = poss.player_touch_locations(events, buildup_keys, player_id)
productivity = poss.player_sequence_productivity(events, buildup_keys, player_id)
player_season_row = season_involvement[season_involvement["Player"].eq(player_name)]

ma.section_heading(f"{player_name}: Season Snapshot")
snapshot_cols = st.columns(4)
with snapshot_cols[0]:
    season_pct = float(player_season_row["Build-Up Involvement %"].iloc[0]) if not player_season_row.empty else 0.0
    _summary_card("Build-Up Involvement % (window)", f"{season_pct:.1f}%")
with snapshot_cols[1]:
    vs_team = season_pct - team_average
    _summary_card("Vs Team Average", f"{vs_team:+.1f} pp")
with snapshot_cols[2]:
    minutes_value = float(player_season_row["Minutes"].iloc[0]) if not player_season_row.empty and "Minutes" in player_season_row and pd.notna(player_season_row["Minutes"].iloc[0]) else np.nan
    _summary_card("Minutes (window)", f"{minutes_value:.0f}" if pd.notna(minutes_value) else "N/A")
with snapshot_cols[3]:
    _summary_card("Position Group", player_role)
st.caption(
    "Build-up involvement is the share of the team's qualifying build-up sequences (possession-sequences starting "
    "outside the final third) in which this player touched the ball, either as the acting player or the pass "
    "receiver. 'Vs Team Average' compares this player's involvement over the same window to every teammate's."
)

detail_cols = st.columns(2)
with detail_cols[0]:
    ma.section_heading("Where They Touch the Ball")
    st.plotly_chart(_touch_map_figure(touches, player_name, f"{player_name}: Build-Up Touch Locations"), width="stretch")
    st.caption("Each point is one action this player took inside a qualifying build-up sequence, coloured by pitch third.")
with detail_cols[1]:
    ma.section_heading("What Kind of Actions")
    st.plotly_chart(_action_breakdown_figure(touches, f"{player_name}: Build-Up Action Types"), width="stretch")
    st.caption("The event types making up this player's build-up involvement -- e.g. mostly passing and receiving vs. duels, interceptions or dribbles.")

ma.section_heading("Is the Involvement Productive?")
if productivity.empty:
    st.info("No productivity comparison is available for this player in the selected window.")
else:
    st.plotly_chart(
        _productivity_figure(productivity, player_name, f"{player_name}: Sequences They Touch vs All Team Sequences"),
        width="stretch",
    )
    st.caption(
        "Compares how often build-up sequences convert to reaching the final third, a shot, or a goal, split between "
        "every team sequence and only the sequences this player personally touched -- a player can have high volume "
        "involvement that isn't especially productive, or lower volume that punches above its weight."
    )

ma.section_heading("Involvement by Match")
if match_trend.empty:
    st.info("No match-by-match build-up data is available for this player in the selected window.")
else:
    st.plotly_chart(
        _match_trend_figure(match_trend, venue_fixtures, player_name, team_average, f"{player_name}: Build-Up Involvement by Match"),
        width="stretch",
    )
    st.caption(
        "Grey bars are each individual match; the red line is a 5-match rolling average so the underlying trend "
        "isn't lost in match-to-match noise. The dashed line is the team's own average involvement across every "
        "player and match in the same window."
    )

ma.section_heading("Involvement by Match-Minute Window")
if window_breakdown.empty:
    st.info("No time-window build-up data is available for this player in the selected window.")
else:
    st.plotly_chart(
        _time_window_figure(window_breakdown, player_name, f"{player_name}: Build-Up Involvement by Match Phase"),
        width="stretch",
    )
    st.caption(
        "Each bar pools every build-up sequence starting in that match-minute window across every match in the "
        "selected window, then shows this player's involvement rate within it -- e.g. a falling pattern late in "
        "matches can reflect a fading role, tactical changes, or simply fewer minutes played that late."
    )

ma.section_heading("Match-by-Match Detail")
if match_trend.empty:
    st.caption("No detail is available for this selection.")
else:
    detail = match_trend.assign(MatchId=match_trend["MatchId"].astype(str)).merge(
        venue_fixtures.assign(MatchId=venue_fixtures["MatchId"].astype(str))[["MatchId", "Date", "Opponent", "Venue"]],
        on="MatchId", how="left",
    ).sort_values("Date")
    detail["Date"] = pd.to_datetime(detail["Date"], errors="coerce").dt.strftime("%d %b %Y")
    detail["Involvement %"] = detail["Involvement %"].round(1)
    st.dataframe(
        detail[["Date", "Opponent", "Venue", "Sequences", "Sequences Touched", "Involvement %"]],
        width="stretch",
        hide_index=True,
        column_config={
            "Involvement %": st.column_config.ProgressColumn("Involvement %", min_value=0, max_value=100, format="%.1f%%"),
        },
    )

with st.expander("Terminology"):
    st.markdown(
        """
        - **Build-up sequence**: a possession-sequence (one Sequence Index) whose first in-possession event starts
          outside the final third for the selected team.
        - **Build-Up Involvement %**: the share of build-up sequences in which this player is recorded as the acting
          player or the pass receiver at least once.
        - **Match-Minute Window**: build-up sequences are grouped by the match minute they started in, then this
          player's involvement rate is computed within each window separately -- this shows *when* in the match the
          player is most involved, not just an overall season number.
        - **Productivity**: the conversion rate (to final third / shot / goal) of build-up sequences this player
          touched, compared to every build-up sequence for the team, using the same funnel definitions as the
          team-level Build-Up Involvement page.
        - **Position Group**: derived from Impect's recorded position label using the same classifier used on the
          Player Profiles and Percentile Rankings pages, for consistency across the app.
        """
    )
