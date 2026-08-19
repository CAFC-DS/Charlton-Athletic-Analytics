# =============================================================================
# SUBSTITUTION IMPACT - timing, personnel, context and before/after output
# =============================================================================
# Built on Opta F24 substitution events (TypeId 18/19) -- Impect's event feed
# has no substitution rows at all, which is why the previous version of this
# page fell back to a weak "low minutes" rotation proxy. Methodology is
# grounded in the substitution-timing literature, most importantly:
#   - Myers (2012, J. Quant. Analysis in Sports): proposed the well-known
#     "58-73-79 minute" rule for trailing teams (first sub before minute 58,
#     second before 73, third before 79).
#   - Silva & Swartz (2016, J. Quant. Analysis in Sports): a more rigorous
#     Bayesian re-analysis that found NO discernible scoring benefit tied to
#     substitution timing itself once team strength and game state are
#     controlled for -- their reading is that managers are already good at
#     spotting when a player needs replacing, so "no signal" isn't "no
#     value". Both findings are shown side by side below rather than
#     presenting the 58-73-79 rule as settled fact.
#   - Del Corral, Barros & Prieto-Rodriguez (2008) and a 2025 EURO 2024
#     study: losing teams substitute earlier and more offensively; winning
#     teams substitute later and more defensively; only a minority of subs
#     are judged to have a clear positive impact on the result.
# All computation stays inside the Opta provider (events + F7 lineups) --
# cross-referencing Opta players to Impect's richer packing/xT metrics would
# require name-matching across two providers with no shared id, which is
# exactly the kind of silent-wrong-answer risk this app avoids elsewhere.
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charting, data, ui
from utils import match_analysis as ma


RED = ui.CHARLTON_RED
DARK = ui.CHARLTON_BLACK
GREEN = "#16803c"
AMBER = "#d89216"
GREY = "#7a7f87"
LIGHT_GREY = "#d0d5dd"
SCORE_STATE_COLOURS = {"Winning": GREEN, "Drawing": AMBER, "Losing": RED}
SHIFT_TYPE_COLOURS = {"More Attacking": RED, "Like-for-Like": "#98a2b3", "More Defensive": DARK, "Unclear": "#c9cdd4"}
SHOT_TYPE_IDS = (13, 14, 15, 16)
SHOT_ON_TARGET_TYPE_IDS = (15, 16)
PASS_TYPE_ID = 1
DISPOSSESSED_TYPE_ID = 50
TOUCH_TYPE_ID = 61
MYERS_CRITICAL_MINUTES = [58, 73, 79]


def _subs_css() -> None:
    st.markdown(
        """
        <style>
        .si-summary-card {
            background: var(--ss-panel);
            border: 1px solid var(--ss-border);
            border-radius: 8px;
            border-top: 3px solid var(--ss-accent);
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-height: 100px;
            padding: 14px 16px;
        }
        .si-summary-label {
            color: var(--ss-muted);
            font-size: 0.85rem;
            font-weight: 650;
            line-height: 1.25;
            margin-bottom: 12px;
        }
        .si-summary-value {
            color: var(--ss-ink);
            display: block;
            font-size: clamp(1.25rem, 1.7vw, 1.65rem);
            font-weight: 400;
            letter-spacing: -0.02em;
            line-height: 1.1;
        }
        .si-note {
            padding: 13px 15px;
            margin: 6px 0 16px;
            border: 1px solid #dfe5ec;
            border-left: 4px solid #2563eb;
            border-radius: 9px;
            background: #f8fafc;
            font-size: 0.89rem;
            color: #253045;
            line-height: 1.5;
        }
        .si-note.caution { border-left-color: #c30017; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _summary_card(label: str, value: object) -> None:
    st.markdown(
        f"""
        <div class="si-summary-card">
            <div class="si-summary-label">{ui.esc(label)}</div>
            <div class="si-summary-value">{ui.esc(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _note(text: str, *, caution: bool = False) -> None:
    css_class = "si-note caution" if caution else "si-note"
    st.markdown(f'<div class="{css_class}">{ui.esc(text)}</div>', unsafe_allow_html=True)


def _team_options(fixtures: pd.DataFrame) -> list[str]:
    values = pd.concat([fixtures.get("Home", pd.Series(dtype=str)), fixtures.get("Away", pd.Series(dtype=str))])
    return sorted(values.dropna().astype(str).loc[lambda s: s.str.strip().ne("")].unique().tolist())


def _default_team_index(teams: list[str]) -> int:
    for index, team in enumerate(teams):
        if "charlton" in team.lower():
            return index
    return 0


def _opta_id(value: object) -> str:
    """Strip Opta's XML-style entity prefix (e.g. "t33" -> "33")."""
    text = "" if value is None else str(value).strip()
    return text[1:] if len(text) > 1 and text[0].isalpha() else text


def _resolve_home_away_names(selected_fixture: pd.Series, match_events: pd.DataFrame) -> tuple[str, str]:
    """Home/Away team names as they actually appear in match_events.

    load_opta_fixtures() sources team names from the raw DVMS fixtures table
    (e.g. "Charlton Athletic"), while load_opta_events()/load_opta_substitutions()
    source them from the F7 teams table (e.g. "Charlton Athletic FC") -- the
    same club, different strings. TeamId is consistent across both once the
    "t" entity prefix is stripped, so it's the reliable join key here.
    """
    fallback_home, fallback_away = str(selected_fixture.get("Home", "")), str(selected_fixture.get("Away", ""))
    if match_events.empty or "TeamId" not in match_events:
        return fallback_home, fallback_away
    id_to_name = (
        match_events[["TeamId", "Team"]].dropna().drop_duplicates().assign(TeamId=lambda d: d["TeamId"].astype(str))
        .set_index("TeamId")["Team"].to_dict()
    )
    home_id = _opta_id(selected_fixture.get("Home Team Id"))
    away_id = _opta_id(selected_fixture.get("Away Team Id"))
    return id_to_name.get(home_id, fallback_home), id_to_name.get(away_id, fallback_away)


def _sub_time_minutes(row: pd.Series) -> float:
    minute = pd.to_numeric(pd.Series([row.get("Minute")]), errors="coerce").iloc[0]
    second = pd.to_numeric(pd.Series([row.get("Second")]), errors="coerce").iloc[0]
    minute = 0.0 if pd.isna(minute) else float(minute)
    second = 0.0 if pd.isna(second) else float(second)
    return minute + second / 60.0


def _fixture_label(row: pd.Series) -> str:
    date = pd.to_datetime(row.get("Date"), errors="coerce")
    date_text = date.strftime("%d %b %Y") if pd.notna(date) else "Undated"
    return f"{date_text} · {row.get('Home')} vs {row.get('Away')}"


def _timeline_figure(match_subs: pd.DataFrame, goals: pd.DataFrame, home: str, away: str, title: str) -> go.Figure:
    fig = go.Figure()
    lanes = {home: 1, away: 0}
    for team_name, y in lanes.items():
        fig.add_hline(y=y, line=dict(color="#e2e7ee", width=14), layer="below")

    team_subs = match_subs.copy()
    team_subs["_Time"] = team_subs.apply(_sub_time_minutes, axis=1)
    team_subs["_Y"] = team_subs["Team"].map(lanes)
    colours = [SHIFT_TYPE_COLOURS.get(shift, GREY) for shift in team_subs["Shift Type"]]
    customdata = np.stack(
        [
            team_subs["Team"],
            team_subs["Player Off"],
            team_subs["Player On"],
            team_subs["Position Off"].fillna("Unknown"),
            team_subs["Position On"].fillna("Unknown"),
            team_subs["Shift Type"],
            team_subs["Score State"],
            team_subs["Sub Number"],
        ],
        axis=-1,
    )
    fig.add_trace(
        go.Scatter(
            x=team_subs["_Time"],
            y=team_subs["_Y"],
            mode="markers",
            marker=dict(size=20, color=colours, line=dict(color="#ffffff", width=2), symbol="circle"),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b> · sub #%{customdata[7]}"
                "<br>%{customdata[1]} (%{customdata[3]}) OFF"
                "<br>%{customdata[2]} (%{customdata[4]}) ON"
                "<br>%{customdata[5]} · %{customdata[6]} at the time<extra></extra>"
            ),
            showlegend=False,
        )
    )

    if not goals.empty:
        goals = goals.copy()
        goals["_Time"] = goals.apply(_sub_time_minutes, axis=1)
        goals["_Y"] = goals["Team"].map(lanes)
        fig.add_trace(
            go.Scatter(
                x=goals["_Time"],
                y=goals["_Y"],
                mode="markers",
                marker=dict(size=16, color=GREEN, symbol="star", line=dict(color="#ffffff", width=1.5)),
                customdata=np.stack([goals["Team"], goals["Player"]], axis=-1),
                hovertemplate="<b>Goal</b> · %{customdata[0]}<br>%{customdata[1]}<extra></extra>",
                name="Goal",
                showlegend=True,
            )
        )

    for label, colour in SHIFT_TYPE_COLOURS.items():
        if label == "Unclear":
            continue
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=colour), name=label, showlegend=True)
        )

    latest_marker = pd.concat([team_subs["_Time"], goals["_Time"] if not goals.empty else pd.Series(dtype=float)])
    axis_max = max(float(latest_marker.max()) + 4 if not latest_marker.empty else 100.0, 100.0)
    fig.update_xaxes(title="Match minute", range=[-2, axis_max], dtick=15)
    fig.update_yaxes(
        tickvals=[0, 1], ticktext=[away, home], range=[-0.6, 1.6], title="",
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0))
    return charting.polish_figure(fig, title, height=360)


def _shift_summary_figure(subs: pd.DataFrame, group_col: str, title: str) -> go.Figure:
    fig = go.Figure()
    if subs.empty:
        return charting.polish_figure(fig, title, height=380)
    counts = subs.groupby([group_col, "Shift Type"]).size().reset_index(name="Count")
    for shift_type, colour in SHIFT_TYPE_COLOURS.items():
        group = counts[counts["Shift Type"].eq(shift_type)]
        if group.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=group[group_col],
                y=group["Count"],
                name=shift_type,
                marker=dict(color=colour, line=dict(color="#ffffff", width=1)),
                hovertemplate=f"%{{x}}<br>{shift_type}: " + "%{y}<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title=group_col)
    fig.update_yaxes(title="Substitutions", rangemode="tozero")
    return charting.polish_figure(fig, title, height=420)


def _timing_histogram(subs: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if subs.empty:
        return charting.polish_figure(fig, title, height=420)
    plot_df = subs.copy()
    plot_df["_Time"] = plot_df.apply(_sub_time_minutes, axis=1)
    for state, colour in SCORE_STATE_COLOURS.items():
        group = plot_df[plot_df["Score State"].eq(state)]
        if group.empty:
            continue
        fig.add_trace(
            go.Histogram(
                x=group["_Time"],
                name=state,
                marker=dict(color=colour),
                xbins=dict(start=0, end=100, size=5),
                opacity=0.85,
            )
        )
    for minute in MYERS_CRITICAL_MINUTES:
        fig.add_vline(x=minute, line=dict(color=DARK, width=1.3, dash="dot"))
    fig.add_annotation(
        x=(MYERS_CRITICAL_MINUTES[0] + MYERS_CRITICAL_MINUTES[-1]) / 2,
        y=1.06,
        yref="paper",
        text="Myers (2012) 58 / 73 / 79 reference minutes",
        showarrow=False,
        font=dict(size=11, color=GREY),
    )
    fig.update_layout(barmode="stack", legend=dict(orientation="h", y=1.14, x=0))
    fig.update_xaxes(title="Match minute", range=[0, 100], dtick=15)
    fig.update_yaxes(title="Substitutions", rangemode="tozero")
    return charting.polish_figure(fig, title, height=440)


def _goal_timing_figure(goal_minutes: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    if goal_minutes.empty:
        return charting.polish_figure(fig, title, height=360)
    bins = list(range(0, 106, 15))
    labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
    bucketed = pd.cut(goal_minutes, bins=bins, labels=labels, right=False, include_lowest=True)
    counts = bucketed.value_counts().reindex(labels).fillna(0)
    share = counts / max(float(counts.sum()), 1) * 100
    fig.add_trace(
        go.Bar(
            x=labels,
            y=share,
            marker=dict(color=RED, line=dict(color="#ffffff", width=1)),
            text=[f"{value:.0f}%" for value in share],
            textposition="outside",
            hovertemplate="%{x} min<br>Share of goals: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_xaxes(title="15-minute segment")
    fig.update_yaxes(title="Share of goals scored", ticksuffix="%")
    fig.update_layout(showlegend=False)
    return charting.polish_figure(fig, title, height=380)


def _window_stats(events: pd.DataFrame, team_name: str, centre_seconds: float, window_seconds: float, match_end_seconds: float) -> dict[str, dict[str, float]]:
    working = events.copy()
    working["_Time"] = pd.to_numeric(working["Minute"], errors="coerce").fillna(0) * 60 + pd.to_numeric(
        working["Second"], errors="coerce"
    ).fillna(0)
    before_start = max(centre_seconds - window_seconds, 0.0)
    after_end = min(centre_seconds + window_seconds, match_end_seconds)

    def _stats_for(team: str, start: float, end: float) -> dict[str, float]:
        window = working[(working["Team"].astype(str).eq(team)) & (working["_Time"] >= start) & (working["_Time"] < end)]
        shots = window["TypeId"].isin(SHOT_TYPE_IDS).sum()
        shots_on_target = window["TypeId"].isin(SHOT_ON_TARGET_TYPE_IDS).sum()
        passes = window["TypeId"].eq(PASS_TYPE_ID)
        pass_attempts = int(passes.sum())
        pass_completed = int((passes & pd.to_numeric(window["Outcome"], errors="coerce").eq(1)).sum())
        touches = int(window["TypeId"].eq(TOUCH_TYPE_ID).sum())
        dispossessed = int(window["TypeId"].eq(DISPOSSESSED_TYPE_ID).sum())
        return {
            "Shots": float(shots),
            "Shots On Target": float(shots_on_target),
            "Pass Attempts": float(pass_attempts),
            "Pass Completion %": (pass_completed / pass_attempts * 100) if pass_attempts else np.nan,
            "Touches": float(touches),
            "Dispossessed": float(dispossessed),
        }

    return {
        "before": _stats_for(team_name, before_start, centre_seconds),
        "after": _stats_for(team_name, centre_seconds, after_end),
        "window_before_min": (centre_seconds - before_start) / 60,
        "window_after_min": (after_end - centre_seconds) / 60,
    }


def _before_after_figure(before: dict[str, float], after: dict[str, float], title: str) -> go.Figure:
    fig = go.Figure()
    metrics = ["Shots", "Shots On Target", "Pass Attempts", "Pass Completion %", "Touches", "Dispossessed"]
    before_values = [before.get(m, 0) for m in metrics]
    after_values = [after.get(m, 0) for m in metrics]
    fig.add_trace(
        go.Bar(
            y=metrics, x=before_values, orientation="h", name="Before", marker_color="#98a2b3",
            text=[f"{v:.1f}" if pd.notna(v) else "N/A" for v in before_values], textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            y=metrics, x=after_values, orientation="h", name="After", marker_color=RED,
            text=[f"{v:.1f}" if pd.notna(v) else "N/A" for v in after_values], textposition="outside",
        )
    )
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Value in window")
    fig.update_yaxes(title="")
    return charting.polish_figure(fig, title, height=380)


ma.page_header(
    "Substitution Impact",
    "Exact substitution timing and personnel from Opta F24 events, read against game context (score state, position "
    "swapped, opponent) and a transparent before/after output comparison -- plus season-level substitution patterns "
    "and a league-wide impact-substitute leaderboard.",
    "Opta F24 substitution (Player Off/On), goal and shot events, cross-referenced with Opta F7 lineup positions. "
    "Impect's event feed has no substitution rows, so this page uses Opta throughout rather than mixing providers.",
    (
        "Before/after windows are single-occurrence comparisons around a rare event (shots, goals); read them "
        "directionally, not as proof that the substitution caused the change -- the same game-state and time-of-match "
        "effects that drive most goals apply regardless of substitutions (see Methodology below). Player-quality "
        "comparisons stay inside the Opta player-id space; Impect's richer packing/xT metrics aren't cross-referenced "
        "here because there is no shared player id between the two providers, and name-matching risks silent errors."
    ),
)
_subs_css()

if data.USE_MOCK_DATA:
    st.warning("Opta feeds are disabled in demo mode. Set CHARLTON_DATA_MODE=production.")
    st.stop()

match_tab, season_tab = st.tabs(["Match Timeline", "Season Patterns"])

with match_tab:
    fixtures = data.load_opta_fixtures()
    if fixtures.empty:
        st.warning("No Opta fixtures are available to the current Snowflake role.")
        st.stop()
    fixtures = fixtures.sort_values(["Date", "FixtureId"], na_position="last").reset_index(drop=True)

    control_cols = st.columns([0.7, 0.7, 1.3])
    with control_cols[0]:
        season_options = sorted(fixtures["Season"].dropna().astype(str).unique().tolist())
        _charlton_mask = (
            fixtures["Home"].astype(str).str.contains("charlton", case=False, na=False)
            | fixtures["Away"].astype(str).str.contains("charlton", case=False, na=False)
        )

        def _charlton_match_count(season_value: str) -> int:
            return int((_charlton_mask & fixtures["Season"].astype(str).eq(str(season_value))).sum())

        preferred_opta_season = data.preferred_season(season_options, match_count=_charlton_match_count)
        opta_season = st.selectbox(
            "Season", season_options,
            index=season_options.index(preferred_opta_season),
            key="subs_match_season",
        )
    season_fixtures = fixtures[fixtures["Season"].astype(str).eq(str(opta_season))].copy()
    with control_cols[1]:
        team_options_match = _team_options(season_fixtures)
        team_filter = st.selectbox(
            "Team", ["All teams", *team_options_match], index=_default_team_index(team_options_match) + 1, key="subs_match_team"
        )
    filtered_fixtures = season_fixtures
    if team_filter != "All teams":
        filtered_fixtures = season_fixtures[
            season_fixtures["Home"].astype(str).eq(team_filter) | season_fixtures["Away"].astype(str).eq(team_filter)
        ].copy()
    if filtered_fixtures.empty:
        st.info("No fixtures match those filters.")
        st.stop()
    with control_cols[2]:
        fixture_options = filtered_fixtures.index.tolist()
        selected_fixture_idx = st.selectbox(
            "Fixture", fixture_options, index=len(fixture_options) - 1,
            format_func=lambda idx: _fixture_label(filtered_fixtures.loc[idx]), key="subs_match_fixture",
        )
    selected_fixture = filtered_fixtures.loc[selected_fixture_idx]
    fixture_id = selected_fixture["FixtureId"]

    with st.spinner("Loading match events..."):
        match_events = data.load_opta_events(fixture_id, limit=50000)
    if match_events.empty:
        st.info("No Opta event rows are available for this fixture.")
        st.stop()
    home_team, away_team = _resolve_home_away_names(selected_fixture, match_events)

    match_subs_full = data.load_opta_substitutions(season=opta_season)
    match_subs = match_subs_full[match_subs_full["FixtureId"].astype(str).eq(str(fixture_id))].copy()
    goal_events = match_events[match_events["TypeId"].eq(16)][["Team", "Player", "Minute", "Second"]].copy()

    ma.section_heading("Fixture Summary")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        _summary_card("Fixture", f"{home_team} vs {away_team}")
    with metric_cols[1]:
        _summary_card("Substitutions", f"{len(match_subs)}")
    with metric_cols[2]:
        home_goals = int(goal_events["Team"].eq(home_team).sum())
        away_goals = int(goal_events["Team"].eq(away_team).sum())
        _summary_card("Score", f"{home_team} {home_goals} - {away_goals} {away_team}")
    with metric_cols[3]:
        match_end_minutes = float(
            pd.to_numeric(match_events["Minute"], errors="coerce").max()
            + pd.to_numeric(match_events["Second"], errors="coerce").fillna(0).max() / 60
        ) if not match_events.empty else 90.0
        _summary_card("Match Length (incl. stoppage)", f"{match_end_minutes:.0f} min")

    if match_subs.empty:
        st.info("No substitutions are recorded for this fixture.")
    else:
        ma.section_heading("Substitution Timeline")
        st.plotly_chart(
            _timeline_figure(match_subs, goal_events, home_team, away_team, f"{home_team} vs {away_team}: Substitutions and Goals"),
            width="stretch",
        )
        st.caption(
            "Colour is the positional shift: red is a more attacking swap, dark is a more defensive swap, grey is "
            "like-for-like. Stars mark goals. Hover a marker for who came off, who came on, their positions, and the "
            "score state at that moment."
        )

        ma.section_heading("Substitution Table")
        table_cols = [
            "Sub Number", "Team", "Minute", "Second", "Player Off", "Position Off", "Player On", "Position On",
            "Shift Type", "Team Goals At Sub", "Opponent Goals At Sub", "Score State", "Goals After Entry", "Assists After Entry",
        ]
        st.dataframe(match_subs[table_cols].sort_values(["Team", "Sub Number"]), width="stretch", hide_index=True)

        ma.section_heading("Before / After Output Explorer")
        _note(
            "This compares team-level Opta event output in a fixed window either side of one substitution. It is a "
            "single occurrence, not an average over many subs -- treat it as descriptive context for the moment, not "
            "a verdict on whether the substitution worked.",
            caution=True,
        )
        sub_options = match_subs.sort_values(["Minute", "Second"]).index.tolist()
        sub_label_lookup = {
            idx: (
                f"{int(row['Minute'])}' {row['Team']}: {row['Player Off']} → {row['Player On']} "
                f"({row['Shift Type']}, {row['Score State']})"
            )
            for idx, row in match_subs.iterrows()
        }
        explorer_cols = st.columns([2.2, 1])
        with explorer_cols[0]:
            selected_sub_idx = st.selectbox(
                "Substitution", sub_options, format_func=lambda idx: sub_label_lookup.get(idx, str(idx)), key="subs_explorer_choice"
            )
        with explorer_cols[1]:
            window_minutes = st.slider("Window either side (minutes)", 5, 20, 10, step=5, key="subs_explorer_window")

        selected_sub = match_subs.loc[selected_sub_idx]
        centre_seconds = _sub_time_minutes(selected_sub) * 60
        window_stats = _window_stats(match_events, str(selected_sub["Team"]), centre_seconds, window_minutes * 60, match_end_minutes * 60)
        opponent_name = away_team if str(selected_sub["Team"]) == home_team else home_team
        opponent_stats = _window_stats(match_events, opponent_name, centre_seconds, window_minutes * 60, match_end_minutes * 60)

        explorer_metric_cols = st.columns(4)
        with explorer_metric_cols[0]:
            _summary_card("Team subbing", str(selected_sub["Team"]))
        with explorer_metric_cols[1]:
            _summary_card("Score state at sub", str(selected_sub["Score State"]))
        with explorer_metric_cols[2]:
            _summary_card("Window used (before / after)", f"{window_stats['window_before_min']:.0f} / {window_stats['window_after_min']:.0f} min")
        with explorer_metric_cols[3]:
            _summary_card("Positional shift", str(selected_sub["Shift Type"]))

        window_chart_cols = st.columns(2)
        with window_chart_cols[0]:
            st.plotly_chart(
                _before_after_figure(window_stats["before"], window_stats["after"], f"{selected_sub['Team']}: Output Before vs After"),
                width="stretch",
            )
        with window_chart_cols[1]:
            st.plotly_chart(
                _before_after_figure(opponent_stats["before"], opponent_stats["after"], f"{opponent_name} (opponent): Output Before vs After"),
                width="stretch",
            )
        st.caption(
            "The opponent panel shows whether the game changed for the other side too -- a substitution can shift the "
            "match without the subbing team's own raw output moving much, especially defensive or game-management subs."
        )

with season_tab:
    control_cols = st.columns([0.8, 1.0, 0.9])
    with control_cols[0]:
        opta_fixtures_probe = data.load_opta_fixtures()
        season_pool = sorted(opta_fixtures_probe["Season"].dropna().astype(str).unique().tolist()) if not opta_fixtures_probe.empty else []
        if not season_pool:
            st.warning("No Opta seasons are available.")
            st.stop()
        _season_tab_charlton_mask = (
            opta_fixtures_probe["Home"].astype(str).str.contains("charlton", case=False, na=False)
            | opta_fixtures_probe["Away"].astype(str).str.contains("charlton", case=False, na=False)
        )

        def _season_tab_charlton_match_count(season_value: str) -> int:
            return int((_season_tab_charlton_mask & opta_fixtures_probe["Season"].astype(str).eq(str(season_value))).sum())

        preferred_season_choice = data.preferred_season(season_pool, match_count=_season_tab_charlton_match_count)
        season_choice = st.selectbox(
            "Season", season_pool,
            index=season_pool.index(preferred_season_choice),
            key="subs_season_season",
        )
    with st.spinner("Loading season substitution data..."):
        all_season_subs = data.load_opta_substitutions(season=season_choice)
    if all_season_subs.empty:
        st.info("No substitution data is available for this season.")
        st.stop()
    team_pool = sorted(all_season_subs["Team"].dropna().astype(str).unique().tolist())
    with control_cols[1]:
        season_team = st.selectbox(
            "Team focus", ["All teams", *team_pool], index=_default_team_index(team_pool) + 1, key="subs_season_team"
        )
    with control_cols[2]:
        st.caption(f"{len(all_season_subs):,} substitutions across {all_season_subs['FixtureId'].nunique():,} fixtures this season.")

    team_season_subs = all_season_subs if season_team == "All teams" else all_season_subs[all_season_subs["Team"].astype(str).eq(season_team)].copy()

    ma.section_heading("Substitution Timing by Game State")
    st.plotly_chart(_timing_histogram(team_season_subs, f"{season_team}: Substitution Timing by Score State"), width="stretch")
    _note(
        "Myers (2012) proposed subbing before minutes 58/73/79 when losing. Silva & Swartz (2016) re-tested this with "
        "a more rigorous Bayesian model and found no discernible scoring benefit tied to timing itself once team "
        "strength is controlled for -- read the reference lines as what a popular rule of thumb recommends, not as "
        "a proven target to hit."
    )

    ma.section_heading("This Team's Goal-Scoring Rhythm")
    team_goal_minutes = pd.Series(dtype=float)
    if season_team != "All teams":
        with st.spinner("Loading goal timing for this team..."):
            team_goals = data.load_opta_goal_events(season=season_choice, team=season_team)
            team_goals = team_goals[team_goals["Team"].astype(str).eq(season_team)]
        if not team_goals.empty:
            team_goal_minutes = team_goals.apply(_sub_time_minutes, axis=1)
    if team_goal_minutes.empty:
        st.caption(
            "Select a single team above to see when it actually scores across the season -- useful context for "
            "judging whether goals around a substitution are typical for this team's rhythm or unusual."
        )
    else:
        st.plotly_chart(_goal_timing_figure(team_goal_minutes, f"{season_team}: Goals Scored by 15-Minute Segment"), width="stretch")
        st.caption(
            "Scoring intensity rising through a match is a well-replicated pattern in the literature (e.g. Ridder, "
            "Cramer & Hopstaken 1994; Armatas, Yiannakos & Sileloglou 2007). This chart is this team's own rate, not "
            "an imported external baseline, so it reflects this team's actual season."
        )

    ma.section_heading("Positional Shift by Score State")
    st.plotly_chart(
        _shift_summary_figure(team_season_subs, "Score State", f"{season_team}: Substitution Type by Score State"),
        width="stretch",
    )
    st.caption(
        "Del Corral, Barros & Prieto-Rodriguez (2008) found winning teams make more defensive substitutions while "
        "losing teams make more offensive ones. Compare the Winning and Losing columns above to see whether this "
        "team's own pattern matches."
    )

    ma.section_heading("League-Wide Impact Substitute Leaderboard")
    leaderboard = (
        all_season_subs.groupby(["PlayerOnId", "Player On", "Team"], as_index=False)
        .agg(
            Appearances=("Player On", "size"),
            **{
                "Goals After Entry": ("Goals After Entry", "sum"),
                "Assists After Entry": ("Assists After Entry", "sum"),
            },
        )
    )
    leaderboard["Goal Involvements After Entry"] = leaderboard["Goals After Entry"] + leaderboard["Assists After Entry"]
    leaderboard = leaderboard[leaderboard["Goal Involvements After Entry"] > 0].sort_values(
        "Goal Involvements After Entry", ascending=False
    ).head(20)
    if leaderboard.empty:
        st.info("No substitute goal or assist contributions are available for this season.")
    else:
        plot_df = leaderboard.sort_values("Goal Involvements After Entry", ascending=True)
        colours = [RED if "charlton" in str(team).lower() else DARK for team in plot_df["Team"]]
        fig = go.Figure(
            go.Bar(
                x=plot_df["Goals After Entry"], y=plot_df["Player On"], orientation="h", name="Goals",
                marker=dict(color=colours), text=[f"{v:.0f}" for v in plot_df["Goals After Entry"]], textposition="inside",
            )
        )
        fig.add_trace(
            go.Bar(
                x=plot_df["Assists After Entry"], y=plot_df["Player On"], orientation="h", name="Assists",
                marker=dict(color=AMBER), text=[f"{v:.0f}" for v in plot_df["Assists After Entry"]], textposition="inside",
            )
        )
        fig.update_layout(barmode="stack", legend=dict(orientation="h", y=1.05, x=0))
        fig.update_xaxes(title="Goal involvements after entering as a substitute")
        fig.update_yaxes(title="")
        fig = charting.polish_figure(
            fig, f"{season_choice}: Top Impact Substitutes", height=charting.horizontal_bar_height(len(plot_df), min_height=420, row_height=32, max_height=740)
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Counts goals and Opta-flagged assists recorded strictly after a player's own substitution-on timestamp "
            "in the same fixture -- e.g. UEFA EURO 2024 analysis found roughly a fifth of substitutions were judged "
            "to positively affect the result; this is a directly-measured version of that idea from your own data."
        )
        st.dataframe(
            leaderboard[["Player On", "Team", "Appearances", "Goals After Entry", "Assists After Entry", "Goal Involvements After Entry"]],
            width="stretch",
            hide_index=True,
        )

with st.expander("Methodology and sources"):
    st.markdown(
        """
        **Data**: Opta F24 event feed only. Substitutions come from paired "Player Off" (TypeId 18) / "Player On"
        (TypeId 19) events, matched by team, period, minute and second -- CAFC_DB has no substitution-linking
        qualifier, so this pairing was verified directly against real fixture data down to the second before being
        used here. Positions come from Opta F7 lineups (starting position for the outgoing player, specialist bench
        position for the incoming player). Shot, goal, pass, touch and dispossession counts use the well-established
        public Opta F24 type-code taxonomy (Pass=1, Miss=13, Post=14, Attempt Saved=15, Goal=16, Player Off=18,
        Player On=19, Dispossessed=50, Ball Touch=61); CAFC_DB does not expose a labelled type dictionary, so these
        were empirically validated here (pass completion rate, shots/match and goals/match all landed at realistic
        real-world figures) rather than assumed from an unverified source.

        **Why not Impect?** The Impect event feed used throughout the rest of this app has zero substitution rows,
        so timing, personnel and before/after context simply aren't available from it. Cross-referencing Opta
        players against Impect's packing/xT metrics was considered and rejected: the two providers don't share a
        player id, and name-matching risks silently pairing the wrong player.

        **Shift Type**: a coarse positional-group comparison (Goalkeeper < Defender < Midfielder < Forward) between
        the outgoing and incoming player. It is a proxy for *what kind* of change was made, not a read of tactical
        intent -- a like-for-like swap can still be a big tactical shift, and this won't catch that.

        **Before/After windows**: fixed-time comparisons of team-level Opta event totals either side of one
        substitution, capped at kickoff and full time. These are single-occurrence, small-sample comparisons around
        naturally rare events (shots, goals) -- read them descriptively, not as causal proof.

        **Cited research**:
        - Myers, B.R. (2012). *A proposed decision rule for the timing of soccer substitutions.* Journal of
          Quantitative Analysis in Sports, 8, Article 9. -- source of the "58/73/79 minute" rule shown as a
          reference line.
        - Silva, R.M. & Swartz, T.B. (2016). *Analysis of Substitution Times in Soccer.* Journal of Quantitative
          Analysis in Sports, 12(3). -- a more rigorous Bayesian re-analysis that found no discernible timing benefit
          once team strength is controlled for; the reason this page avoids presenting the 58/73/79 rule as proven.
        - Del Corral, J., Barros, C.P. & Prieto-Rodriguez, J. (2008). *The determinants of player substitutions: A
          survival analysis of the Spanish soccer league.* Journal of Sports Economics, 9, 160-172. -- source of the
          winning-teams-sub-defensively / losing-teams-sub-offensively finding tested above.
        - Ridder, G., Cramer, J.S. & Hopstaken, P. (1994). *Down to ten: estimating the effect of a red card in
          soccer.* Journal of the American Statistical Association, 89, 1124-1127. -- early source for goal-scoring
          intensity rising through a match, replicated with this team's own data above rather than reused directly.
        - Frontiers in Sports and Active Living (2025). *The influence of substitution decisions made by national
          team coaches on final match outcomes at UEFA EURO 2024.* -- source for the roughly-one-in-five substitutions
          judged to positively affect the result.
        """
    )
