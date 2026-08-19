# =============================================================================
# MATCH OVERVIEW - match-centre style selected-fixture summary
# =============================================================================
from __future__ import annotations

import math
from textwrap import dedent

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charting, data, match_analysis as ma, pitch, ui


MISSING = "\u2014"
DOT = "\u00b7"
FOOTBALL_ICON = "\u26bd"
YELLOW_CARD_ICON = "\U0001f7e8"
RED_CARD_ICON = "\U0001f7e5"


def _html_fragment(markup: str) -> str:
    return dedent(markup).strip()


def _html(markup: str) -> None:
    st.markdown(_html_fragment(markup), unsafe_allow_html=True)


def _overview_css() -> None:
    _html(
        """
        <style>
        .mo-scorecard {
            background: #ffffff;
            border: 1px solid var(--ss-border);
            border-radius: 12px;
            box-shadow: 0 10px 26px rgba(16, 24, 40, 0.08);
            margin: 10px 0 22px;
            overflow: hidden;
        }

        .mo-scorecard-top {
            align-items: center;
            background: linear-gradient(135deg, #111111 0%, #2a1115 58%, #9c0214 145%);
            color: #ffffff;
            display: flex;
            gap: 14px;
            justify-content: center;
            padding: 12px 18px;
            text-align: center;
        }

        .mo-scorecard-top span {
            color: rgba(255, 255, 255, 0.78);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .mo-score-main {
            align-items: stretch;
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(180px, 230px) minmax(0, 1fr);
            gap: 18px;
            padding: 28px 30px 26px;
        }

        .mo-team {
            align-items: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-width: 0;
            text-align: center;
        }

        .mo-team-side {
            color: var(--ss-muted);
            font-size: 0.78rem;
            font-weight: 850;
            letter-spacing: 0.12em;
            margin-bottom: 10px;
            text-transform: uppercase;
        }

        .mo-team-name {
            color: var(--ss-ink);
            font-size: clamp(1.35rem, 2.4vw, 2.35rem);
            font-weight: 900;
            letter-spacing: -0.03em;
            line-height: 1.04;
            margin: 0;
        }

        .mo-team-substat {
            color: var(--ss-muted);
            font-size: 0.92rem;
            font-weight: 700;
            margin-top: 12px;
        }

        .mo-score-centre {
            align-items: center;
            background: #f8fafc;
            border: 1px solid #e6edf5;
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 18px 14px;
            text-align: center;
        }

        .mo-status {
            background: #111111;
            border-radius: 999px;
            color: #ffffff;
            display: inline-flex;
            font-size: 0.74rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            line-height: 1;
            margin-bottom: 10px;
            padding: 7px 11px;
            text-transform: uppercase;
        }

        .mo-score {
            color: var(--ss-ink);
            font-size: clamp(2.6rem, 5vw, 4.8rem);
            font-weight: 950;
            letter-spacing: -0.08em;
            line-height: 0.92;
            margin-bottom: 10px;
            white-space: nowrap;
        }

        .mo-score-note {
            color: var(--ss-muted);
            font-size: 0.82rem;
            font-weight: 700;
            line-height: 1.35;
        }

        .mo-facts-grid {
            display: grid;
            gap: 12px;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            margin: 8px 0 18px;
        }

        .mo-fact-card {
            background: #ffffff;
            border: 1px solid #e6edf5;
            border-radius: 10px;
            border-top: 3px solid var(--ss-accent);
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-height: 96px;
            padding: 15px 16px;
        }

        .mo-fact-label {
            color: var(--ss-muted);
            font-size: 0.74rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
            text-transform: uppercase;
        }

        .mo-fact-value {
            color: var(--ss-ink);
            font-size: 1.05rem;
            font-weight: 850;
            line-height: 1.2;
        }

        .mo-stat-panel {
            background: #ffffff;
            border: 1px solid #e6edf5;
            border-radius: 12px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            margin: 8px 0 20px;
            overflow: hidden;
        }

        .mo-stat-header,
        .mo-stat-row {
            align-items: center;
            display: grid;
            gap: 12px;
        }

        .mo-stat-header {
            background: #f8fafc;
            border-bottom: 1px solid #e6edf5;
            color: var(--ss-muted);
            font-size: 0.74rem;
            font-weight: 850;
            grid-template-columns: minmax(0, 1fr) minmax(120px, max-content) minmax(0, 1fr);
            letter-spacing: 0.08em;
            padding: 12px 18px;
            text-transform: uppercase;
        }

        .mo-stat-header-team {
            line-height: 1.25;
            overflow-wrap: anywhere;
        }

        .mo-stat-header-team-left {
            text-align: left;
        }

        .mo-stat-header-team-right {
            text-align: right;
        }

        .mo-stat-header-title {
            text-align: center;
            white-space: nowrap;
        }

        .mo-stat-row {
            border-bottom: 1px solid #f0f2f5;
            grid-template-columns: 82px minmax(70px, 1fr) minmax(138px, 170px) minmax(70px, 1fr) 82px;
            padding: 12px 18px;
        }

        .mo-stat-row:last-child {
            border-bottom: 0;
        }

        .mo-stat-value {
            color: var(--ss-ink);
            font-size: 0.98rem;
            font-weight: 850;
            white-space: nowrap;
        }

        .mo-stat-value-left {
            text-align: right;
        }

        .mo-stat-label {
            color: var(--ss-ink);
            font-size: 0.9rem;
            font-weight: 850;
            text-align: center;
        }

        .mo-stat-track {
            background: #edf1f5;
            border-radius: 999px;
            height: 8px;
            overflow: hidden;
            position: relative;
        }

        .mo-stat-track-left .mo-stat-fill {
            right: 0;
        }

        .mo-stat-fill {
            background: var(--ss-accent);
            border-radius: 999px;
            display: block;
            height: 100%;
            min-width: 2px;
            position: absolute;
            top: 0;
        }

        .mo-stat-fill-away {
            background: #111111;
            left: 0;
        }

        .mo-note {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 10px;
            color: #7c2d12;
            font-size: 0.9rem;
            line-height: 1.45;
            margin: 8px 0 18px;
            padding: 12px 14px;
        }

        .mo-shot-type-key {
            align-items: center;
            background: #ffffff;
            border: 1px solid #e6edf5;
            border-radius: 10px;
            color: var(--ss-muted);
            display: flex;
            flex-wrap: wrap;
            gap: 10px 14px;
            margin: -4px 0 14px;
            padding: 9px 11px;
        }

        .mo-shot-type-key-title {
            color: var(--ss-ink);
            font-size: 0.72rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            margin-right: 2px;
            text-transform: uppercase;
        }

        .mo-shot-type-key-item {
            align-items: center;
            display: inline-flex;
            font-size: 0.78rem;
            font-weight: 750;
            gap: 6px;
            white-space: nowrap;
        }

        .mo-shot-key-shape {
            background: #111111;
            display: inline-block;
            height: 10px;
            width: 10px;
        }

        .mo-shot-key-circle {
            border-radius: 999px;
        }

        .mo-shot-key-circle-open {
            background: #ffffff;
            border: 2px solid #111111;
            border-radius: 999px;
        }

        .mo-shot-key-diamond {
            transform: rotate(45deg);
        }

        .mo-shot-key-triangle {
            background: transparent;
            border-bottom: 11px solid #111111;
            border-left: 6px solid transparent;
            border-right: 6px solid transparent;
            display: inline-block;
            height: 0;
            width: 0;
        }

        .mo-link-strip {
            display: grid;
            gap: 10px;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            margin: 6px 0 16px;
        }

        @media (max-width: 860px) {
            .mo-score-main {
                grid-template-columns: 1fr;
            }

            .mo-stat-row {
                grid-template-columns: 58px minmax(54px, 1fr) minmax(112px, 132px) minmax(54px, 1fr) 58px;
                gap: 8px;
                padding-left: 12px;
                padding-right: 12px;
            }

            .mo-stat-header {
                gap: 8px;
                grid-template-columns: minmax(0, 1fr) minmax(88px, max-content) minmax(0, 1fr);
                padding-left: 12px;
                padding-right: 12px;
            }

            .mo-stat-value {
                font-size: 0.86rem;
            }

            .mo-stat-label {
                font-size: 0.82rem;
            }
        }
        </style>
        """
    )


def _to_number(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _metric_text(value: object, value_type: str = "count") -> str:
    number = _to_number(value, np.nan)
    if not math.isfinite(number):
        return MISSING
    if value_type == "percent":
        return f"{number:.1f}%"
    if value_type == "decimal":
        return f"{number:.2f}"
    if value_type == "xg":
        return f"{number:.2f}"
    return f"{number:,.0f}"


def _date_text(value: object) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "Date TBC"
    time_part = timestamp.strftime("%H:%M")
    if time_part == "00:00":
        return timestamp.strftime("%a %d %b %Y")
    return timestamp.strftime("%a %d %b %Y, %H:%M")


def _match_status(row: pd.Series, events: pd.DataFrame) -> str:
    timestamp = pd.to_datetime(row.get("Date"), errors="coerce")
    if pd.notna(timestamp):
        try:
            now = pd.Timestamp.now(tz=timestamp.tz) if timestamp.tz is not None else pd.Timestamp.now()
            if timestamp > now and events.empty:
                return "Fixture"
        except TypeError:
            pass
    return "FT"


def _goal_text(row: pd.Series, side: str) -> str:
    column = "Home Goals" if side == "home" else "Away Goals"
    value = row.get(column)
    return _metric_text(value, "count")


def _shot_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    action_type = events["Action Type"].astype(str).str.upper() if "Action Type" in events else pd.Series("", index=events.index)
    shots = events[action_type.eq("SHOT") | events["Shot xG"].notna()].copy()
    if "Shot xG" in shots:
        shots["Shot xG"] = pd.to_numeric(shots["Shot xG"], errors="coerce").fillna(0)
    if "Post-Shot xG" in shots:
        shots["Post-Shot xG"] = pd.to_numeric(shots["Post-Shot xG"], errors="coerce")
    return shots


def _pass_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    action_type = events["Action Type"].astype(str).str.upper() if "Action Type" in events else pd.Series("", index=events.index)
    passes = events[action_type.eq("PASS") | events["Receiver"].notna()].copy()
    for column in ["Start X", "Start Y", "End X", "End Y", "PXT Pass", "Pass Distance"]:
        if column in passes:
            passes[column] = pd.to_numeric(passes[column], errors="coerce")
    if {"Start X", "End X"}.issubset(passes.columns):
        passes["Territory Gain"] = passes["End X"] - passes["Start X"]
    else:
        passes["Territory Gain"] = np.nan
    result = passes["Result"].astype(str).str.upper() if "Result" in passes else pd.Series("", index=passes.index)
    passes["_Completed"] = result.eq("SUCCESS")
    if {"Start X", "End X"}.issubset(passes.columns):
        passes["_Final Third Entry"] = (passes["Start X"] < pitch.FINAL_THIRD_X) & (passes["End X"] >= pitch.FINAL_THIRD_X)
    else:
        passes["_Final Third Entry"] = False
    return passes


def _team_frames(frame: pd.DataFrame, teams: list[str]) -> dict[str, pd.DataFrame]:
    if frame.empty or "Team" not in frame:
        return {team: pd.DataFrame(columns=frame.columns) for team in teams}
    return {team: frame[frame["Team"].astype(str) == str(team)].copy() for team in teams}


def _is_goal(shots: pd.DataFrame) -> pd.Series:
    if shots.empty:
        return pd.Series(dtype="bool")
    result = shots["Result"].astype(str).str.upper() if "Result" in shots else pd.Series("", index=shots.index)
    action = shots["Action"].astype(str).str.upper() if "Action" in shots else pd.Series("", index=shots.index)
    return result.eq("SUCCESS") | action.isin(["GOAL", "SHOT_GOAL"])


def _is_on_target(shots: pd.DataFrame) -> pd.Series:
    if shots.empty:
        return pd.Series(dtype="bool")
    on_target = _is_goal(shots)
    if "Post-Shot xG" in shots:
        on_target = on_target | pd.to_numeric(shots["Post-Shot xG"], errors="coerce").notna()
    action = shots["Action"].astype(str).str.upper() if "Action" in shots else pd.Series("", index=shots.index)
    return on_target | action.str.contains("SAVED|ON_TARGET|SAVE", regex=True, na=False)


def _sum(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _count_contains(frame: pd.DataFrame, columns: list[str], keywords: list[str]) -> int:
    if frame.empty:
        return 0
    mask = pd.Series(False, index=frame.index)
    pattern = "|".join(keywords)
    for column in columns:
        if column in frame:
            mask = mask | frame[column].astype(str).str.upper().str.contains(pattern, regex=True, na=False)
    return int(mask.sum())


def _truthy_flag(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(False, index=series.index)
    text = series.astype(str).str.strip().str.upper()
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return text.isin(["TRUE", "YES", "Y", "T"]) | numeric.ne(0)


def _has_label(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(False, index=series.index)
    text = series.astype(str).str.strip().str.upper()
    return series.notna() & ~text.isin(["", "NONE", "NAN", "NULL", "FALSE", "0"])


def _overview_stats(row: pd.Series, events: pd.DataFrame) -> pd.DataFrame:
    home = str(row.get("Home", "Home"))
    away = str(row.get("Away", "Away"))
    teams = [home, away]
    team_events = _team_frames(events, teams)
    shots = _shot_events(events)
    passes = _pass_events(events)
    team_shots = _team_frames(shots, teams)
    team_passes = _team_frames(passes, teams)
    rows: list[dict[str, object]] = []

    def add(label: str, home_value: float, away_value: float, value_type: str = "count", show_zero: bool = False) -> None:
        if not show_zero and abs(_to_number(home_value)) < 1e-9 and abs(_to_number(away_value)) < 1e-9:
            return
        rows.append({"Stat": label, "Home": home_value, "Away": away_value, "Type": value_type})

    add("Goals", row.get("Home Goals"), row.get("Away Goals"), show_zero=True)

    if events.empty:
        return pd.DataFrame(rows)

    add("Expected Goals (xG)", _sum(team_shots[home], "Shot xG"), _sum(team_shots[away], "Shot xG"), "xg")
    add("Post-Shot xG", _sum(team_shots[home], "Post-Shot xG"), _sum(team_shots[away], "Post-Shot xG"), "xg")
    add("Shots", len(team_shots[home]), len(team_shots[away]), show_zero=True)
    add("Shots On Target", int(_is_on_target(team_shots[home]).sum()), int(_is_on_target(team_shots[away]).sum()), show_zero=True)
    add(
        "Big Chances",
        int((pd.to_numeric(team_shots[home].get("Shot xG", pd.Series(dtype=float)), errors="coerce").fillna(0) >= 0.30).sum()),
        int((pd.to_numeric(team_shots[away].get("Shot xG", pd.Series(dtype=float)), errors="coerce").fillna(0) >= 0.30).sum()),
    )

    home_passes = team_passes[home]
    away_passes = team_passes[away]
    home_completed = int(home_passes["_Completed"].sum()) if "_Completed" in home_passes else 0
    away_completed = int(away_passes["_Completed"].sum()) if "_Completed" in away_passes else 0

    # Add Possession %
    fixture_id = None
    try:
        match_date = pd.to_datetime(row.get("Date"), errors="coerce")
        if pd.notna(match_date):
            season = str(row.get("Season"))
            optas = data.load_opta_fixtures(season=season)
            optas["_Date"] = pd.to_datetime(optas["Date"], errors="coerce")
            mask = (optas["_Date"] == match_date) & (
                (optas["Home"].str.contains(home, case=False, na=False) | optas["Away"].str.contains(home, case=False, na=False)) &
                (optas["Home"].str.contains(away, case=False, na=False) | optas["Away"].str.contains(away, case=False, na=False))
            )
            matching = optas[mask]
            if not matching.empty:
                fixture_id = matching.iloc[0]["FixtureId"]
    except Exception:
        pass

    if fixture_id:
        try:
            possession_df = data.load_fixture_effective_possession(fixture_id)
            if not possession_df.empty:
                home_poss = possession_df.iloc[0].get("Home Possession %")
                away_poss = possession_df.iloc[0].get("Away Possession %")
                if home_poss is not None and away_poss is not None:
                    add("Possession", home_poss, away_poss, "percent", show_zero=True)
        except Exception:
            pass
    elif not team_passes[home].empty or not team_passes[away].empty:
        # Fallback to pass-based possession if Opta isn't available
        total_p = len(home_passes) + len(away_passes)
        if total_p > 0:
            add("Possession (Pass %)", (len(home_passes) / total_p * 100), (len(away_passes) / total_p * 100), "percent", show_zero=True)

    add("Passes", len(home_passes), len(away_passes), show_zero=True)
    add("Accurate Passes", home_completed, away_completed)
    add(
        "Pass Accuracy",
        (home_completed / len(home_passes) * 100) if len(home_passes) else np.nan,
        (away_completed / len(away_passes) * 100) if len(away_passes) else np.nan,
        "percent",
    )
    add(
        "Progressive Passes",
        int((pd.to_numeric(home_passes.get("Territory Gain", pd.Series(dtype=float)), errors="coerce") >= 10).sum()),
        int((pd.to_numeric(away_passes.get("Territory Gain", pd.Series(dtype=float)), errors="coerce") >= 10).sum()),
    )
    add(
        "Regressive Passes",
        int((pd.to_numeric(home_passes.get("Territory Gain", pd.Series(dtype=float)), errors="coerce") <= -10).sum()),
        int((pd.to_numeric(away_passes.get("Territory Gain", pd.Series(dtype=float)), errors="coerce") <= -10).sum()),
    )
    add(
        "Final Third Entries",
        int(home_passes["_Final Third Entry"].sum()) if "_Final Third Entry" in home_passes else 0,
        int(away_passes["_Final Third Entry"].sum()) if "_Final Third Entry" in away_passes else 0,
    )
    add(
        "Crosses",
        _count_contains(team_events[home], ["Action", "Action Type"], ["CROSS"]),
        _count_contains(team_events[away], ["Action", "Action Type"], ["CROSS"]),
    )
    add(
        "Defensive Actions",
        _count_contains(team_events[home], ["Action", "Action Type"], ["TACKLE", "INTERCEPTION", "CLEARANCE", "BLOCK", "DUEL", "RECOVERY"]),
        _count_contains(team_events[away], ["Action", "Action Type"], ["TACKLE", "INTERCEPTION", "CLEARANCE", "BLOCK", "DUEL", "RECOVERY"]),
    )

    if "Set Piece" in events or "Set Piece Category" in events:
        home_set_pieces = 0
        away_set_pieces = 0
        if "Set Piece" in team_events[home]:
            home_set_pieces = int(_truthy_flag(team_events[home]["Set Piece"]).sum())
        if "Set Piece" in team_events[away]:
            away_set_pieces = int(_truthy_flag(team_events[away]["Set Piece"]).sum())
        if "Set Piece Category" in events:
            home_set_pieces = max(home_set_pieces, int(_has_label(team_events[home]["Set Piece Category"]).sum()))
            away_set_pieces = max(away_set_pieces, int(_has_label(team_events[away]["Set Piece Category"]).sum()))
        add("Set Pieces", home_set_pieces, away_set_pieces)

    return pd.DataFrame(rows)


def _stat_fill_width(value: object, other: object, value_type: str) -> float:
    number = max(_to_number(value), 0)
    other_number = max(_to_number(other), 0)
    if value_type == "percent":
        maximum = 100.0
    else:
        maximum = max(number, other_number, 1.0)
    return min(max(number / maximum * 100, 0), 100)


def _render_stat_comparison(stats: pd.DataFrame, home: str, away: str) -> None:
    if stats.empty:
        st.info("No match statistics can be calculated for this fixture.")
        return

    rows = []
    for _, row in stats.iterrows():
        value_type = str(row.get("Type", "count"))
        home_value = row.get("Home")
        away_value = row.get("Away")
        home_width = _stat_fill_width(home_value, away_value, value_type)
        away_width = _stat_fill_width(away_value, home_value, value_type)
        rows.append(
            _html_fragment(
            f"""
            <div class="mo-stat-row">
                <div class="mo-stat-value mo-stat-value-left">{ui.esc(_metric_text(home_value, value_type))}</div>
                <div class="mo-stat-track mo-stat-track-left">
                    <span class="mo-stat-fill" style="width: {home_width:.1f}%;"></span>
                </div>
                <div class="mo-stat-label">{ui.esc(str(row.get("Stat", "")))}</div>
                <div class="mo-stat-track">
                    <span class="mo-stat-fill mo-stat-fill-away" style="width: {away_width:.1f}%;"></span>
                </div>
                <div class="mo-stat-value">{ui.esc(_metric_text(away_value, value_type))}</div>
            </div>
            """
            )
        )

    _html(
        f"""
        <div class="mo-stat-panel">
            <div class="mo-stat-header">
                <div class="mo-stat-header-team mo-stat-header-team-left">{ui.esc(home)}</div>
                <div class="mo-stat-header-title">Team Stats</div>
                <div class="mo-stat-header-team mo-stat-header-team-right">{ui.esc(away)}</div>
            </div>
            {"".join(rows)}
        </div>
        """
    )


def _render_scorecard(row: pd.Series, stats: pd.DataFrame, events: pd.DataFrame) -> None:
    home = str(row.get("Home", "Home"))
    away = str(row.get("Away", "Away"))
    xg_row = stats[stats["Stat"] == "Expected Goals (xG)"] if not stats.empty and "Stat" in stats else pd.DataFrame()
    home_xg = _metric_text(xg_row["Home"].iloc[0], "xg") if not xg_row.empty else MISSING
    away_xg = _metric_text(xg_row["Away"].iloc[0], "xg") if not xg_row.empty else MISSING
    score = f"{_goal_text(row, 'home')} - {_goal_text(row, 'away')}"
    venue_note = "Verified Home/Away" if bool(row.get("Venue Verified", True)) else "Home/Away Order Not Verified"
    competition = str(row.get("Competition", "Competition"))
    season = str(row.get("Season", "Season"))

    _html(
        f"""
        <div class="mo-scorecard">
            <div class="mo-scorecard-top">
                <span>{ui.esc(competition)}</span>
                <span>{ui.esc(DOT)}</span>
                <span>{ui.esc(season)}</span>
                <span>{ui.esc(DOT)}</span>
                <span>{ui.esc(_date_text(row.get("Date")))}</span>
            </div>
            <div class="mo-score-main">
                <div class="mo-team">
                    <div class="mo-team-side">Home</div>
                    <h2 class="mo-team-name">{ui.esc(home)}</h2>
                    <div class="mo-team-substat">xG {ui.esc(home_xg)}</div>
                </div>
                <div class="mo-score-centre">
                    <div class="mo-status">{ui.esc(_match_status(row, events))}</div>
                    <div class="mo-score">{ui.esc(score)}</div>
                    <div class="mo-score-note">{ui.esc(venue_note)}</div>
                </div>
                <div class="mo-team">
                    <div class="mo-team-side">Away</div>
                    <h2 class="mo-team-name">{ui.esc(away)}</h2>
                    <div class="mo-team-substat">xG {ui.esc(away_xg)}</div>
                </div>
            </div>
        </div>
        """
    )


def _render_facts(row: pd.Series, events: pd.DataFrame, stats: pd.DataFrame) -> None:
    shots_row = stats[stats["Stat"] == "Shots"] if not stats.empty and "Stat" in stats else pd.DataFrame()
    xg_row = stats[stats["Stat"] == "Expected Goals (xG)"] if not stats.empty and "Stat" in stats else pd.DataFrame()
    total_shots = int(_to_number(shots_row["Home"].iloc[0]) + _to_number(shots_row["Away"].iloc[0])) if not shots_row.empty else 0
    total_xg = _to_number(xg_row["Home"].iloc[0]) + _to_number(xg_row["Away"].iloc[0]) if not xg_row.empty else 0.0
    facts = [
        ("Competition", row.get("Competition", MISSING)),
        ("Date", _date_text(row.get("Date"))),
        ("Season", row.get("Season", MISSING)),
        ("Match ID", row.get("MatchId", MISSING)),
        ("Event Rows", f"{len(events):,}"),
        ("Total Shots", f"{total_shots:,}" if total_shots else MISSING),
        ("Total xG", f"{total_xg:.2f}" if total_xg else MISSING),
        ("Home/Away Data", "Verified" if bool(row.get("Venue Verified", True)) else "Not Verified"),
    ]
    html = []
    for label, value in facts:
        html.append(
            _html_fragment(
            f"""
            <div class="mo-fact-card">
                <div class="mo-fact-label">{ui.esc(str(label))}</div>
                <div class="mo-fact-value">{ui.esc(str(value))}</div>
            </div>
            """
            )
        )
    _html(f'<div class="mo-facts-grid">{"".join(html)}</div>')


def _render_quick_links() -> None:
    link_cols = st.columns(6)
    with link_cols[0]:
        st.page_link("views/xg_timeline.py", label="xG Timeline")
    with link_cols[1]:
        st.page_link("views/game_control_momentum.py", label="Momentum")
    with link_cols[2]:
        st.page_link("views/shot_map.py", label="Shot Map")
    with link_cols[3]:
        st.page_link("views/pass_map.py", label="Pass Map")
    with link_cols[4]:
        st.page_link("views/passing_network.py", label="Passing Network")
    with link_cols[5]:
        st.page_link("views/event_data_table.py", label="Event Table")


def _render_data_note(row: pd.Series, events: pd.DataFrame) -> None:
    notes = []
    if not bool(row.get("Venue Verified", True)):
        notes.append(
            "This source does not verify true home/away status for the selected season; the teams are shown in the app's consistent source order."
        )
    if events.empty:
        notes.append(
            "Event-level facts such as xG, shots and pass accuracy are not available for this fixture, so only the score and match metadata can be shown."
        )
    else:
        notes.append(
            "Official possession is not available in the connected Impect tables, so Team Stats does not show possession. Event-row coverage is shown only as the Event Rows match fact."
        )
    if notes:
        _html(f'<div class="mo-note">{" ".join(ui.esc(note) for note in notes)}</div>')


def _empty_chart(title: str, message: str, height: int = 430) -> go.Figure:
    fig = charting.polish_figure(go.Figure(), title, height=height)
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color=ui.CHARLTON_MUTED, size=14),
    )
    return fig


def _compact_legend_label(value: object, limit: int = 26) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _match_visual_header(
    fig: go.Figure,
    *,
    height: int | None = None,
    margin: dict[str, int] | None = None,
) -> go.Figure:
    """Keep Match Overview chart titles and keys aligned without overlap."""
    for trace in fig.data:
        if getattr(trace, "showlegend", None) is not False and getattr(trace, "name", None):
            trace.name = _compact_legend_label(trace.name)

    title_text = fig.layout.title.text if fig.layout.title and fig.layout.title.text else None
    if title_text:
        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(size=17, color=ui.CHARLTON_BLACK),
                x=0.01,
                xanchor="left",
                y=0.98,
                yanchor="top",
            )
        )
    fig.update_layout(
        height=height,
        margin=margin or dict(l=52, r=24, t=84, b=54),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=11),
            title=dict(text=""),
        ),
    )
    return fig


def _minute_series(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="float64")
    second = pd.to_numeric(frame["Second"], errors="coerce") if "Second" in frame else pd.Series(np.nan, index=frame.index)
    minute = pd.to_numeric(frame["Minute"], errors="coerce") if "Minute" in frame else pd.Series(np.nan, index=frame.index)
    period = pd.to_numeric(frame["Period"], errors="coerce") if "Period" in frame else pd.Series(np.nan, index=frame.index)

    period_base_seconds = np.select(
        [period.eq(1), period.eq(2), period.eq(3), period.eq(4)],
        [0, 45 * 60, 90 * 60, 105 * 60],
        default=np.nan,
    )
    offset_bucket = np.floor(second / 10000).clip(lower=0)
    fallback_base_seconds = offset_bucket * 45 * 60
    base_seconds = pd.Series(period_base_seconds, index=frame.index).where(pd.notna(period_base_seconds), fallback_base_seconds)
    period_seconds = second % 10000
    elapsed_seconds = second.where(second < 10000, period_seconds + base_seconds)
    out = np.floor(elapsed_seconds / 60) + 1
    out = out.where(second.notna(), minute)
    return pd.to_numeric(out, errors="coerce").clip(lower=0, upper=130)


def _timeline_end_minute(events: pd.DataFrame, values: pd.DataFrame) -> int:
    source = events if not events.empty else values
    minutes = _minute_series(source).dropna() if not source.empty else pd.Series(dtype="float64")
    if minutes.empty and "Minute" in values:
        minutes = pd.to_numeric(values["Minute"], errors="coerce").dropna()
    if minutes.empty:
        return 95
    observed_max = float(minutes.max())
    period = pd.to_numeric(source["Period"], errors="coerce") if "Period" in source else pd.Series(dtype="float64")
    extra_time = bool(period.dropna().ge(3).any()) or observed_max > 105
    cap = 130 if extra_time else 105
    floor = 120 if extra_time else 95
    return int(max(floor, min(cap, math.ceil(observed_max / 5) * 5)))


def _event_threat(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty or "Team" not in events:
        return pd.DataFrame(columns=["Team", "Minute", "Threat"])
    values = events.copy()
    metric_cols = [col for col in ["PXT Pass", "PXT Shot", "Shot xG", "Team xT"] if col in values]
    if not metric_cols:
        return pd.DataFrame(columns=["Team", "Minute", "Threat"])
    for column in metric_cols:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    values["Threat"] = values[metric_cols].clip(lower=0).max(axis=1).fillna(0)
    values["Minute"] = _minute_series(values)
    values = values[(values["Threat"] > 0) & values["Minute"].notna()].copy()
    return values[["Team", "Minute", "Threat"]].reset_index(drop=True)


def _card_events(events: pd.DataFrame) -> pd.DataFrame:
    columns = ["Team", "Player", "Minute", "Card", "Icon"]
    if events.empty:
        return pd.DataFrame(columns=columns)

    action_type = events["Action Type"].astype(str).str.upper() if "Action Type" in events else pd.Series("", index=events.index)
    action = events["Action"].astype(str).str.upper() if "Action" in events else pd.Series("", index=events.index)
    result = events["Result"].astype(str).str.upper() if "Result" in events else pd.Series("", index=events.index)
    combined = action_type + " " + action + " " + result
    mask = combined.str.contains("YELLOW_CARD|RED_CARD|SECOND_YELLOW|BOOKING|CARD", regex=True, na=False)
    cards = events[mask].copy()
    if cards.empty:
        return pd.DataFrame(columns=columns)

    card_text = combined.loc[cards.index]
    cards["Minute"] = _minute_series(cards)
    cards["Card"] = np.select(
        [
            card_text.str.contains("RED_CARD|SECOND_YELLOW", regex=True, na=False),
            card_text.str.contains("YELLOW_CARD|BOOKING|CARD", regex=True, na=False),
        ],
        ["Red Card", "Yellow Card"],
        default="Card",
    )
    cards["Icon"] = np.where(cards["Card"].eq("Red Card"), RED_CARD_ICON, YELLOW_CARD_ICON)
    for column in columns:
        if column not in cards:
            cards[column] = np.nan
    return cards[columns].dropna(subset=["Minute"]).reset_index(drop=True)


def _momentum_chart(events: pd.DataFrame, home: str, away: str, title: str) -> go.Figure:
    values = _event_threat(events)
    values = values[values["Team"].astype(str).isin([str(home), str(away)])].copy() if not values.empty else values
    if values.empty:
        return _empty_chart(title, "No positive event-threat values available")

    values["Minute"] = values["Minute"].round().astype(int).clip(lower=0, upper=130)
    max_minute = _timeline_end_minute(events, values)
    minutes = pd.Index(range(0, max_minute + 1), name="Minute")
    summary = values.groupby(["Team", "Minute"], as_index=False)["Threat"].sum()
    home_series = (
        summary[summary["Team"].astype(str) == str(home)]
        .set_index("Minute")["Threat"]
        .reindex(minutes, fill_value=0.0)
    )
    away_series = (
        summary[summary["Team"].astype(str) == str(away)]
        .set_index("Minute")["Threat"]
        .reindex(minutes, fill_value=0.0)
    )
    net = home_series - away_series
    wave = net.rolling(window=5, center=True, min_periods=1).sum()
    wave = wave.rolling(window=3, center=True, min_periods=1).mean()
    if wave.abs().max() <= 0:
        return _empty_chart(title, "No positive event-threat values available")

    fig = go.Figure()
    x_values = minutes.to_numpy(dtype=float)
    positive = wave.clip(lower=0)
    negative = wave.clip(upper=0)
    max_axis = max(float(wave.abs().max()) * 1.35, 0.05)
    plot_axis = max_axis * 1.28
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=positive,
            mode="lines",
            name=str(home),
            line=dict(color=ui.CHARLTON_RED, width=1.2, shape="spline", smoothing=0.55),
            fill="tozeroy",
            fillcolor="rgba(195, 0, 23, 0.88)",
            hovertemplate=f"{home}<br>Minute: %{{x:.0f}}<br>Momentum: %{{y:.3f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=negative,
            mode="lines",
            name=str(away),
            line=dict(color="#7a756a", width=1.2, shape="spline", smoothing=0.55),
            fill="tozeroy",
            fillcolor="rgba(92, 88, 78, 0.68)",
            hovertemplate=f"{away}<br>Minute: %{{x:.0f}}<br>Momentum: %{{y:.3f}}<extra></extra>",
        )
    )

    shots = _shot_events(events)
    goals = shots[_is_goal(shots)].copy() if not shots.empty else pd.DataFrame()
    if not goals.empty:
        goals["Minute"] = _minute_series(goals)
        goals = goals[goals["Minute"].notna()].copy()
        for team, y_value in [(home, max_axis * 1.04), (away, -max_axis * 1.04)]:
            team_goals = goals[goals["Team"].astype(str) == str(team)]
            if team_goals.empty:
                continue
            goal_names = _lineup_label_names(team_goals["Player"]) if "Player" in team_goals else ["Goal"] * len(team_goals)
            fig.add_trace(
                go.Scatter(
                    x=team_goals["Minute"],
                    y=[y_value] * len(team_goals),
                    mode="text",
                    name=f"{team} goals",
                    text=[f"{FOOTBALL_ICON} {name}" for name in goal_names],
                    textposition="middle center",
                    textfont=dict(size=14, color=ui.CHARLTON_BLACK),
                    customdata=np.stack(
                        [
                            team_goals["Player"].fillna("Unknown") if "Player" in team_goals else pd.Series("Unknown", index=team_goals.index),
                            team_goals["Minute"].fillna(0),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=f"{team} goal<br>%{{customdata[0]}}<br>Minute: %{{customdata[1]:.0f}}<extra></extra>",
                    showlegend=False,
                )
            )

    cards = _card_events(events)
    if not cards.empty:
        for team, y_base in [(home, max_axis * 0.82), (away, -max_axis * 0.82)]:
            team_cards = cards[cards["Team"].astype(str) == str(team)]
            if team_cards.empty:
                continue
            customdata = np.stack(
                [
                    team_cards["Player"].fillna("Unknown"),
                    team_cards["Card"].fillna("Card"),
                    team_cards["Minute"].fillna(0),
                ],
                axis=-1,
            )
            card_names = _lineup_label_names(team_cards["Player"]) if "Player" in team_cards else ["Card"] * len(team_cards)
            fig.add_trace(
                go.Scatter(
                    x=team_cards["Minute"],
                    y=[y_base] * len(team_cards),
                    mode="text",
                    name=f"{team} cards",
                    text=[f"{icon} {name}" for icon, name in zip(team_cards["Icon"], card_names, strict=False)],
                    textposition="middle center",
                    textfont=dict(size=13, color=ui.CHARLTON_BLACK),
                    customdata=customdata,
                    hovertemplate=f"{team}<br>%{{customdata[1]}}: %{{customdata[0]}}<br>Minute: %{{customdata[2]:.0f}}<extra></extra>",
                    showlegend=False,
                )
            )

    fig.add_hline(y=0, line=dict(color="#8b8578", width=1.1))
    fig.add_vline(x=45, line=dict(color="#c9c1b2", width=1, dash="dot"))
    fig.add_annotation(
        x=45,
        y=max_axis * 1.02,
        text="<b>HT</b>",
        xref="x",
        yref="y",
        showarrow=False,
        font=dict(size=10, color=ui.CHARLTON_MUTED),
        bgcolor="rgba(255,255,255,0.72)",
        bordercolor="rgba(216,221,230,0.75)",
        borderpad=3,
    )
    fig.add_annotation(
        x=1,
        y=max_axis * 0.94,
        text=f"<b>{ui.esc(str(home)).upper()}</b>",
        xref="x",
        yref="y",
        xanchor="left",
        showarrow=False,
        font=dict(size=11, color=ui.CHARLTON_RED),
    )
    fig.add_annotation(
        x=1,
        y=-max_axis * 0.94,
        text=f"<b>{ui.esc(str(away)).upper()}</b>",
        xref="x",
        yref="y",
        xanchor="left",
        showarrow=False,
        font=dict(size=11, color="#5c584e"),
    )
    fig.update_layout(
        height=500,
        xaxis_title="<b>Minute</b>",
        yaxis_title="<b>Net Event Threat</b>",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
        plot_bgcolor="#ffffff",
    )
    fig.update_xaxes(range=[0, max_minute], dtick=15)
    fig.update_yaxes(
        range=[-plot_axis, plot_axis],
        tickvals=[-max_axis * 0.55, 0, max_axis * 0.55],
        ticktext=[
            f"<b>{charting.wrap_label(away, width=18, max_lines=2)}</b>",
            "0",
            f"<b>{charting.wrap_label(home, width=18, max_lines=2)}</b>",
        ],
        tickfont=dict(size=11, color=ui.CHARLTON_BLACK),
        automargin=True,
        zeroline=False,
    )
    fig = charting.polish_figure(fig, title)
    fig.update_layout(plot_bgcolor="#ffffff")
    return _match_visual_header(fig, height=500, margin=dict(l=150, r=24, t=84, b=58))


def _mode_value(values: pd.Series) -> str:
    clean = values.dropna().astype(str).str.strip()
    clean = clean[~clean.str.lower().isin(["", "nan", "none", "null"])]
    if clean.empty:
        return ""
    mode = clean.mode()
    return str(mode.iloc[0] if not mode.empty else clean.iloc[0])


def _lineup_label_names(players: pd.Series) -> list[str]:
    names = players.fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    last_names = names.apply(lambda value: value.split()[-1] if value.split() else value)
    counts = last_names.value_counts()
    labels = []
    for name, last in zip(names, last_names, strict=False):
        parts = name.split()
        if counts.get(last, 0) > 1 and len(parts) > 1:
            labels.append(f"{parts[0][0]}. {last}")
        else:
            labels.append(last)
    return labels


def _position_band(position: object, x_value: object) -> str:
    text = str(position or "").upper()
    if "GOAL" in text or text in {"GK", "G"}:
        return "GK"
    if any(token in text for token in ["DEF", "BACK", "CB", "LB", "RB", "LWB", "RWB"]):
        return "DEF"
    if any(token in text for token in ["MID", "DMF", "CMF", "AMF"]):
        return "MID"
    if any(token in text for token in ["FWD", "FORWARD", "WING", "STRIKER", "CF", "LW", "RW"]):
        return "FWD"
    x = _to_number(x_value, np.nan)
    if not math.isfinite(x):
        return "MID"
    if x <= -36:
        return "GK"
    if x <= -10:
        return "DEF"
    if x <= 20:
        return "MID"
    return "FWD"


def _formation_summary(lineup: pd.DataFrame) -> str:
    if lineup.empty or "Band" not in lineup:
        return ""
    counts = lineup["Band"].value_counts()
    defenders = int(counts.get("DEF", 0))
    midfielders = int(counts.get("MID", 0))
    forwards = int(counts.get("FWD", 0))
    if defenders + midfielders + forwards < 7:
        return ""
    return f"{defenders}-{midfielders}-{forwards}"


def _lineup_players(events: pd.DataFrame, team: str, max_players: int = 11) -> pd.DataFrame:
    columns = ["Player", "Position", "Position Display", "Actions", "First Second", "Plot X", "Plot Y", "Label", "Band"]
    if events.empty or "Team" not in events or "Player" not in events:
        return pd.DataFrame(columns=columns)
    team_events = events[events["Team"].astype(str) == str(team)].dropna(subset=["Player"]).copy()
    if team_events.empty:
        return pd.DataFrame(columns=columns)

    team_events["_Second"] = pd.to_numeric(team_events["Second"], errors="coerce") if "Second" in team_events else np.nan
    team_events["_Start X"] = pd.to_numeric(team_events["Start X"], errors="coerce") if "Start X" in team_events else np.nan
    team_events["_Start Y"] = pd.to_numeric(team_events["Start Y"], errors="coerce") if "Start Y" in team_events else np.nan
    team_events["_Position"] = team_events["Position"] if "Position" in team_events else ""

    grouped = team_events.groupby("Player", as_index=False).agg(
        Position=("_Position", _mode_value),
        Actions=("Player", "size"),
        **{
            "First Second": ("_Second", "min"),
            "Avg X": ("_Start X", "mean"),
            "Avg Y": ("_Start Y", "mean"),
        },
    )
    grouped = grouped.dropna(subset=["Avg X", "Avg Y"], how="all")
    if grouped.empty:
        return pd.DataFrame(columns=columns)

    grouped["First Second"] = pd.to_numeric(grouped["First Second"], errors="coerce").fillna(999999)
    grouped = grouped.sort_values(["First Second", "Actions"], ascending=[True, False]).head(max_players).copy()
    grouped["Plot X"] = pd.to_numeric(grouped["Avg X"], errors="coerce").clip(pitch.PITCH_X_MIN + 5, pitch.PITCH_X_MAX - 5)
    grouped["Plot Y"] = pd.to_numeric(grouped["Avg Y"], errors="coerce").clip(pitch.PITCH_Y_MIN + 5, pitch.PITCH_Y_MAX - 5)
    grouped["Position Display"] = grouped["Position"].apply(ui.clean_position)
    grouped["Band"] = [_position_band(position, x) for position, x in zip(grouped["Position"], grouped["Plot X"], strict=False)]
    grouped["_Short Label"] = _lineup_label_names(grouped["Player"])
    grouped["Label"] = grouped["_Short Label"].apply(lambda value: charting.wrap_label(value, width=11, max_lines=2))
    grouped = grouped.sort_values(["Plot X", "Plot Y"]).reset_index(drop=True)
    return grouped[columns]


def _lineup_chart(events: pd.DataFrame, team: str, color: str, mirror: bool = False) -> go.Figure:
    lineup = _lineup_players(events, team)
    formation = _formation_summary(lineup)
    title = f"{team}: Event-Derived XI Shape" + (f" ({formation})" if formation else "")
    fig = pitch.pitch_image_figure(title, height=620, legend=False)
    if lineup.empty:
        fig.add_annotation(
            text="No player-location data available",
            x=0,
            y=0,
            xref="x",
            yref="y",
            showarrow=False,
            font=dict(size=15, color=ui.CHARLTON_MUTED),
            bgcolor="rgba(255,255,255,0.80)",
            bordercolor=ui.CHARLTON_BORDER,
            borderpad=8,
        )
        return fig
    lineup = lineup.copy()
    if mirror:
        lineup["Plot X"] = -pd.to_numeric(lineup["Plot X"], errors="coerce")

    customdata = np.stack(
        [
            lineup["Player"].fillna("Unknown"),
            lineup["Position Display"].fillna("Unknown position"),
            lineup["Actions"].fillna(0),
            lineup["First Second"].fillna(0) / 60,
        ],
        axis=-1,
    )
    fig.add_trace(
        go.Scatter(
            x=lineup["Plot X"],
            y=lineup["Plot Y"],
            mode="markers",
            name=str(team),
            marker=dict(
                size=26,
                color=color,
                opacity=0.94,
                line=dict(color="#ffffff", width=2.0),
            ),
            customdata=customdata,
            hovertemplate=(
                "%{customdata[0]}"
                "<br>Position: %{customdata[1]}"
                "<br>Actions: %{customdata[2]:.0f}"
                "<br>First event: %{customdata[3]:.1f} mins<extra></extra>"
            ),
            showlegend=False,
        )
    )

    for _, row in lineup.iterrows():
        y_offset = 5.2 if _to_number(row["Plot Y"]) <= 0 else -5.2
        label_y = min(max(_to_number(row["Plot Y"]) + y_offset, pitch.PITCH_Y_MIN + 2.5), pitch.PITCH_Y_MAX - 2.5)
        fig.add_annotation(
            x=row["Plot X"],
            y=label_y,
            text=f"<b>{row['Label']}</b>",
            xref="x",
            yref="y",
            showarrow=False,
            font=dict(size=10, color=ui.CHARLTON_BLACK),
            bgcolor="rgba(255,255,255,0.84)",
            bordercolor="rgba(216,221,230,0.95)",
            borderpad=3,
        )

    fig.add_annotation(
        x=0,
        y=pitch.PITCH_Y_MIN - 3.8,
        text="\u2190 Attacking Direction" if mirror else "Attacking Direction \u2192",
        xref="x",
        yref="y",
        showarrow=False,
        font=dict(size=11, color=ui.CHARLTON_MUTED),
        bgcolor="rgba(255,255,255,0.72)",
        bordercolor="rgba(216,221,230,0.75)",
        borderpad=4,
    )
    return fig


def _render_lineup_note() -> None:
    _html(
        """
        <div class="mo-note">
            Lineup graphics use the selected match event feed: the eleven players shown are selected from earliest involvement and action volume,
            then plotted by average event location. This is an event-derived shape, not an official submitted XI or confirmed formation.
        </div>
        """
    )


def _render_shot_type_key() -> None:
    _html(
        """
        <div class="mo-shot-type-key" aria-label="Shot type shape key">
            <span class="mo-shot-type-key-title">Shape Key</span>
            <span class="mo-shot-type-key-item"><span class="mo-shot-key-shape mo-shot-key-circle"></span>Right Foot</span>
            <span class="mo-shot-type-key-item"><span class="mo-shot-key-shape mo-shot-key-diamond"></span>Left Foot</span>
            <span class="mo-shot-type-key-item"><span class="mo-shot-key-triangle"></span>Header</span>
            <span class="mo-shot-type-key-item"><span class="mo-shot-key-shape mo-shot-key-circle-open"></span>Foot</span>
            <span class="mo-shot-type-key-item"><span class="mo-shot-key-shape"></span>Other</span>
        </div>
        """
    )


ma.page_header(
    "Match Overview",
    "A match-centre view for the selected fixture: scoreline, match facts, mirrored team stats and quick links into deeper match analysis.",
    "Match metadata comes from typed CAFC_DB Impect dimensions. Stats are calculated from provider events where selected-match rows are available.",
)
_overview_css()

season = ma.select_match_season(key="match_overview_season")
matches = ma.load_matches(season)
if matches.empty:
    st.warning("No match data is available for this season.")
    st.stop()

ma.section_heading("Select Match")
match_row = ma.match_selector(matches, key="match_overview_match")

try:
    events = data.load_match_events(season=season, match_id=match_row.get("MatchId"), limit=20000)
except Exception as exc:
    st.warning(f"Could not load event-level match data for this fixture: {exc}")
    events = pd.DataFrame(columns=data.MATCH_EVENT_COLUMNS)

home_team = str(match_row.get("Home", "Home"))
away_team = str(match_row.get("Away", "Away"))
stats = _overview_stats(match_row, events)
shots = _shot_events(events)

ma.section_heading("Match Centre")
_render_scorecard(match_row, stats, events)
_render_data_note(match_row, events)

ma.section_heading("Match Facts")
_render_facts(match_row, events, stats)

ma.section_heading("Team Stats")
_render_stat_comparison(stats, home_team, away_team)

if not events.empty:
    fixture_id = None
    try:
        match_date = pd.to_datetime(match_row.get("Date"), errors="coerce")
        if pd.notna(match_date):
            optas = data.load_opta_fixtures(season=season)
            optas["_Date"] = pd.to_datetime(optas["Date"], errors="coerce")
            mask = (optas["_Date"] == match_date) & (
                (optas["Home"].str.contains(home_team, case=False, na=False) | optas["Away"].str.contains(home_team, case=False, na=False)) &
                (optas["Home"].str.contains(away_team, case=False, na=False) | optas["Away"].str.contains(away_team, case=False, na=False))
            )
            matching = optas[mask]
            if not matching.empty:
                fixture_id = matching.iloc[0]["FixtureId"]
    except Exception:
        pass

    if fixture_id:
        ma.section_heading("Official Tactical Map")
        st.caption("Official tactical positions from Opta F7 starting lineups.")
        try:
            all_lineups = data.load_opta_lineups(fixture_id)
            formations = data.load_opta_formations(fixture_id)
            
            tactical_cols = st.columns(2, gap="large")
            with tactical_cols[0]:
                home_rows = all_lineups[all_lineups["Team"].str.contains(home_team, case=False, na=False)]
                home_form = formations[formations["Side"].str.casefold() == "home"].iloc[0].get("Formation") if not formations.empty else None
                st.plotly_chart(
                    pitch.formation_map(
                        home_rows, 
                        home_team, 
                        f"{home_team} Formation", 
                        formation=home_form,
                        marker_color=ui.get_team_color(home_team)
                    ), 
                    width="stretch", 
                    key="mo_lineup_home_tactical"
                )
            with tactical_cols[1]:
                away_rows = all_lineups[all_lineups["Team"].str.contains(away_team, case=False, na=False)]
                away_form = formations[formations["Side"].str.casefold() == "away"].iloc[0].get("Formation") if not formations.empty else None
                st.plotly_chart(
                    pitch.formation_map(
                        away_rows, 
                        away_team, 
                        f"{away_team} Formation", 
                        formation=away_form, 
                        mirror=True,
                        marker_color=ui.get_team_color(away_team)
                    ), 
                    width="stretch", 
                    key="mo_lineup_away_tactical"
                )
        except Exception as exc:
            st.error(f"Could not load tactical formation: {exc}")

    ma.section_heading("Match Momentum")
    
    st.plotly_chart(
        _momentum_chart(events, home_team, away_team, "Match Momentum"),
        width="stretch",
    )
    st.caption("Momentum is a rolling signed event-threat wave: above the line favours the home team, below the line favours the away team.")

if not shots.empty:
    ma.section_heading("xG and Shot Locations")
    shot_flow_cols = st.columns([1, 1], gap="large")
    with shot_flow_cols[0]:
        xg_fig = _match_visual_header(
            pitch.xg_timeline(shots, "Cumulative xG", end_minute=_timeline_end_minute(events, shots)),
            height=560,
            margin=dict(l=54, r=24, t=84, b=56),
        )
        st.plotly_chart(xg_fig, width="stretch")
    with shot_flow_cols[1]:
        shot_fig = _match_visual_header(
            pitch.shot_map_half_pitch(shots, None, "Shot Locations"),
            height=560,
            margin=dict(l=16, r=16, t=84, b=24),
        )
        st.plotly_chart(shot_fig, width="stretch")
        _render_shot_type_key()

ma.section_heading("Detailed Match Views")
_render_quick_links()
