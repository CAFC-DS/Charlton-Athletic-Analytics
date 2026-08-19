import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import charting, data, ui


MATCH_SOURCE = (
    "Match rows use typed CAFC_DB Impect match and squad dimensions. Scores are reconstructed "
    "from the provider event feed, including own-goal attribution."
)
ACTION_SOURCE = (
    "Event views use CAFC_DB.IMPECT_RAW.EVENTS through the app's normalised event adapter, "
    "including timestamps, players, action labels, outcomes and adjusted coordinates."
)
PLAYER_PROXY_SOURCE = (
    "Player proxy views use CAFC_DB Impect player-iteration KPI facts for the selected fixture teams. "
    "These are player iteration averages, not official single-match ratings."
)

RED = ui.CHARLTON_RED
BLUE = ui.CHARLTON_DEEP_RED
DARK = ui.CHARLTON_BLACK
GREY = "#7a7f87"
LIGHT_GREY = ui.CHARLTON_BORDER
GOLD = "#c69214"

SHOT_KEYWORDS = ["SHOT", "GOAL", "PENALTY"]
PASS_KEYWORDS = ["PASS"]
FINAL_THIRD_KEYWORDS = ["FINAL_THIRD", "PASS_TO_FINAL_THIRD", "SHOT", "GOAL"]
DEFENSIVE_KEYWORDS = ["TACKLE", "INTERCEPTION", "CLEARANCE", "DUEL", "RECOVERY", "BLOCK", "PRESSURE"]
SUBSTITUTION_KEYWORDS = ["SUBSTITUTION", "SUB_ON", "SUB_OFF", "PLAYER_ON", "PLAYER_OFF"]


def _inject_match_css() -> None:
    st.markdown(
        """
        <style>
        .ma-hero {
            background:
                radial-gradient(circle at 91% 12%, rgba(255, 255, 255, 0.16), transparent 18%),
                linear-gradient(135deg, #111111 0%, #241113 48%, #9c0214 130%);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 8px;
            padding: 30px 34px;
            margin: 4px 0 20px;
            box-shadow: 0 14px 36px rgba(16, 24, 40, 0.10);
            color: #ffffff;
            overflow: hidden;
            position: relative;
        }

        .ma-hero::before {
            content: "";
            position: absolute;
            inset: 0;
            border-top: 5px solid #c30017;
            pointer-events: none;
        }

        .ma-hero-inner {
            align-items: center;
            display: flex;
            gap: 26px;
            justify-content: space-between;
            position: relative;
            z-index: 1;
        }

        .ma-hero-copy {
            min-width: 0;
        }

        .ma-eyebrow {
            color: #ffffff;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .ma-title {
            color: #ffffff;
            font-size: clamp(2rem, 3vw, 2.8rem);
            line-height: 1.05;
            margin: 0 0 12px;
            font-weight: 850;
        }

        .ma-caption {
            color: rgba(255, 255, 255, 0.80);
            max-width: 900px;
            line-height: 1.55;
            font-size: 1rem;
            margin: 0 0 14px;
        }

        .ma-source {
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 8px;
            color: rgba(255, 255, 255, 0.86);
            padding: 10px 12px;
            margin-top: 10px;
            font-size: 0.9rem;
        }

        .ma-limitation {
            color: #ffecd5;
            background: rgba(251, 146, 60, 0.14);
            border-color: rgba(251, 146, 60, 0.34);
        }

        .ma-badge {
            flex: 0 0 auto;
            width: clamp(76px, 10vw, 120px);
            height: clamp(76px, 10vw, 120px);
            object-fit: contain;
            filter: drop-shadow(0 14px 22px rgba(0, 0, 0, 0.35));
        }

        .ma-section {
            margin: 28px 0 12px;
            color: #667085;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .ma-card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 12px;
            margin: 8px 0 18px;
        }

        .ma-card {
            border: 1px solid #e6edf5;
            border-radius: 8px;
            background: #ffffff;
            padding: 16px 18px;
            min-height: 120px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            border-top: 3px solid #c30017;
        }

        .ma-card-icon {
            color: #c30017;
            font-weight: 850;
            margin-bottom: 10px;
        }

        .ma-card-title {
            color: #172033;
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .ma-card-body {
            color: #667085;
            font-size: 0.9rem;
            line-height: 1.45;
        }

        @media (max-width: 760px) {
            .ma-hero-inner {
                align-items: flex-start;
                flex-direction: column-reverse;
            }

            .ma-hero {
                padding: 26px 22px;
            }

            .ma-badge {
                width: 78px;
                height: 78px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, caption: str, basis: str = MATCH_SOURCE, limitation: str | None = None, visualisation_note: bool = True) -> None:
    ui.apply_statsearch_theme()
    _inject_match_css()
    badge = ui.badge_html("ma-badge", "Charlton Athletic crest")
    limitation_html = ""
    if limitation:
        limitation_html = f'<div class="ma-source ma-limitation"><strong>Limitation:</strong> {ui.esc(limitation)}</div>'
    html = (
        '<div class="ma-hero">'
        '<div class="ma-hero-inner">'
        '<div class="ma-hero-copy">'
        '<div class="ma-eyebrow">Charlton Match Analysis</div>'
        f'<h1 class="ma-title">{ui.esc(title)}</h1>'
        f'<p class="ma-caption">{ui.esc(caption)}</p>'
        f'<div class="ma-source"><strong>Data basis:</strong> {ui.esc(basis)}</div>'
        f"{limitation_html}"
        "</div>"
        f"{badge}"
        "</div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)
    if visualisation_note:
        ui.visualisation_fullscreen_note()


def section_heading(label: str) -> None:
    st.markdown(f'<div class="ma-section">{ui.esc(label)}</div>', unsafe_allow_html=True)


def analysis_card_grid(cards: list[dict[str, str]]) -> None:
    html = []
    for index, card in enumerate(cards, start=1):
        html.append(
            f"""
            <div class="ma-card">
                <div class="ma-card-icon">{index:02d}</div>
                <div class="ma-card-title">{ui.esc(card["title"])}</div>
                <div class="ma-card-body">{ui.esc(card["body"])}</div>
            </div>
            """
        )
    st.markdown(f'<div class="ma-card-grid">{"".join(html)}</div>', unsafe_allow_html=True)


def polish_figure(fig: go.Figure, title: str | None = None) -> go.Figure:
    return charting.polish_figure(fig, title)


def select_match_season(key: str | None = None) -> str | None:
    seasons = data.list_seasons().get("matches", [])
    if not seasons:
        st.caption("No match season selector is available from the data source.")
        return None
    return st.selectbox("Match season", seasons, index=len(seasons) - 1, key=key)


def select_player_season(key: str | None = None) -> str | None:
    seasons = data.list_seasons().get("players", [])
    if not seasons:
        st.caption("No player metric season selector is available from the data source.")
        return None
    return st.selectbox("Player metric season", seasons, index=len(seasons) - 1, key=key)


def load_matches(season: str | None = None) -> pd.DataFrame:
    matches = data.load_matches(season=season).copy().reset_index(drop=True)
    if "Date" in matches:
        matches["Date"] = pd.to_datetime(matches["Date"], errors="coerce")
        matches = matches.sort_values("Date").reset_index(drop=True)
    for col in ["Home Goals", "Away Goals"]:
        if col in matches:
            matches[col] = pd.to_numeric(matches[col], errors="coerce")
    return matches


def match_label(row: pd.Series) -> str:
    date_value = row.get("Date")
    if pd.notna(date_value):
        try:
            date_text = pd.to_datetime(date_value).strftime("%d %b %Y")
        except (TypeError, ValueError):
            date_text = str(date_value)
    else:
        date_text = "Undated"

    home_goals = row.get("Home Goals")
    away_goals = row.get("Away Goals")
    score = ""
    if pd.notna(home_goals) and pd.notna(away_goals):
        score = f" | {home_goals:.0f}-{away_goals:.0f}"
    return f"{date_text} | {row.get('Match', 'Unknown match')}{score}"


def match_selector(matches: pd.DataFrame, key: str, label: str = "Match") -> pd.Series:
    options = matches.index.tolist()
    default_pos = len(options) - 1
    if {"Home", "Away"}.issubset(matches.columns):
        charlton_rows = matches[
            matches["Home"].fillna("").astype(str).str.contains("charlton", case=False)
            | matches["Away"].fillna("").astype(str).str.contains("charlton", case=False)
        ]
        if not charlton_rows.empty:
            default_pos = options.index(charlton_rows.index[-1])
    selected_match_id = st.session_state.get("selected_match_id")
    if selected_match_id and "MatchId" in matches:
        found = matches.index[matches["MatchId"].astype(str) == str(selected_match_id)].tolist()
        if found:
            default_pos = options.index(found[0])
    selected_index = st.selectbox(
        label,
        options,
        index=default_pos,
        format_func=lambda idx: match_label(matches.loc[idx]),
        key=key,
    )
    row = matches.loc[selected_index]
    if "MatchId" in row and pd.notna(row["MatchId"]):
        st.session_state["selected_match_id"] = str(row["MatchId"])
    return row


def fixture_teams(row: pd.Series) -> list[str]:
    teams = [row.get("Home"), row.get("Away")]
    return [str(team) for team in teams if pd.notna(team) and str(team)]


def team_selector_for_match(row: pd.Series, key: str, label: str = "Team context") -> str:
    teams = fixture_teams(row)
    if not teams:
        st.caption("No teams are available for this selected fixture.")
        return ""
    charlton = [index for index, team in enumerate(teams) if "charlton" in team.casefold()]
    default = charlton[0] if charlton else 0
    return st.selectbox(label, teams, index=default, key=key)


def team_match_summary(row: pd.Series, team_name: str) -> dict[str, object]:
    is_home = str(row.get("Home")) == str(team_name)
    goals_for = row.get("Home Goals") if is_home else row.get("Away Goals")
    goals_against = row.get("Away Goals") if is_home else row.get("Home Goals")
    try:
        goal_difference = float(goals_for) - float(goals_against)
    except (TypeError, ValueError):
        goal_difference = np.nan
    if pd.isna(goal_difference):
        result = "Unknown"
        points = 0
    elif goal_difference > 0:
        result = "Win"
        points = 3
    elif goal_difference < 0:
        result = "Loss"
        points = 0
    else:
        result = "Draw"
        points = 1
    venue = "Home" if is_home else "Away"
    if not bool(row.get("Venue Verified", True)):
        venue = "Listed home" if is_home else "Listed away"
    return {
        "Team": team_name,
        "Opponent": row.get("Away") if is_home else row.get("Home"),
        "Venue": venue,
        "Goals For": goals_for,
        "Goals Against": goals_against,
        "Goal Difference": goal_difference,
        "Result": result,
        "Points": points,
    }


def team_match_rows(matches: pd.DataFrame, team_name: str) -> pd.DataFrame:
    if matches.empty or not {"Home", "Away", "Home Goals", "Away Goals"}.issubset(matches.columns):
        return pd.DataFrame()
    home = matches["Home"].astype(str) == str(team_name)
    away = matches["Away"].astype(str) == str(team_name)
    rows = matches[home | away].copy()
    if rows.empty:
        return rows

    is_home = rows["Home"].astype(str) == str(team_name)
    rows["Opponent"] = np.where(is_home, rows["Away"], rows["Home"])
    rows["Goals For"] = np.where(is_home, rows["Home Goals"], rows["Away Goals"])
    rows["Goals Against"] = np.where(is_home, rows["Away Goals"], rows["Home Goals"])
    rows["Goal Difference"] = rows["Goals For"] - rows["Goals Against"]
    rows["Team Result"] = np.select(
        [rows["Goal Difference"] > 0, rows["Goal Difference"] < 0],
        ["Win", "Loss"],
        default="Draw",
    )
    rows["Points"] = np.select(
        [rows["Team Result"] == "Win", rows["Team Result"] == "Draw"],
        [3, 1],
        default=0,
    )
    if "Date" in rows:
        rows = rows.sort_values("Date")
    rows["Cumulative Points"] = rows["Points"].cumsum()
    rows["Cumulative Goal Difference"] = rows["Goal Difference"].cumsum()
    rows["Rolling Points"] = rows["Points"].rolling(5, min_periods=1).mean().round(2)
    rows["Fixture Number"] = np.arange(1, len(rows) + 1)
    return rows


def team_record_table(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, fixture in matches.iterrows():
        for team_name in fixture_teams(fixture):
            summary = team_match_summary(fixture, team_name)
            rows.append(summary)
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    grouped = table.groupby("Team", as_index=False).agg(
        Played=("Team", "size"),
        Wins=("Result", lambda s: int((s == "Win").sum())),
        Draws=("Result", lambda s: int((s == "Draw").sum())),
        Losses=("Result", lambda s: int((s == "Loss").sum())),
        GF=("Goals For", "sum"),
        GA=("Goals Against", "sum"),
        Points=("Points", "sum"),
    )
    grouped["GD"] = grouped["GF"] - grouped["GA"]
    grouped["Goals / Match"] = (grouped["GF"] / grouped["Played"].replace(0, np.nan)).round(2)
    return grouped.sort_values(["Points", "GD", "GF"], ascending=False).reset_index(drop=True)


def scoreline_chart(row: pd.Series, title: str | None = None) -> go.Figure:
    teams = fixture_teams(row)
    goals = [row.get("Home Goals"), row.get("Away Goals")]
    colors = [RED, DARK]
    labels = [charting.wrap_label(team, width=18, max_lines=2) for team in teams]
    text = [charting.metric_text(goal, "Goals") for goal in goals]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=goals,
            marker_color=colors[: len(teams)],
            text=text,
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([teams, text], axis=-1),
            hovertemplate="%{customdata[0]}<br>Goals: %{customdata[1]}<extra></extra>",
        )
    )
    fig.update_layout(height=360, yaxis_title="Goals", xaxis_title="", showlegend=False)
    fig.update_yaxes(tickformat=".0f")
    fig = polish_figure(fig, title or "Selected match scoreline")
    fig.update_layout(margin=dict(l=34, r=44, t=68, b=66))
    return fig


def goal_trend_chart(team_matches: pd.DataFrame, title: str) -> go.Figure:
    x = team_matches["Date"] if "Date" in team_matches else team_matches["Fixture Number"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=team_matches["Goals For"],
            mode="lines+markers",
            name="Goals For",
            line=dict(color=RED, width=3),
            marker=dict(size=8),
            hovertemplate="Goals for: %{y:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=team_matches["Goals Against"],
            mode="lines+markers",
            name="Goals Against",
            line=dict(color=DARK, width=3),
            marker=dict(size=8),
            hovertemplate="Goals against: %{y:.0f}<extra></extra>",
        )
    )
    fig.update_layout(height=500, yaxis_title="Goals", xaxis_title="Match date", legend=dict(orientation="h", y=1.08))
    fig.update_yaxes(tickformat=".0f")
    return polish_figure(fig, title)


def momentum_chart(team_matches: pd.DataFrame, title: str) -> go.Figure:
    plot_df = team_matches.copy().reset_index(drop=True)
    if "Fixture Number" not in plot_df:
        plot_df["Fixture Number"] = np.arange(1, len(plot_df) + 1)
    plot_df = plot_df[pd.to_numeric(plot_df["Fixture Number"], errors="coerce").le(46)].copy()
    x = plot_df["Fixture Number"]
    goal_difference = pd.to_numeric(plot_df["Goal Difference"], errors="coerce").fillna(0)
    cumulative_goal_difference = (
        pd.to_numeric(plot_df["Cumulative Goal Difference"], errors="coerce").fillna(goal_difference.cumsum())
        if "Cumulative Goal Difference" in plot_df
        else goal_difference.cumsum()
    )
    rolling_points = pd.to_numeric(plot_df["Rolling Points"], errors="coerce")
    points = pd.to_numeric(plot_df["Points"], errors="coerce").fillna(0) if "Points" in plot_df else pd.Series(0, index=plot_df.index)
    date_text = (
        pd.to_datetime(plot_df["Date"], errors="coerce").dt.strftime("%d %b %Y").fillna("")
        if "Date" in plot_df
        else pd.Series("", index=plot_df.index)
    )
    opponents = plot_df["Opponent"].fillna("Unknown") if "Opponent" in plot_df else pd.Series("Unknown", index=plot_df.index)
    results = plot_df["Team Result"].fillna("") if "Team Result" in plot_df else pd.Series("", index=plot_df.index)
    customdata = np.stack([date_text, opponents, results, points], axis=-1)
    bar_colors = np.where(
        goal_difference > 0,
        "rgba(195, 0, 23, 0.78)",
        np.where(goal_difference < 0, "rgba(17, 17, 17, 0.42)", "rgba(102, 112, 133, 0.30)"),
    )
    y_bound = max(1.0, float(goal_difference.abs().max()) if len(goal_difference) else 1.0)
    y_bound = float(np.ceil(y_bound + 0.5))
    tick_step = 1 if len(plot_df) <= 12 else 2 if len(plot_df) <= 24 else 5
    x_axis_max = min(46.5, max(1.5, float(x.max()) + 0.5)) if len(x) else 46.5
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=goal_difference,
            name="Match Goal Difference",
            marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.78)", width=0.8)),
            opacity=0.92,
            customdata=customdata,
            hovertemplate=(
                "<b>Gameweek %{x:.0f}</b><br>Date: %{customdata[0]}"
                "<br>Opponent: %{customdata[1]}"
                "<br>Result: %{customdata[2]}"
                "<br>Goal difference: %{y:+.0f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=rolling_points,
            mode="lines+markers",
            name="5-Match Rolling Points",
            line=dict(color=DARK, width=3),
            marker=dict(size=7, color=DARK, line=dict(color="#ffffff", width=1.2)),
            yaxis="y2",
            customdata=customdata,
            hovertemplate=(
                "<b>Gameweek %{x:.0f}</b><br>Date: %{customdata[0]}"
                "<br>Opponent: %{customdata[1]}"
                "<br>Match points: %{customdata[3]:.0f}"
                "<br>5-match rolling points: %{y:.2f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=cumulative_goal_difference,
            mode="lines",
            name="Cumulative Goal Difference",
            line=dict(color=RED, width=2.4, dash="dot"),
            fill="tozeroy",
            fillcolor="rgba(195, 0, 23, 0.08)",
            hovertemplate="<b>Gameweek %{x:.0f}</b><br>Cumulative goal difference: %{y:+.0f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color="#98a2b3", width=1.2), layer="below")
    fig.update_layout(
        height=540,
        bargap=0.34,
        hovermode="x unified",
        xaxis=dict(title="Gameweek", tickmode="linear", dtick=tick_step, range=[0.5, x_axis_max]),
        yaxis=dict(title="Goal Difference", range=[min(-y_bound, float(cumulative_goal_difference.min()) - 1), max(y_bound, float(cumulative_goal_difference.max()) + 1)], zeroline=False),
        yaxis2=dict(title="Rolling Points / Match", overlaying="y", side="right", range=[-0.05, 3.05], showgrid=False, zeroline=False),
    )
    fig.update_yaxes(tickformat=".0f")
    fig = polish_figure(fig, title)
    fig.update_layout(
        margin=dict(l=52, r=54, t=104, b=58),
        title=dict(
            text=title,
            font=dict(size=20, color=DARK),
            x=0.01,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=12),
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0.01,
            title=dict(text=""),
        ),
    )
    return fig


def load_match_actions(season: str | None = None, match_row: pd.Series | None = None) -> pd.DataFrame:
    try:
        actions = data.load_match_action_counts(season=season).copy()
    except Exception as exc:
        st.warning(f"Could not load match event action counts: {exc}")
        return pd.DataFrame(columns=["MatchId", "Season", "Team", "Action", "Actions"])
    if actions.empty:
        return pd.DataFrame(columns=["MatchId", "Season", "Team", "Action", "Actions"])
    if "Actions" in actions:
        actions["Actions"] = pd.to_numeric(actions["Actions"], errors="coerce").fillna(0)

    if match_row is not None:
        match_id = match_row.get("MatchId") if "MatchId" in match_row else None
        if match_id is not None and pd.notna(match_id) and "MatchId" in actions:
            actions = actions[actions["MatchId"].astype(str) == str(match_id)]
        else:
            teams = fixture_teams(match_row)
            actions = actions[actions["Team"].astype(str).isin(teams)]
    return actions.reset_index(drop=True)


def filter_actions(actions: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    if actions.empty or "Action" not in actions:
        return pd.DataFrame(columns=actions.columns)
    pattern = "|".join(re.escape(keyword) for keyword in keywords)
    return actions[actions["Action"].astype(str).str.contains(pattern, case=False, na=False)].copy()


def action_summary(actions: pd.DataFrame, keywords: list[str] | None = None) -> pd.DataFrame:
    filtered = filter_actions(actions, keywords) if keywords else actions.copy()
    if filtered.empty:
        return pd.DataFrame(columns=["Team", "Actions"])
    return filtered.groupby("Team", as_index=False)["Actions"].sum().sort_values("Actions", ascending=False)


def action_breakdown(actions: pd.DataFrame, team_name: str | None = None, keywords: list[str] | None = None) -> pd.DataFrame:
    filtered = filter_actions(actions, keywords) if keywords else actions.copy()
    if filtered.empty:
        return pd.DataFrame(columns=["Team", "Action", "Actions"])
    if team_name:
        filtered = filtered[filtered["Team"].astype(str) == str(team_name)]
    if filtered.empty:
        return pd.DataFrame(columns=["Team", "Action", "Actions"])
    return filtered.groupby(["Team", "Action"], as_index=False)["Actions"].sum().sort_values("Actions", ascending=False)


def action_bar(summary: pd.DataFrame, selected: str | None, title: str) -> go.Figure:
    if summary.empty:
        return go.Figure()
    plot_df = summary.sort_values("Actions", ascending=True)
    colors = [RED if str(team) == str(selected) else GREY for team in plot_df["Team"]]
    plot_df["_Label"] = plot_df["Team"].apply(lambda value: charting.wrap_label(value, width=18, max_lines=2))
    plot_df["_Text"] = charting.outside_bar_text(plot_df["Actions"], "Actions")
    fig = go.Figure(
        go.Bar(
            x=plot_df["Actions"],
            y=plot_df["_Label"],
            orientation="h",
            marker_color=colors,
            text=plot_df["_Text"],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([plot_df["Team"], plot_df["_Text"]], axis=-1),
            hovertemplate="%{customdata[0]}<br>Actions: %{customdata[1]}<extra></extra>",
        )
    )
    fig.update_layout(
        height=charting.horizontal_bar_height(len(plot_df), min_height=360, row_height=54, max_height=520),
        xaxis_title="Actions",
        yaxis_title="",
        showlegend=False,
    )
    fig.update_xaxes(tickformat=".0f")
    fig = polish_figure(fig, title)
    fig.update_layout(margin=dict(l=34, r=78, t=68, b=54))
    return fig


def action_stack(actions: pd.DataFrame, keywords: list[str], title: str) -> go.Figure:
    filtered = filter_actions(actions, keywords)
    if filtered.empty:
        return go.Figure()
    plot_df = filtered.groupby(["Team", "Action"], as_index=False)["Actions"].sum()
    plot_df["Team Label"] = plot_df["Team"].apply(lambda value: charting.wrap_label(value, width=18, max_lines=2))
    plot_df["Action Label"] = plot_df["Action"].apply(lambda value: charting.wrap_label(value, width=20, max_lines=2))
    fig = px.bar(
        plot_df,
        x="Team Label",
        y="Actions",
        color="Action Label",
        color_discrete_sequence=[RED, DARK, GREY, BLUE, GOLD, "#b42318", "#475467"],
        custom_data=["Team", "Action"],
    )
    fig.update_traces(hovertemplate="%{customdata[0]}<br>%{customdata[1]}: %{y:,.0f}<extra></extra>")
    fig.update_layout(height=480, xaxis_title="", yaxis_title="Actions", legend_title_text="", bargap=0.24)
    fig.update_yaxes(tickformat=".0f")
    fig = polish_figure(fig, title)
    fig.update_layout(
        margin=dict(l=34, r=34, t=74, b=122),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
    )
    return fig


def action_donut(breakdown: pd.DataFrame, title: str) -> go.Figure:
    if breakdown.empty:
        return go.Figure()
    plot_df = breakdown.groupby("Action", as_index=False)["Actions"].sum().sort_values("Actions", ascending=False)
    plot_df["_Label"] = plot_df["Action"].apply(lambda value: charting.wrap_label(value, width=18, max_lines=2))
    fig = go.Figure(
        go.Pie(
            labels=plot_df["_Label"],
            values=plot_df["Actions"],
            hole=0.58,
            marker=dict(colors=[RED, BLUE, DARK, GREY, GOLD, "#b42318", "#475467"]),
            textinfo="percent",
            textposition="inside",
            customdata=np.stack([plot_df["Action"], plot_df["Actions"]], axis=-1),
            hovertemplate="%{customdata[0]}<br>Actions: %{customdata[1]:,.0f}<extra></extra>",
        )
    )
    fig = polish_figure(fig, title)
    fig.update_layout(
        height=430,
        showlegend=True,
        margin=dict(l=20, r=120, t=68, b=40),
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
    )
    return fig


def action_rate_table(matches: pd.DataFrame, season: str | None, keywords: list[str]) -> pd.DataFrame:
    actions = load_match_actions(season=season)
    filtered = filter_actions(actions, keywords)
    if filtered.empty:
        return pd.DataFrame()
    summary = filtered.groupby("MatchId", as_index=False)["Actions"].sum()
    cols = [col for col in ["MatchId", "Date", "Match", "Home", "Away", "Home Goals", "Away Goals"] if col in matches]
    out = matches[cols].merge(summary, on="MatchId", how="left") if "MatchId" in matches else pd.DataFrame()
    if out.empty:
        return out
    out["Actions"] = out["Actions"].fillna(0)
    return out.sort_values("Actions", ascending=False)


def player_rows_for_match(match_row: pd.Series, season: str | None = None) -> pd.DataFrame:
    try:
        players = data.load_players(season=season).copy().reset_index(drop=True)
    except Exception as exc:
        st.warning(f"Could not load player metrics: {exc}")
        return pd.DataFrame()
    teams = fixture_teams(match_row)
    if "Team" not in players or not teams:
        return pd.DataFrame()
    players = players[players["Team"].astype(str).isin(teams)].copy()
    if players.empty:
        return players
    if "Position" in players:
        players["_Position Display"] = players["Position"].apply(ui.clean_position)
    else:
        players["_Position Display"] = "Unknown position"
    for metric in data.PLAYER_METRICS:
        if metric in players:
            players[metric] = pd.to_numeric(players[metric], errors="coerce")
    if "Minutes" in players:
        players["Minutes"] = pd.to_numeric(players["Minutes"], errors="coerce")
    return players


def _percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return (values.rank(pct=True, ascending=higher_is_better) * 100).round(1)


def player_rating_table(players: pd.DataFrame) -> pd.DataFrame:
    if players.empty:
        return pd.DataFrame()
    out = players.copy()
    weights = {
        "Goals /90": 0.22,
        "Assists /90": 0.18,
        "Bypassed Opponents /90": 0.24,
        "Pass %": 0.14,
        "Passes to Final 3rd /90": 0.22,
    }
    components = []
    applied_weight = 0.0
    for metric, weight in weights.items():
        if metric in out:
            col = f"{metric} Percentile"
            out[col] = _percentile(out[metric])
            components.append(out[col] * weight)
            applied_weight += weight
    if components and applied_weight:
        out["Rating Proxy"] = (sum(components) / applied_weight).round(1)
    else:
        out["Rating Proxy"] = np.nan
    return out.sort_values("Rating Proxy", ascending=False)


def player_scatter(players: pd.DataFrame, x: str, y: str, size: str | None, title: str) -> go.Figure:
    if players.empty:
        return go.Figure()
    players = players.copy()
    players["_Text"] = ""
    if "Rating Proxy" in players:
        top_indexes = players.sort_values("Rating Proxy", ascending=False).head(2).index
        players.loc[top_indexes, "_Text"] = players.loc[top_indexes, "Player"].apply(
            lambda value: charting.wrap_label(value, width=16, max_lines=2)
        )
    custom_cols = [players["Player"], players[x], players[y]]
    if size in players.columns:
        custom_cols.append(players[size])
    customdata = np.stack(custom_cols, axis=-1)
    fig = px.scatter(
        players,
        x=x,
        y=y,
        size=size if size in players.columns else None,
        color="Team" if "Team" in players else None,
        text="_Text",
        hover_data=[col for col in ["Team", "_Position Display", "Minutes"] if col in players],
        color_discrete_sequence=[RED, DARK, GREY, BLUE],
    )
    hover = f"%{{customdata[0]}}<br>{x}: {charting.hover_value('customdata[1]', x)}<br>{y}: {charting.hover_value('customdata[2]', y)}"
    if size in players.columns:
        hover += f"<br>{size}: {charting.hover_value('customdata[3]', size)}"
    fig.update_traces(
        customdata=customdata,
        hovertemplate=hover + "<extra></extra>",
        textposition="top center",
        marker=dict(line=dict(width=1.2, color="#ffffff"), opacity=0.9),
    )
    fig.update_layout(height=560, showlegend=True)
    charting.format_xaxis(fig, x)
    charting.format_yaxis(fig, y)
    return polish_figure(fig, title)


def player_rating_bar(players: pd.DataFrame, title: str, top_n: int = 16) -> go.Figure:
    if players.empty or "Rating Proxy" not in players:
        return go.Figure()
    plot_df = players.sort_values("Rating Proxy", ascending=True).tail(top_n)
    colors = [RED if "charlton" in str(team).casefold() else GREY for team in plot_df.get("Team", [])]
    plot_df["_Label"] = plot_df["Player"].apply(lambda value: charting.wrap_label(value, width=18, max_lines=2))
    plot_df["_Text"] = charting.outside_bar_text(plot_df["Rating Proxy"], "Rating Proxy")
    fig = go.Figure(
        go.Bar(
            x=plot_df["Rating Proxy"],
            y=plot_df["_Label"],
            orientation="h",
            marker_color=colors if colors else RED,
            text=plot_df["_Text"],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([plot_df["Player"], plot_df["_Text"]], axis=-1),
            hovertemplate="%{customdata[0]}<br>Rating proxy: %{customdata[1]}<extra></extra>",
        )
    )
    fig.update_layout(
        height=charting.horizontal_bar_height(len(plot_df), row_height=32),
        xaxis_title="Rating proxy",
        yaxis_title="",
        showlegend=False,
    )
    fig.update_xaxes(tickformat=".1f")
    fig = polish_figure(fig, title)
    fig.update_layout(margin=dict(l=36, r=78, t=68, b=54))
    return fig


def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def metric_value(value: object, metric: str | None = None) -> str:
    return charting.metric_text(value, metric)
