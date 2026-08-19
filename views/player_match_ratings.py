# =============================================================================
# PLAYER MATCH RATINGS - selected-fixture position-aware rating proxy
# =============================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charting, data, match_analysis as ma, ui


COMPONENT_METRICS: dict[str, list[tuple[str, bool]]] = {
    "Shooting / End Product": [
        ("Goals /90", False),
        ("xG /90", False),
        ("Post-Shot xG /90", False),
        ("Shots /90", False),
    ],
    "Creation & Progression": [
        ("Assists /90", False),
        ("Packing xG /90", False),
        ("Passes to Final 3rd /90", False),
        ("Bypassed Opponents /90", False),
        ("Bypassed Defenders /90", False),
        ("Receiving Progression /90", False),
        ("Dribble Progression /90", False),
        ("Pass Progression /90", False),
        ("Cross Progression /90", False),
    ],
    "Passing & Ball Security": [
        ("Pass %", False),
        ("Successful Passes /90", False),
        ("Ball Security %", False),
        ("Ball Losses /90", True),
        ("Critical Ball Losses /90", True),
        ("Losses Per 100 Actions", True),
    ],
    "Defending & Duels": [
        ("Ball Wins /90", False),
        ("Ball Win Value /90", False),
        ("Ground Duel Win %", False),
        ("Aerial Duel Win %", False),
    ],
    "Goalkeeping": [
        ("Goals Prevented /90", False),
        ("Save Actions /90", False),
        ("Goals Conceded /90", True),
    ],
}


POSITION_WEIGHTS: dict[str, dict[str, float]] = {
    "Goalkeepers": {
        "Goalkeeping": 60,
        "Passing & Ball Security": 20,
        "Defending & Duels": 10,
        "Creation & Progression": 10,
        "Shooting / End Product": 0,
    },
    "Defenders": {
        "Defending & Duels": 40,
        "Creation & Progression": 27,
        "Passing & Ball Security": 25,
        "Shooting / End Product": 8,
        "Goalkeeping": 0,
    },
    "Midfielders": {
        "Creation & Progression": 35,
        "Passing & Ball Security": 30,
        "Defending & Duels": 20,
        "Shooting / End Product": 15,
        "Goalkeeping": 0,
    },
    "Attackers": {
        "Shooting / End Product": 38,
        "Creation & Progression": 32,
        "Passing & Ball Security": 15,
        "Defending & Duels": 15,
        "Goalkeeping": 0,
    },
    "Overall": {
        "Creation & Progression": 30,
        "Passing & Ball Security": 25,
        "Defending & Duels": 20,
        "Shooting / End Product": 20,
        "Goalkeeping": 5,
    },
}


POSITION_ORDER = ["Goalkeepers", "Defenders", "Midfielders", "Attackers", "Unknown"]
POSITION_COLORS = {
    "Goalkeepers": "#c69214",
    "Defenders": ui.CHARLTON_BLACK,
    "Midfielders": ui.CHARLTON_RED,
    "Attackers": "#12b76a",
    "Unknown": "#7a7f87",
}
AVERAGE_LINE_COLOR = "#2563eb"


def _rating_css() -> None:
    st.markdown(
        """
        <style>
        .pmr-note {
            background: #f8fafc;
            border: 1px solid var(--ss-border);
            border-left: 4px solid var(--ss-accent);
            border-radius: 10px;
            color: var(--ss-muted);
            font-size: 0.9rem;
            line-height: 1.5;
            margin: 8px 0 16px;
            padding: 13px 15px;
        }

        .pmr-note strong {
            color: var(--ss-ink);
        }

        .pmr-card-grid {
            display: grid;
            gap: 12px;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            margin: 8px 0 18px;
        }

        .pmr-card {
            background: #ffffff;
            border: 1px solid #e6edf5;
            border-radius: 10px;
            border-top: 3px solid var(--ss-accent);
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-height: 132px;
            padding: 15px 16px;
        }

        .pmr-card-index {
            color: var(--ss-accent);
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            margin-bottom: 12px;
        }

        .pmr-card-title {
            color: var(--ss-ink);
            font-size: 1rem;
            font-weight: 850;
            line-height: 1.2;
            margin-bottom: 8px;
        }

        .pmr-card-body {
            color: var(--ss-muted);
            font-size: 0.89rem;
            line-height: 1.45;
        }

        .pmr-summary-grid {
            display: grid;
            gap: 12px;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            margin: 8px 0 18px;
        }

        .pmr-summary-card {
            background: #ffffff;
            border: 1px solid var(--ss-border);
            border-radius: 10px;
            border-top: 3px solid var(--ss-accent);
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-height: 96px;
            padding: 13px 15px;
        }

        .pmr-summary-label {
            color: var(--ss-muted);
            font-size: 0.74rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            line-height: 1.25;
            margin-bottom: 9px;
            text-transform: uppercase;
        }

        .pmr-summary-value {
            color: var(--ss-ink);
            font-size: clamp(1.35rem, 1.8vw, 1.7rem);
            font-weight: 600;
            letter-spacing: -0.04em;
            line-height: 1.08;
            overflow-wrap: anywhere;
        }

        .pmr-summary-value-text {
            font-size: clamp(0.86rem, 1vw, 1.02rem);
            font-weight: 850;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }

        .pmr-colour-key {
            align-items: center;
            background: #ffffff;
            border: 1px solid var(--ss-border);
            border-radius: 10px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            display: flex;
            flex-wrap: wrap;
            gap: 10px 16px;
            margin: 4px 0 12px;
            padding: 11px 14px;
        }

        .pmr-colour-key-title {
            color: var(--ss-ink);
            font-size: 0.82rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .pmr-key-item {
            align-items: center;
            display: inline-flex;
            font-size: 0.88rem;
            font-weight: 700;
            gap: 7px;
        }

        .pmr-key-swatch {
            border: 1px solid rgba(17, 17, 17, 0.16);
            border-radius: 999px;
            display: inline-block;
            height: 12px;
            width: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _position_group(position: object) -> str:
    text = "" if position is None else str(position)
    if not text or text.lower() == "nan":
        return "Unknown"

    primary = text.split(",")[0].strip().upper()
    if "GOALKEEPER" in primary:
        return "Goalkeepers"
    if "DEFENDER" in primary or "WINGBACK" in primary or "FULLBACK" in primary:
        return "Defenders"
    if "WINGER" in primary or "FORWARD" in primary or "STRIKER" in primary:
        return "Attackers"
    if "MIDFIELD" in primary:
        return "Midfielders"
    return "Unknown"


def _score_metric(frame: pd.DataFrame, metric: str, lower_is_better: bool) -> pd.Series | None:
    if metric not in frame:
        return None
    values = pd.to_numeric(frame[metric], errors="coerce")
    if not values.notna().any():
        return None
    ranked_values = -values if lower_is_better else values
    return (ranked_values.rank(pct=True) * 100).round(1)


def _component_score(frame: pd.DataFrame, metric_specs: list[tuple[str, bool]]) -> pd.Series:
    scores = []
    for metric, lower_is_better in metric_specs:
        metric_score = _score_metric(frame, metric, lower_is_better)
        if metric_score is not None:
            scores.append(metric_score)
    if not scores:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.concat(scores, axis=1).mean(axis=1).round(1)


def _weighted_rating(row: pd.Series) -> float:
    weights = POSITION_WEIGHTS.get(str(row.get("Position Group")), POSITION_WEIGHTS["Overall"])
    weighted_total = 0.0
    applied_weight = 0.0

    for component, weight in weights.items():
        if weight <= 0:
            continue
        value = row.get(component)
        if pd.notna(value):
            weighted_total += float(value) * float(weight)
            applied_weight += float(weight)

    if applied_weight <= 0:
        return np.nan
    return round(weighted_total / applied_weight, 1)


def _rating_table(players: pd.DataFrame) -> pd.DataFrame:
    if players.empty:
        return pd.DataFrame()

    out = players.copy()
    out["Position Group"] = out.get("Position", pd.Series("Unknown", index=out.index)).apply(_position_group)
    for component, metric_specs in COMPONENT_METRICS.items():
        out[component] = _component_score(out, metric_specs)
    for position_group, weights in POSITION_WEIGHTS.items():
        if position_group == "Overall":
            continue
        mask = out["Position Group"].astype(str) == position_group
        for component, weight in weights.items():
            if weight <= 0 and component in out:
                out.loc[mask, component] = np.nan
    out["Rating Proxy"] = out.apply(_weighted_rating, axis=1)
    return out.sort_values("Rating Proxy", ascending=False).reset_index(drop=True)


def _card_grid(cards: list[dict[str, str]]) -> None:
    html = "".join(
        (
            '<div class="pmr-card">'
            f'<div class="pmr-card-index">{index:02d}</div>'
            f'<div class="pmr-card-title">{ui.esc(card["title"])}</div>'
            f'<div class="pmr-card-body">{ui.esc(card["body"])}</div>'
            "</div>"
        )
        for index, card in enumerate(cards, start=1)
    )
    st.markdown(f'<div class="pmr-card-grid">{html}</div>', unsafe_allow_html=True)


def _summary_card(label: str, value: object, *, text_value: bool = False) -> str:
    value_class = "pmr-summary-value pmr-summary-value-text" if text_value else "pmr-summary-value"
    return (
        '<div class="pmr-summary-card">'
        f'<div class="pmr-summary-label">{ui.esc(label)}</div>'
        f'<div class="{value_class}">{ui.esc(value)}</div>'
        "</div>"
    )


def _render_rating_summary(leader: pd.Series, ranked: pd.DataFrame) -> None:
    html = "".join(
        [
            _summary_card("Top Player", str(leader.get("Player", "Unknown")), text_value=True),
            _summary_card("Rating Proxy", ma.metric_value(leader.get("Rating Proxy"), "Rating Proxy")),
            _summary_card("Team", str(leader.get("Team", "Unknown")), text_value=True),
            _summary_card("Position Group", str(leader.get("Position Group", "Unknown")), text_value=True),
            _summary_card("Ranked Players", f"{len(ranked):,}"),
        ]
    )
    st.markdown(f'<div class="pmr-summary-grid">{html}</div>', unsafe_allow_html=True)


def _methodology_cards() -> None:
    _card_grid(
        [
            {
                "title": "Event Impact",
                "body": "Public rating systems usually start from player actions and assign positive or negative value according to the action, outcome and context.",
            },
            {
                "title": "Component Scores",
                "body": "This proxy groups available Charlton app metrics into shooting, creation, passing security, defending and goalkeeping components.",
            },
            {
                "title": "Position Weighting",
                "body": "The component weights change by position group, so keepers are judged mainly on goalkeeping while attackers receive more weight for end product.",
            },
            {
                "title": "Displayed Pool Percentiles",
                "body": "Each metric is converted to a percentile within the selected fixture-team player pool, then combined into a 0-100 proxy score.",
            },
        ]
    )


def _component_cards() -> None:
    cards = []
    for component, metric_specs in COMPONENT_METRICS.items():
        metrics = []
        for metric, lower_is_better in metric_specs:
            suffix = " lower is better" if lower_is_better else ""
            metrics.append(f"{metric}{suffix}")
        cards.append({"title": component, "body": ", ".join(metrics)})
    _card_grid(cards)


def _scope_title(scope: str, team: str | None, position_group: str | None) -> str:
    if scope == "By Team + Position Group":
        return f"{team}: {position_group} Rating Ranking"
    if scope == "By Team":
        return f"{team}: Player Rating Ranking"
    if scope == "By Position Group":
        return f"{position_group}: Player Rating Ranking"
    return "Overall Player Rating Ranking"


def _is_charlton_team(team: object) -> bool:
    return "charlton" in str(team).lower()


def _unique_labels(values: pd.Series | list[object]) -> list[str]:
    labels = [str(value) for value in values if pd.notna(value) and str(value)]
    return sorted(set(labels))


def _team_color_map(team_labels: list[str]) -> dict[str, str]:
    teams = _unique_labels(team_labels)
    charlton_teams = [team for team in teams if _is_charlton_team(team)]
    other_teams = [team for team in teams if team not in charlton_teams]

    color_map: dict[str, str] = {}
    if charlton_teams:
        for team in charlton_teams:
            color_map[team] = ui.CHARLTON_RED
        for index, team in enumerate(other_teams):
            color_map[team] = ui.CHARLTON_BLACK
        return color_map

    for index, team in enumerate(teams):
        color_map[team] = ui.CHARLTON_RED if index == 0 else ui.CHARLTON_BLACK
    return color_map


def _color_values(frame: pd.DataFrame, color_by: str) -> tuple[list[str], list[str]]:
    if color_by == "Position Group":
        labels = frame["Position Group"].fillna("Unknown").astype(str).tolist()
        return [POSITION_COLORS.get(label, "#7a7f87") for label in labels], labels

    labels = frame["Team"].fillna("Unknown").astype(str).tolist() if "Team" in frame else ["Unknown"] * len(frame)
    color_map = _team_color_map(labels)
    return [color_map[label] for label in labels], labels


def _colour_key_items(frame: pd.DataFrame, color_by: str) -> list[tuple[str, str]]:
    if frame.empty:
        return []
    if color_by == "Position Group":
        groups = [group for group in POSITION_ORDER if group in set(frame["Position Group"].dropna().astype(str))]
        return [(group, POSITION_COLORS.get(group, "#7a7f87")) for group in groups]

    labels = frame["Team"].fillna("Unknown").astype(str).tolist() if "Team" in frame else ["Unknown"]
    color_map = _team_color_map(labels)
    return [(team, color_map[team]) for team in _unique_labels(labels)]


def _render_colour_key(frame: pd.DataFrame, color_by: str) -> None:
    items = _colour_key_items(frame, color_by)
    if not items:
        return
    title = "Colour Key: Position Groups" if color_by == "Position Group" else "Colour Key: Teams"
    item_html = "".join(
        (
            f'<span class="pmr-key-item" style="color:{ui.esc(color)}">'
            f'<span class="pmr-key-swatch" style="background:{ui.esc(color)}"></span>'
            f"{ui.esc(label)}"
            "</span>"
        )
        for label, color in items
    )
    st.markdown(
        f'<div class="pmr-colour-key"><span class="pmr-colour-key-title">{ui.esc(title)}</span>{item_html}</div>',
        unsafe_allow_html=True,
    )


def _rating_bar(players: pd.DataFrame, title: str, top_n: int, color_by: str) -> go.Figure:
    if players.empty or "Rating Proxy" not in players:
        return go.Figure()

    plot_df = players.sort_values("Rating Proxy", ascending=True).tail(top_n).copy()
    plot_df["_Label"] = plot_df["Player"].apply(lambda value: charting.wrap_label(value, width=19, max_lines=2))
    plot_df["_Text"] = charting.outside_bar_text(plot_df["Rating Proxy"], "Rating Proxy")
    colors, color_labels = _color_values(plot_df, color_by)
    average_rating = pd.to_numeric(players["Rating Proxy"], errors="coerce").mean()

    customdata = np.stack(
        [
            plot_df["Player"],
            plot_df["Team"] if "Team" in plot_df else pd.Series("Unknown", index=plot_df.index),
            plot_df["Position Group"],
            plot_df["_Position Display"] if "_Position Display" in plot_df else pd.Series("Unknown", index=plot_df.index),
            plot_df["_Text"],
            plot_df["Shooting / End Product"],
            plot_df["Creation & Progression"],
            plot_df["Passing & Ball Security"],
            plot_df["Defending & Duels"],
            plot_df["Goalkeeping"],
            color_labels,
        ],
        axis=-1,
    )

    fig = go.Figure(
        go.Bar(
            x=plot_df["Rating Proxy"],
            y=plot_df["_Label"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="#ffffff", width=0.8)),
            text=plot_df["_Text"],
            textposition="outside",
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b>"
                "<br>Team: %{customdata[1]}"
                "<br>Position Group: %{customdata[2]}"
                "<br>Position: %{customdata[3]}"
                "<br>Rating Proxy: %{customdata[4]}"
                "<br>Shooting / End Product: %{customdata[5]:.1f}"
                "<br>Creation & Progression: %{customdata[6]:.1f}"
                "<br>Passing & Ball Security: %{customdata[7]:.1f}"
                "<br>Defending & Duels: %{customdata[8]:.1f}"
                "<br>Goalkeeping: %{customdata[9]:.1f}<extra></extra>"
            ),
        )
    )
    if pd.notna(average_rating):
        fig.add_vline(
            x=float(average_rating),
            line=dict(color=AVERAGE_LINE_COLOR, width=2.5, dash="dash"),
            annotation_text=f"Average: {average_rating:.1f}",
            annotation_position="top",
        )

    fig.update_layout(
        height=charting.horizontal_bar_height(len(plot_df), min_height=460, row_height=36, max_height=760),
        xaxis_title="Rating Proxy",
        yaxis_title="",
        showlegend=False,
    )
    fig.update_xaxes(range=[0, 105], tickformat=".1f")
    fig = charting.polish_figure(fig, title)
    fig.update_layout(margin=dict(l=36, r=94, t=82, b=54))
    return fig


def _ranking_table(frame: pd.DataFrame) -> pd.DataFrame:
    display_cols = ma.available_columns(
        frame,
        [
            "Player",
            "Team",
            "Position Group",
            "_Position Display",
            "Minutes",
            "Rating Proxy",
            "Shooting / End Product",
            "Creation & Progression",
            "Passing & Ball Security",
            "Defending & Duels",
            "Goalkeeping",
        ],
    )
    return frame.sort_values("Rating Proxy", ascending=False)[display_cols]


def _select_player_metric_season(match_season: str | None) -> str | None:
    seasons = data.list_seasons().get("players", [])
    if not seasons:
        st.caption("No player metric season selector is available from the data source.")
        return None
    default_index = seasons.index(match_season) if match_season in seasons else len(seasons) - 1
    return st.selectbox("Player Metric Season", seasons, index=default_index, key="ratings_player_season")


def _select_match_season_for_ratings() -> str | None:
    season_map = data.list_seasons()
    seasons = season_map.get("matches", [])
    if not seasons:
        st.caption("No match season selector is available from the data source.")
        return None
    player_seasons = set(season_map.get("players", []))
    # Search from the most recent season backward: some seasons are listed by
    # the data source without having any actual match rows yet (or any more),
    # and matching the oldest such season by scanning forward left this page
    # defaulting to a dead season with a warning on first load.
    default_index = next(
        (index for index in range(len(seasons) - 1, -1, -1) if seasons[index] in player_seasons),
        len(seasons) - 1,
    )
    return st.selectbox("Match Season", seasons, index=default_index, key="ratings_match_season")


def _player_metric_teams(player_season: str | None) -> set[str]:
    try:
        players = data.load_players(season=player_season)
    except Exception:
        return set()
    if players.empty or "Team" not in players:
        return set()
    return set(players["Team"].dropna().astype(str))


def _matches_with_player_metric_teams(matches: pd.DataFrame, metric_teams: set[str]) -> pd.DataFrame:
    if matches.empty or not metric_teams or not {"Home", "Away"}.issubset(matches.columns):
        return matches
    home_available = matches["Home"].astype(str).isin(metric_teams)
    away_available = matches["Away"].astype(str).isin(metric_teams)
    filtered = matches[home_available | away_available].copy()
    return filtered if not filtered.empty else matches


ma.page_header(
    "Player Match Ratings",
    "Rank players from the selected fixture teams with a transparent, position-aware rating proxy.",
    ma.PLAYER_PROXY_SOURCE,
    "The audited Impect tables do not expose an official single-match player rating. This page uses player iteration averages for the teams in the selected fixture, so it should be read as a transparent comparison proxy rather than an official match score.",
)
_rating_css()

with st.expander("What the Rating Consists Of", expanded=True):
    st.markdown(
        """
        <div class="pmr-note">
            <strong>Research basis:</strong> public football rating systems commonly combine many event and statistic inputs, then apply context and position-sensitive weighting. Exact provider formulas are proprietary, so this app uses an inspectable proxy built from the metrics currently available in the Charlton data model.
        </div>
        """,
        unsafe_allow_html=True,
    )
    _methodology_cards()
    ma.section_heading("Current Proxy Components")
    _component_cards()

match_season = _select_match_season_for_ratings()
matches = ma.load_matches(match_season)
if matches.empty:
    st.warning("No match data is available for this season.")
    st.stop()

player_season = _select_player_metric_season(match_season)
metric_teams = _player_metric_teams(player_season)
rating_matches = _matches_with_player_metric_teams(matches, metric_teams)
if len(rating_matches) < len(matches):
    st.caption("Match options are filtered to fixtures containing at least one team with player metric rows for the selected player season.")

match_row = ma.match_selector(rating_matches, key="ratings_match")
players = ma.player_rows_for_match(match_row, player_season)
ratings = _rating_table(players)
if ratings.empty:
    st.info("Player aggregate rows are not available for the teams in this selected fixture.")
    st.stop()

team_options = sorted(ratings["Team"].dropna().astype(str).unique().tolist()) if "Team" in ratings else []
position_options = [group for group in POSITION_ORDER if group in set(ratings["Position Group"].dropna().astype(str))]

ma.section_heading("Ranking Controls")
control_cols = st.columns([1.2, 1.1, 1.1, 0.8, 0.8])
ranking_scope = control_cols[0].selectbox(
    "Ranking Scope",
    ["Overall", "By Team", "By Position Group", "By Team + Position Group"],
)

selected_team = team_options[0] if team_options else None
selected_position = position_options[0] if position_options else None
if ranking_scope in {"By Team", "By Team + Position Group"} and team_options:
    selected_team = control_cols[1].selectbox("Team", team_options)
else:
    control_cols[1].caption("Team filter applies to team-ranking scopes.")

if ranking_scope in {"By Position Group", "By Team + Position Group"} and position_options:
    selected_position = control_cols[2].selectbox("Position Group", position_options)
else:
    control_cols[2].caption("Position filter applies to position-ranking scopes.")

max_minutes = int(np.nanmax(pd.to_numeric(ratings.get("Minutes", pd.Series([0])), errors="coerce").fillna(0)))
default_minutes = min(180, max_minutes)
minimum_minutes = control_cols[3].slider("Minimum Minutes", 0, max(max_minutes, 1), default_minutes, step=30)

max_rows = max(len(ratings), 1)
if max_rows <= 5:
    top_n = max_rows
    control_cols[4].metric("Players Shown", top_n)
else:
    top_n = control_cols[4].slider("Players Shown", 5, min(30, max_rows), min(16, max_rows))

ranked = ratings.copy()
if "Minutes" in ranked:
    ranked = ranked[pd.to_numeric(ranked["Minutes"], errors="coerce").fillna(0).ge(minimum_minutes)].copy()
if ranking_scope in {"By Team", "By Team + Position Group"} and selected_team:
    ranked = ranked[ranked["Team"].astype(str) == str(selected_team)].copy()
if ranking_scope in {"By Position Group", "By Team + Position Group"} and selected_position:
    ranked = ranked[ranked["Position Group"].astype(str) == str(selected_position)].copy()

if ranked.empty:
    st.info("No players match the current ranking filters.")
    st.stop()

leader = ranked.sort_values("Rating Proxy", ascending=False).iloc[0]
ma.section_heading("Rating Proxy Summary")
_render_rating_summary(leader, ranked)

ma.section_heading("Player Rating Ranking")
color_by = "Team"
visible_ranked = ranked.sort_values("Rating Proxy", ascending=True).tail(top_n).copy()
_render_colour_key(visible_ranked, color_by)
st.plotly_chart(
    _rating_bar(ranked, _scope_title(ranking_scope, selected_team, selected_position), top_n, color_by),
    width="stretch",
)

ma.section_heading("Rating Breakdown")
st.dataframe(
    _ranking_table(ranked),
    width="stretch",
    hide_index=True,
    column_config={
        "Minutes": st.column_config.NumberColumn("Minutes", format="%.0f"),
        "Rating Proxy": st.column_config.ProgressColumn("Rating Proxy", format="%.1f", min_value=0, max_value=100),
        "Shooting / End Product": st.column_config.ProgressColumn("Shooting / End Product", format="%.1f", min_value=0, max_value=100),
        "Creation & Progression": st.column_config.ProgressColumn("Creation & Progression", format="%.1f", min_value=0, max_value=100),
        "Passing & Ball Security": st.column_config.ProgressColumn("Passing & Ball Security", format="%.1f", min_value=0, max_value=100),
        "Defending & Duels": st.column_config.ProgressColumn("Defending & Duels", format="%.1f", min_value=0, max_value=100),
        "Goalkeeping": st.column_config.ProgressColumn("Goalkeeping", format="%.1f", min_value=0, max_value=100),
    },
)
