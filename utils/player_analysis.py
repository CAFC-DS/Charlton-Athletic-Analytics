import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import charting, data, ui


PLAYER_SOURCE = (
    "Player metrics come from CAFC_DB Impect player-iteration KPI facts: identity fields, minutes, "
    "goals, assists, xG, passing, progression, duel, ball-win and goalkeeper fields where available."
)

RED = ui.CHARLTON_RED
BLUE = ui.CHARLTON_DEEP_RED
DARK = ui.CHARLTON_BLACK
GREY = "#7a7f87"
LIGHT_GREY = ui.CHARLTON_BORDER
GREEN = "#16a34a"
AMBER = "#f59e0b"
SOFT_RED = "#fee4e8"
PERFORMANCE_GREEN = "#15803d"
PERFORMANCE_RED = "#dc2626"

PROFILE_METRIC_META = {
    "Goals /90": ("Attacking", True, "Goals"),
    "Assists /90": ("Attacking", True, "Assists"),
    "xG /90": ("Attacking", True, "xG"),
    "Post-Shot xG /90": ("Attacking", True, "Post-shot xG"),
    "Shots /90": ("Attacking", True, "Shots"),
    "Pass %": ("Passing", True, "Pass %"),
    "Successful Passes /90": ("Passing", True, "Pass volume"),
    "Passes to Final 3rd /90": ("Passing", True, "Final-third passes"),
    "Pass Progression /90": ("Progression", True, "Pass progression"),
    "Cross Progression /90": ("Progression", True, "Cross progression"),
    "Bypassed Opponents /90": ("Progression", True, "Opponents bypassed"),
    "Bypassed Defenders /90": ("Progression", True, "Defenders bypassed"),
    "Receiving Progression /90": ("Progression", True, "Receiving progression"),
    "Dribble Progression /90": ("Progression", True, "Dribble progression"),
    "Ball Wins /90": ("Defending", True, "Ball wins"),
    "Ball Win Value /90": ("Defending", True, "Ball-win value"),
    "Ground Duel Win %": ("Defending", True, "Ground duels"),
    "Aerial Duel Win %": ("Defending", True, "Aerial duels"),
    "Ball Security %": ("Possession", True, "Ball security"),
    "Ball Losses /90": ("Possession", False, "Low ball losses"),
    "Losses Per 100 Actions": ("Possession", False, "Loss rate"),
    "Critical Ball Losses /90": ("Possession", False, "Low critical losses"),
    "Ball Loss Threat /90": ("Possession", False, "Low loss threat"),
    "Team-Mates Bypassed By Losses /90": ("Possession", False, "Low teammate exposure"),
    "Neutral Passes /90": ("Possession", True, "Circulation passes"),
    "Goals Prevented /90": ("Goalkeeping", True, "Goals prevented"),
    "Save Actions /90": ("Goalkeeping", True, "Save actions"),
    "Post-Shot xG Faced /90": ("Goalkeeping", True, "PSxG faced"),
    "Goals Conceded /90": ("Goalkeeping", False, "Low goals conceded"),
}

ROLE_RADAR_METRICS = {
    "Goalkeeper": [
        "Goals Prevented /90",
        "Save Actions /90",
        "Goals Conceded /90",
        "Pass %",
        "Successful Passes /90",
        "Pass Progression /90",
        "Passes to Final 3rd /90",
    ],
    "Centre Back": [
        "Ball Wins /90",
        "Ball Win Value /90",
        "Aerial Duel Win %",
        "Ground Duel Win %",
        "Pass %",
        "Pass Progression /90",
    ],
    "Full Back": [
        "Ball Wins /90",
        "Ground Duel Win %",
        "Pass %",
        "Passes to Final 3rd /90",
        "Cross Progression /90",
        "Dribble Progression /90",
    ],
    "Defensive Midfielder": [
        "Ball Wins /90",
        "Ball Win Value /90",
        "Pass %",
        "Pass Progression /90",
        "Bypassed Opponents /90",
        "Ball Security %",
        "Ball Losses /90",
    ],
    "Central Midfielder": [
        "Pass %",
        "Successful Passes /90",
        "Pass Progression /90",
        "Bypassed Opponents /90",
        "Passes to Final 3rd /90",
        "Ball Wins /90",
    ],
    "Attacking Midfielder": [
        "Assists /90",
        "xG /90",
        "Shots /90",
        "Passes to Final 3rd /90",
        "Bypassed Opponents /90",
        "Dribble Progression /90",
    ],
    "Forward / Winger": [
        "Goals /90",
        "xG /90",
        "Shots /90",
        "Assists /90",
        "Receiving Progression /90",
        "Dribble Progression /90",
    ],
    "Outfield": [
        "Goals /90",
        "Assists /90",
        "Pass %",
        "Bypassed Opponents /90",
        "Ball Wins /90",
        "Passes to Final 3rd /90",
    ],
}

ROLE_CATEGORY_ORDER = {
    "Goalkeeper": ["Goalkeeping", "Passing", "Progression"],
    "Centre Back": ["Defending", "Passing", "Progression", "Possession"],
    "Full Back": ["Defending", "Progression", "Passing", "Possession"],
    "Defensive Midfielder": ["Defending", "Passing", "Progression", "Possession"],
    "Central Midfielder": ["Passing", "Progression", "Defending", "Possession", "Attacking"],
    "Attacking Midfielder": ["Attacking", "Progression", "Passing", "Possession"],
    "Forward / Winger": ["Attacking", "Progression", "Passing", "Possession"],
    "Outfield": ["Attacking", "Progression", "Passing", "Defending", "Possession"],
}

ROLE_DETAIL_METRICS = {
    "Goalkeeper": [
        "Goals Prevented /90",
        "Save Actions /90",
        "Post-Shot xG Faced /90",
        "Goals Conceded /90",
        "Pass %",
        "Successful Passes /90",
        "Passes to Final 3rd /90",
        "Pass Progression /90",
        "Bypassed Opponents /90",
        "Bypassed Defenders /90",
    ]
}


def _inject_player_css() -> None:
    st.markdown(
        """
        <style>
        .pa-hero {
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

        .pa-hero::before {
            content: "";
            position: absolute;
            inset: 0;
            border-top: 5px solid #c30017;
            pointer-events: none;
        }

        .pa-hero-inner {
            align-items: center;
            display: flex;
            gap: 26px;
            justify-content: space-between;
            position: relative;
            z-index: 1;
        }

        .pa-hero-copy {
            min-width: 0;
        }

        .pa-eyebrow {
            color: #ffffff;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .pa-title {
            color: #ffffff;
            font-size: clamp(2rem, 3vw, 2.8rem);
            line-height: 1.05;
            margin: 0 0 12px;
            font-weight: 850;
        }

        .pa-caption {
            color: rgba(255, 255, 255, 0.80);
            max-width: 900px;
            line-height: 1.55;
            font-size: 1rem;
            margin: 0 0 14px;
        }

        .pa-source {
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 8px;
            color: rgba(255, 255, 255, 0.86);
            padding: 10px 12px;
            margin-top: 10px;
            font-size: 0.9rem;
        }

        .pa-limitation {
            color: #ffecd5;
            background: rgba(251, 146, 60, 0.14);
            border-color: rgba(251, 146, 60, 0.34);
        }

        .pa-badge {
            flex: 0 0 auto;
            width: clamp(76px, 10vw, 120px);
            height: clamp(76px, 10vw, 120px);
            object-fit: contain;
            filter: drop-shadow(0 14px 22px rgba(0, 0, 0, 0.35));
        }

        .pa-section {
            margin: 28px 0 12px;
            color: #667085;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .pa-chart-title {
            color: #172033;
            font-size: clamp(1.35rem, 2vw, 1.75rem);
            font-weight: 850;
            line-height: 1.2;
            margin: 4px 0 14px;
        }

        .pa-card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 12px;
            margin: 8px 0 18px;
        }

        .pa-card {
            border: 1px solid #e6edf5;
            border-radius: 8px;
            background: #ffffff;
            padding: 16px 18px;
            min-height: 120px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            border-top: 3px solid #c30017;
        }

        .pa-card-icon {
            color: #c30017;
            font-weight: 850;
            margin-bottom: 10px;
        }

        .pa-card-title {
            color: #172033;
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .pa-card-body {
            color: #667085;
            font-size: 0.9rem;
            line-height: 1.45;
        }

        @media (max-width: 760px) {
            .pa-hero-inner {
                align-items: flex-start;
                flex-direction: column-reverse;
            }

            .pa-hero {
                padding: 26px 22px;
            }

            .pa-badge {
                width: 78px;
                height: 78px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, caption: str, basis: str = PLAYER_SOURCE, limitation: str | None = None, visualisation_note: bool = True) -> None:
    ui.apply_statsearch_theme()
    _inject_player_css()
    badge = ui.badge_html("pa-badge", "Charlton Athletic crest")
    limitation_html = ""
    if limitation:
        limitation_html = f'<div class="pa-source pa-limitation"><strong>Limitation:</strong> {ui.esc(limitation)}</div>'
    html = (
        '<div class="pa-hero">'
        '<div class="pa-hero-inner">'
        '<div class="pa-hero-copy">'
        '<div class="pa-eyebrow">Charlton Player Analysis</div>'
        f'<h1 class="pa-title">{ui.esc(title)}</h1>'
        f'<p class="pa-caption">{ui.esc(caption)}</p>'
        f'<div class="pa-source"><strong>Data basis:</strong> {ui.esc(basis)}</div>'
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
    st.markdown(f'<div class="pa-section">{ui.esc(label)}</div>', unsafe_allow_html=True)


def chart_title(label: str) -> None:
    st.markdown(f'<div class="pa-chart-title">{ui.esc(label)}</div>', unsafe_allow_html=True)


def analysis_card_grid(cards: list[dict[str, str]]) -> None:
    html = []
    for index, card in enumerate(cards, start=1):
        html.append(
            '<div class="pa-card">'
            f'<div class="pa-card-icon">{index:02d}</div>'
            f'<div class="pa-card-title">{ui.esc(card["title"])}</div>'
            f'<div class="pa-card-body">{ui.esc(card["body"])}</div>'
            "</div>"
        )
    st.markdown(f'<div class="pa-card-grid">{"".join(html)}</div>', unsafe_allow_html=True)


def select_season(key: str | None = None) -> str | None:
    seasons = data.list_seasons().get("players", [])
    if not seasons:
        st.caption("No player season selector is available from the data source.")
        return None
    default = data.preferred_season(seasons)
    return st.selectbox("Season", seasons, index=seasons.index(default), key=key)


def load_player_data(season: str | None = None) -> pd.DataFrame:
    players = data.load_players(season=season).copy().reset_index(drop=True)
    if "Position" in players:
        players["_Position Display"] = players["Position"].apply(ui.clean_position)
    else:
        players["_Position Display"] = "Unknown position"
    for metric in metric_columns(players):
        players[metric] = pd.to_numeric(players[metric], errors="coerce")
    if "Minutes" in players:
        players["Minutes"] = pd.to_numeric(players["Minutes"], errors="coerce")
    return players


def metric_columns(df: pd.DataFrame) -> list[str]:
    return [metric for metric in data.PLAYER_METRICS if metric in df.columns]


def metric_meta(metric: str) -> tuple[str, bool, str]:
    """Public accessor for a metric's (category, higher_is_better, short label)."""
    return _metric_meta(metric)


STYLE_METRIC_CATEGORY_ORDER = ["Attacking", "Passing", "Progression", "Possession", "Defending", "Goalkeeping"]


def style_metric_groups(players: pd.DataFrame) -> dict[str, list[str]]:
    """Every available style metric (data.PLAYER_PROFILE_METRICS), grouped by category.

    Used for league-wide individual rankings, mirroring the team-level
    METRIC_GROUPS structure on the League Rankings page.
    """
    groups: dict[str, list[str]] = {category: [] for category in STYLE_METRIC_CATEGORY_ORDER}
    for metric in data.PLAYER_PROFILE_METRICS:
        if metric not in players or not pd.to_numeric(players[metric], errors="coerce").notna().any():
            continue
        category, _, _ = _metric_meta(metric)
        groups.setdefault(category, []).append(metric)
    return {category: metrics for category, metrics in groups.items() if metrics}


def scatter_metric_columns(df: pd.DataFrame) -> list[str]:
    """Broader metric set for player scatter axes and bubble sizing."""
    preferred_order = [*data.PLAYER_PROFILE_METRICS, "Minutes", "Age"]
    seen = set()
    metrics = []
    for metric in preferred_order:
        if metric in seen or metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce")
        if values.notna().any():
            metrics.append(metric)
            seen.add(metric)
    return metrics or metric_columns(df)


def player_selector(players: pd.DataFrame, key: str, label: str = "Player") -> str:
    names = players["Player"].dropna().astype(str).drop_duplicates().tolist()
    names = sorted(names, key=lambda value: (value.strip().split()[-1].casefold(), value.casefold()))
    selected_from_state = st.session_state.get("selected_player")
    default = names.index(selected_from_state) if selected_from_state in names else 0
    selected = st.selectbox(label, names, index=default, key=key)
    st.session_state["selected_player"] = selected
    return selected


def player_row(players: pd.DataFrame, player_name: str) -> pd.Series:
    matches = players[players["Player"].astype(str) == str(player_name)].copy()
    if "Minutes" in matches:
        matches["_selector_minutes"] = pd.to_numeric(matches["Minutes"], errors="coerce").fillna(-1)
        matches = matches.sort_values("_selector_minutes", ascending=False, kind="stable")
    return matches.iloc[0]


def percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    ranks = values.rank(pct=True, ascending=higher_is_better)
    return (ranks * 100).round(1)


def add_metric_ranks(players: pd.DataFrame) -> pd.DataFrame:
    out = players.copy()
    for metric in metric_columns(out):
        category, higher_is_better, label = _metric_meta(metric)
        out[f"{metric} Rank"] = pd.to_numeric(out[metric], errors="coerce").rank(
            ascending=not higher_is_better,
            method="min",
        ).astype("Int64")
        out[f"{metric} Percentile"] = percentile(out[metric], higher_is_better=higher_is_better)
    return out


def metric_value(value: object, metric: str) -> str:
    return charting.metric_text(value, metric)


def _classify_position_text(text: str) -> str | None:
    text = f" {text.upper().replace('_', ' ')} "
    tokens = set(text.replace(",", " ").split())
    if "GK" in tokens or "GOALKEEPER" in tokens:
        return "Goalkeeper"
    if (
        {"CB", "LCB", "RCB", "DEF"}.intersection(tokens)
        or "CENTRE BACK" in text
        or "CENTER BACK" in text
        or "CENTRAL DEFENDER" in text
        or "CENTRE DEFENDER" in text
        or "CENTER DEFENDER" in text
    ):
        return "Centre Back"
    if (
        {"LB", "RB", "LWB", "RWB", "WB", "FB"}.intersection(tokens)
        or "FULL BACK" in text
        or "LEFT BACK" in text
        or "RIGHT BACK" in text
        or "WINGBACK" in text
        or "WING BACK" in text
    ):
        return "Full Back"
    if {"DM", "DMF", "CDM", "RDMF", "LDMF"}.intersection(tokens) or "DEFENSIVE MIDFIELD" in text or "DEFENSE MIDFIELD" in text:
        return "Defensive Midfielder"
    if {"AM", "AMF", "CAM"}.intersection(tokens) or "ATTACKING MIDFIELD" in text:
        return "Attacking Midfielder"
    if {"CM", "CMF", "LCMF", "RCMF", "MID", "MF"}.intersection(tokens) or "CENTRAL MIDFIELD" in text or "CENTRE MIDFIELD" in text:
        return "Central Midfielder"
    if {"CF", "ST", "FW", "FWD", "LW", "RW", "LWF", "RWF", "WF"}.intersection(tokens) or "WINGER" in text or "FORWARD" in text or "STRIKER" in text:
        return "Forward / Winger"
    return None


def position_group(position: object) -> str:
    text = "" if position is None else str(position)
    primary = text.split(",")[0]
    primary_group = _classify_position_text(primary)
    if primary_group:
        return primary_group
    full_group = _classify_position_text(text)
    if full_group:
        return full_group
    return "Outfield"


def add_position_groups(players: pd.DataFrame) -> pd.DataFrame:
    out = players.copy()
    out["Role Group"] = out["Position"].apply(position_group) if "Position" in out else "Outfield"
    return out


def _profile_numeric(players: pd.DataFrame, metric: str) -> pd.Series:
    return pd.to_numeric(players[metric], errors="coerce") if metric in players else pd.Series(np.nan, index=players.index)


def _metric_meta(metric: str) -> tuple[str, bool, str]:
    return PROFILE_METRIC_META.get(metric, ("General", True, metric))


def _available_metrics(players: pd.DataFrame, metrics: list[str], min_non_null: int = 3) -> list[str]:
    available = []
    for metric in metrics:
        if metric in players and pd.to_numeric(players[metric], errors="coerce").notna().sum() >= min_non_null:
            available.append(metric)
    return available


def profile_categories_for_role(role_group: str) -> list[str]:
    return ROLE_CATEGORY_ORDER.get(role_group, ROLE_CATEGORY_ORDER["Outfield"])


def profile_metrics_for_role(players: pd.DataFrame, role_group: str) -> list[str]:
    if role_group == "Goalkeeper":
        metrics = _available_metrics(players, ROLE_RADAR_METRICS["Goalkeeper"], min_non_null=1)
        return metrics or [metric for metric in ROLE_RADAR_METRICS["Goalkeeper"] if metric in players]

    metrics = _available_metrics(players, ROLE_RADAR_METRICS.get(role_group, ROLE_RADAR_METRICS["Outfield"]))
    if len(metrics) >= 4:
        return metrics
    fallback = _available_metrics(players, ROLE_RADAR_METRICS["Outfield"])
    return (metrics + [metric for metric in fallback if metric not in metrics])[:6]


def profile_peer_group(players: pd.DataFrame, role_group: str, min_peers: int = 8) -> pd.DataFrame:
    grouped = add_position_groups(players)
    peers = grouped[grouped["Role Group"] == role_group].copy()
    if role_group == "Goalkeeper" and not peers.empty:
        return peers
    if len(peers) >= min_peers:
        return peers
    if role_group != "Goalkeeper":
        outfield = grouped[grouped["Role Group"] != "Goalkeeper"].copy()
        if len(outfield) >= min_peers:
            return outfield
    return grouped


def _percentile_value(series: pd.Series, target_index: object, higher_is_better: bool = True) -> float:
    values = pd.to_numeric(series, errors="coerce")
    ranks = values.rank(pct=True, ascending=higher_is_better)
    value = ranks.loc[target_index] if target_index in ranks.index else np.nan
    return round(float(value) * 100, 1) if pd.notna(value) else np.nan


def _rank_value(series: pd.Series, target_index: object, higher_is_better: bool = True) -> int | pd.NA:
    values = pd.to_numeric(series, errors="coerce")
    ranks = values.rank(method="min", ascending=not higher_is_better)
    value = ranks.loc[target_index] if target_index in ranks.index else np.nan
    return int(value) if pd.notna(value) else pd.NA


def _metric_direction_label(higher_is_better: bool) -> str:
    return "Higher raw value is better" if higher_is_better else "Lower raw value is better"


def _performance_band(percentile_value: object) -> str:
    value = pd.to_numeric(pd.Series([percentile_value]), errors="coerce").iloc[0]
    if pd.isna(value):
        return "No data"
    return "Better than peer median" if float(value) >= 50 else "Worse than peer median"


def _performance_colour(percentile_value: object) -> str:
    return PERFORMANCE_GREEN if _performance_band(percentile_value) == "Better than peer median" else PERFORMANCE_RED


PROFILE_METRIC_ROW_COLUMNS = [
    "Category",
    "Metric",
    "Radar Label",
    "Value",
    "Display Value",
    "Role Percentile",
    "Overall Percentile",
    "Role Rank",
    "Higher Is Better",
]


def player_profile_context(players: pd.DataFrame, player_name: str) -> dict[str, object]:
    players = add_position_groups(players)
    row = player_row(players, player_name)
    role = row.get("Role Group", "Outfield")
    peers = profile_peer_group(players, str(role))
    metrics = profile_metrics_for_role(players, str(role))
    rows = []
    for metric in metrics:
        category, higher_is_better, label = _metric_meta(metric)
        peer_series = _profile_numeric(peers, metric)
        overall_series = _profile_numeric(players, metric)
        rows.append(
            {
                "Category": category,
                "Metric": metric,
                "Radar Label": label,
                "Value": row.get(metric),
                "Display Value": metric_value(row.get(metric), metric),
                "Role Percentile": _percentile_value(peer_series, row.name, higher_is_better)
                if row.name in peers.index
                else _percentile_value(overall_series, row.name, higher_is_better),
                "Overall Percentile": _percentile_value(overall_series, row.name, higher_is_better),
                "Role Rank": _rank_value(peer_series, row.name, higher_is_better)
                if row.name in peers.index
                else _rank_value(overall_series, row.name, higher_is_better),
                "Higher Is Better": higher_is_better,
            }
        )
    metric_rows = pd.DataFrame(rows, columns=PROFILE_METRIC_ROW_COLUMNS)
    score = metric_rows["Role Percentile"].dropna().mean() if not metric_rows.empty else np.nan
    return {
        "row": row,
        "role": role,
        "peers": peers,
        "metrics": metric_rows,
        "score": round(float(score), 1) if pd.notna(score) else np.nan,
    }


def profile_category_rows(players: pd.DataFrame, player_name: str) -> pd.DataFrame:
    context = player_profile_context(players, player_name)
    row = context["row"]
    peers = context["peers"]
    role = str(context["role"])
    allowed_categories = set(profile_categories_for_role(role))
    allowed_metrics = ROLE_DETAIL_METRICS.get(role)
    players = add_position_groups(players)
    rows = []
    for metric, (category, higher_is_better, label) in PROFILE_METRIC_META.items():
        if allowed_metrics is not None and metric not in allowed_metrics:
            continue
        if category not in allowed_categories:
            continue
        if metric not in players or pd.isna(row.get(metric)):
            continue
        peer_series = _profile_numeric(peers, metric)
        overall_series = _profile_numeric(players, metric)
        rows.append(
            {
                "Category": category,
                "Metric": metric,
                "Label": label,
                "Value": row.get(metric),
                "Display Value": metric_value(row.get(metric), metric),
                "Role Percentile": _percentile_value(peer_series, row.name, higher_is_better)
                if row.name in peers.index
                else _percentile_value(overall_series, row.name, higher_is_better),
                "Overall Percentile": _percentile_value(overall_series, row.name, higher_is_better),
                "Role Rank": _rank_value(peer_series, row.name, higher_is_better)
                if row.name in peers.index
                else _rank_value(overall_series, row.name, higher_is_better),
            }
        )
    return pd.DataFrame(rows)


def profile_strengths(metric_rows: pd.DataFrame, count: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metric_rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    sorted_rows = metric_rows.dropna(subset=["Role Percentile"]).copy()
    return (
        sorted_rows.sort_values("Role Percentile", ascending=False).head(count),
        sorted_rows.sort_values("Role Percentile", ascending=True).head(count),
    )


def player_profile_radar(metric_rows: pd.DataFrame, player_name: str, role_group: str, score: float | int | None = None) -> go.Figure:
    if metric_rows.empty:
        return go.Figure()
    plot_df = metric_rows.dropna(subset=["Role Percentile"]).copy()
    if plot_df.empty:
        return go.Figure()
    labels = plot_df["Radar Label"].apply(lambda value: charting.wrap_label(value, width=13, max_lines=2)).tolist()
    values = pd.to_numeric(plot_df["Role Percentile"], errors="coerce").fillna(0).tolist()
    closed_labels = labels + [labels[0]]
    closed_values = values + [values[0]]
    customdata = np.vstack(
        [
            plot_df["Metric"],
            plot_df["Display Value"],
            plot_df["Role Percentile"],
            plot_df["Overall Percentile"],
        ]
    ).T
    closed_customdata = np.vstack([customdata, customdata[0]])
    fig = go.Figure(
        go.Scatterpolar(
            r=closed_values,
            theta=closed_labels,
            mode="lines+markers",
            fill="toself",
            line=dict(color=RED, width=3),
            fillcolor="rgba(195, 0, 23, 0.16)",
            marker=dict(size=9, color=RED, line=dict(color="#ffffff", width=1.5)),
            customdata=closed_customdata,
            hovertemplate=(
                "%{customdata[0]}"
                "<br>Value: %{customdata[1]}"
                "<br>Role percentile: %{customdata[2]:.0f}"
                "<br>Overall percentile: %{customdata[3]:.0f}<extra></extra>"
            ),
        )
    )
    score_text = "N/A" if score is None or pd.isna(score) else f"{float(score):.0f}"
    fig.add_annotation(
        text=f"<b>{score_text}</b><br><span style='font-size:11px'>{role_group}<br>score</span>",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=18, color=DARK),
        align="center",
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Arial, sans-serif", color=DARK, size=12),
        height=520,
        margin=dict(l=34, r=34, t=28, b=34),
        polar=dict(
            bgcolor="#ffffff",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80],
                tickfont=dict(size=10, color=GREY),
                gridcolor="#e9edf3",
                linecolor="#e9edf3",
                angle=90,
            ),
            angularaxis=dict(
                rotation=90,
                direction="clockwise",
                tickfont=dict(size=11, color=DARK),
                gridcolor="#eef2f6",
                linecolor="#eef2f6",
            ),
        ),
        showlegend=False,
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=LIGHT_GREY, font_size=13, font_color=DARK),
    )
    return fig


def format_age(birthdate: object) -> str:
    date = pd.to_datetime(birthdate, errors="coerce")
    if pd.isna(date):
        return "Unknown"
    today = pd.Timestamp.today()
    age = today.year - date.year - ((today.month, today.day) < (date.month, date.day))
    return f"{age:.0f}"


def radar_rows(players: pd.DataFrame, player_name: str) -> pd.DataFrame:
    row = player_row(players, player_name)
    rows = []
    for metric in metric_columns(players):
        category, higher_is_better, label = _metric_meta(metric)
        metric_pct = percentile(players[metric], higher_is_better=higher_is_better)
        metric_rank = players[metric].rank(ascending=not higher_is_better, method="min")
        rows.append(
            {
                "Metric": metric,
                "Value": metric_value(row.get(metric), metric),
                "Percentile": float(metric_pct.loc[row.name]) if row.name in metric_pct.index else np.nan,
                "Rank": int(metric_rank.loc[row.name]) if row.name in metric_rank.index else 0,
            }
        )
    return pd.DataFrame(rows)


def polish_figure(fig: go.Figure, title: str | None = None) -> go.Figure:
    return charting.polish_figure(fig, title)


def player_radar(labels: list[str], values: list[float], name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=[charting.wrap_label(label, width=15, max_lines=2) for label in labels + [labels[0]]],
            fill="toself",
            name=name,
            line=dict(color=RED, width=3),
            fillcolor="rgba(215, 25, 32, 0.22)",
            hovertemplate="%{theta}<br>Percentile: %{r:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickformat=".0f"),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        showlegend=False,
        height=520,
    )
    return polish_figure(fig)


def ranked_bar(
    df: pd.DataFrame,
    metric: str,
    selected: str | None = None,
    top_n: int | None = None,
    higher_is_better: bool = True,
) -> go.Figure:
    plot_df = df.copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])
    plot_df["_Rank"] = plot_df[metric].rank(
        ascending=not higher_is_better,
        method="min",
    ).astype("Int64")
    if top_n:
        selected_row = plot_df[plot_df["Player"].astype(str) == str(selected)] if selected else pd.DataFrame()
        plot_df = plot_df.sort_values("_Rank", ascending=True).head(top_n)
        if selected is not None and selected not in plot_df["Player"].astype(str).tolist() and not selected_row.empty:
            plot_df = pd.concat([plot_df, selected_row], ignore_index=True)

    plot_df = plot_df.sort_values("_Rank", ascending=False)
    colors = [RED if str(player) == str(selected) else GREY for player in plot_df["Player"]]
    plot_df["_Label"] = plot_df["Player"].apply(lambda value: charting.wrap_label(value, width=18, max_lines=2))
    plot_df["_Text"] = charting.outside_bar_text(plot_df[metric], metric)
    fig = go.Figure(
        go.Bar(
            x=plot_df[metric],
            y=plot_df["_Label"],
            orientation="h",
            marker_color=colors,
            text=plot_df["_Text"],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([plot_df["Player"], plot_df["_Text"]], axis=-1),
            hovertemplate="%{customdata[0]}<br>" + metric + ": %{customdata[1]}<extra></extra>",
        )
    )
    fig.update_layout(
        height=charting.horizontal_bar_height(len(plot_df), row_height=32),
        xaxis_title=metric,
        yaxis_title="",
        showlegend=False,
    )
    charting.format_xaxis(fig, metric)
    fig = polish_figure(fig, f"{metric} player ranking")
    fig.update_layout(margin=dict(l=36, r=78, t=68, b=54))
    return fig


def metric_scatter(
    players: pd.DataFrame,
    x: str,
    y: str,
    selected: str | None = None,
    size: str | None = None,
    color: str | None = None,
    title: str | None = None,
    show_title: bool = True,
    show_legend: bool = False,
    highlight_players: list[str] | None = None,
    highlight_teams: list[str] | None = None,
    highlight_top_x: bool = False,
    highlight_top_y: bool = False,
    top_n: int = 10,
    highlight_u21: bool = False,
    highlight_u19: bool = False,
    color_metric: str | None = None,
    label_highlights: bool = True,
    show_median_lines: bool = False,
) -> go.Figure:
    plot_df = players.copy()
    numeric_metrics = [metric for metric in [x, y, size, color_metric] if metric in plot_df.columns]
    for metric in numeric_metrics:
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[x, y]).copy()
    if plot_df.empty:
        return polish_figure(go.Figure(), title if show_title else None)

    highlight_players = [str(player) for player in (highlight_players or [])]
    highlight_teams = [str(team) for team in (highlight_teams or [])]
    selected_text = "" if selected is None else str(selected)

    top_x_players: set[str] = set()
    top_y_players: set[str] = set()
    if highlight_top_x:
        top_x_players = set(plot_df.nlargest(min(top_n, len(plot_df)), x)["Player"].astype(str))
    if highlight_top_y:
        top_y_players = set(plot_df.nlargest(min(top_n, len(plot_df)), y)["Player"].astype(str))

    def highlight_reason(row: pd.Series) -> str:
        reasons = []
        player = str(row.get("Player", ""))
        team = str(row.get("Team", ""))
        if selected_text and player == selected_text:
            reasons.append("Selected player")
        if player in highlight_players and "Selected player" not in reasons:
            reasons.append("Highlighted player")
        if team in highlight_teams:
            reasons.append("Highlighted team")
        if player in top_x_players:
            reasons.append(f"Top {min(top_n, len(plot_df))} X")
        if player in top_y_players:
            reasons.append(f"Top {min(top_n, len(plot_df))} Y")
        age_value = pd.to_numeric(pd.Series([row.get("_Age")]), errors="coerce").iloc[0] if "_Age" in row else np.nan
        if highlight_u19 and pd.notna(age_value) and age_value <= 19:
            reasons.append("U19")
        elif highlight_u21 and pd.notna(age_value) and age_value <= 21:
            reasons.append("U21")
        return ", ".join(reasons) if reasons else "None"

    plot_df["_Highlight Reason"] = plot_df.apply(highlight_reason, axis=1)
    plot_df["_Highlighted"] = plot_df["_Highlight Reason"] != "None"

    if size in plot_df.columns:
        size_values = pd.to_numeric(plot_df[size], errors="coerce")
        size_min = size_values.min()
        size_max = size_values.max()
        if pd.notna(size_min) and pd.notna(size_max) and size_max > size_min:
            # A provider metric can be populated for most players while still
            # being null for a few rows. Plotly rejects NaN marker sizes, so
            # keep those players visible at the neutral/default size.
            plot_df["_Marker Size"] = (
                8 + ((size_values - size_min) / (size_max - size_min) * 16)
            ).fillna(11)
        else:
            plot_df["_Marker Size"] = 11
    else:
        plot_df["_Marker Size"] = 11

    if label_highlights:
        plot_df["_Text"] = np.where(
            plot_df["_Highlighted"],
            plot_df["Player"].apply(lambda value: charting.wrap_label(value, width=15, max_lines=2)),
            "",
        )
    else:
        plot_df["_Text"] = charting.selected_text(plot_df["Player"], selected)

    custom_cols = [
        plot_df["Player"].fillna("Unknown"),
        plot_df["Team"].fillna("Unknown") if "Team" in plot_df else pd.Series("Unknown", index=plot_df.index),
        plot_df["_Position Display"].fillna("Unknown") if "_Position Display" in plot_df else pd.Series("Unknown", index=plot_df.index),
        plot_df["Minutes"] if "Minutes" in plot_df else pd.Series(np.nan, index=plot_df.index),
        plot_df[x],
        plot_df[y],
        plot_df[size] if size in plot_df.columns else pd.Series(np.nan, index=plot_df.index),
        plot_df["_Highlight Reason"],
        plot_df[color_metric] if color_metric in plot_df.columns else pd.Series(np.nan, index=plot_df.index),
    ]
    customdata = np.stack(custom_cols, axis=-1)

    fig = go.Figure()
    marker = dict(
        size=plot_df["_Marker Size"],
        opacity=0.56,
        line=dict(width=0.8, color="#ffffff"),
    )
    if color_metric in plot_df.columns and pd.to_numeric(plot_df[color_metric], errors="coerce").notna().any():
        marker.update(
            color=plot_df[color_metric],
            colorscale=[
                [0, "#0b2a53"],
                [0.5, "#8b95a3"],
                [1, "#f4d03f"],
            ],
            colorbar=dict(
                title=dict(text=charting.wrap_label(color_metric, width=12, max_lines=3), side="right"),
                thickness=14,
                len=0.70,
                outlinewidth=0,
            ),
        )
    else:
        marker.update(color="rgba(52, 64, 84, 0.48)")

    hover = (
        "<b>%{customdata[0]}</b>"
        "<br>Team: %{customdata[1]}"
        "<br>Position: %{customdata[2]}"
        f"<br>{x}: {charting.hover_value('customdata[4]', x)}"
        f"<br>{y}: {charting.hover_value('customdata[5]', y)}"
        "<br>Minutes: %{customdata[3]:,.0f}"
    )
    if size in plot_df.columns:
        hover += f"<br>{size}: {charting.hover_value('customdata[6]', size)}"
    if color_metric in plot_df.columns:
        hover += f"<br>{color_metric}: {charting.hover_value('customdata[8]', color_metric)}"
    hover += "<br>Highlight: %{customdata[7]}<extra></extra>"

    fig.add_trace(
        go.Scatter(
            x=plot_df[x],
            y=plot_df[y],
            mode="markers",
            name="Player pool",
            marker=marker,
            customdata=customdata,
            hovertemplate=hover,
            showlegend=False,
        )
    )

    highlighted = plot_df[plot_df["_Highlighted"]].copy()
    if not highlighted.empty:
        fig.add_trace(
            go.Scatter(
                x=highlighted[x],
                y=highlighted[y],
                mode="markers+text",
                name="Highlighted",
                text=highlighted["_Text"],
                textposition="top center",
                marker=dict(
                    size=np.maximum(pd.to_numeric(highlighted["_Marker Size"], errors="coerce").fillna(11) + 7, 17),
                    color=RED,
                    opacity=0.96,
                    line=dict(width=1.8, color="#ffffff"),
                ),
                customdata=customdata[plot_df["_Highlighted"].to_numpy()],
                hovertemplate=hover,
                showlegend=show_legend,
            )
        )

    if show_median_lines:
        x_median = pd.to_numeric(plot_df[x], errors="coerce").median()
        y_median = pd.to_numeric(plot_df[y], errors="coerce").median()
        if pd.notna(x_median):
            fig.add_vline(x=x_median, line=dict(color="#98a2b3", width=1.2, dash="dash"), opacity=0.70)
        if pd.notna(y_median):
            fig.add_hline(y=y_median, line=dict(color="#98a2b3", width=1.2, dash="dash"), opacity=0.70)

    fig.update_traces(
        cliponaxis=False,
    )
    fig.update_layout(xaxis_title=x, yaxis_title=y)
    charting.format_xaxis(fig, x)
    charting.format_yaxis(fig, y)
    chart_title = charting.wrap_label(title or f"{x} vs {y}", width=58, max_lines=2)
    fig = polish_figure(fig, chart_title if show_title else None)
    fig.update_layout(
        height=650,
        showlegend=show_legend,
        margin=dict(l=42, r=72 if color_metric in plot_df.columns else 36, t=78 if show_title else 36, b=68),
        plot_bgcolor="#f7f8fa",
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
    )
    return fig


def _comparison_selected_rows(players: pd.DataFrame, player_refs: list[object]) -> list[tuple[object, pd.Series]]:
    selected = []
    used = set()
    for ref in player_refs:
        if ref in players.index:
            idx = ref
        else:
            matches = players[players["Player"].astype(str) == str(ref)]
            if matches.empty:
                continue
            idx = matches.index[0]
        if idx in used:
            continue
        used.add(idx)
        selected.append((idx, players.loc[idx]))
    return selected


def comparison_percentile_rows(players: pd.DataFrame, player_refs: list[object]) -> pd.DataFrame:
    players = add_position_groups(players)
    selected = _comparison_selected_rows(players, player_refs)
    metrics: list[str] = []
    for _, row in selected:
        role = str(row.get("Role Group", "Outfield"))
        for metric in profile_metrics_for_role(players, role):
            if metric in players and metric not in metrics:
                metrics.append(metric)
    if not metrics:
        metrics = metric_columns(players)

    rows = []
    duplicate_names = pd.Series([str(row.get("Player", "")) for _, row in selected]).duplicated(keep=False).any()
    for idx, row in selected:
        role = str(row.get("Role Group", "Outfield"))
        peers = profile_peer_group(players, role)
        player_name = str(row.get("Player", "Unknown"))
        player_label = f"{player_name} | {row.get('Team', 'Unknown')}" if duplicate_names else player_name
        for metric in metrics:
            if metric not in players:
                continue
            category, higher_is_better, label = _metric_meta(metric)
            peer_series = _profile_numeric(peers, metric)
            overall_series = _profile_numeric(players, metric)
            percentile_value = (
                _percentile_value(peer_series, idx, higher_is_better)
                if idx in peers.index
                else _percentile_value(overall_series, idx, higher_is_better)
            )
            rank_value = (
                _rank_value(peer_series, idx, higher_is_better)
                if idx in peers.index
                else _rank_value(overall_series, idx, higher_is_better)
            )
            rows.append(
                {
                    "Player": player_name,
                    "Player Label": player_label,
                    "Team": row.get("Team", "Unknown"),
                    "Position": row.get("_Position Display", row.get("Position", "Unknown")),
                    "Role Group": role,
                    "Category": category,
                    "Metric": metric,
                    "Metric Label": f"<b>{charting.wrap_label(label, width=14, max_lines=2)}</b>",
                    "Value": row.get(metric),
                    "Display Value": metric_value(row.get(metric), metric),
                    "Percentile": percentile_value,
                    "Rank": rank_value,
                    "Peer Group": role if idx in peers.index else "All players",
                    "Higher Is Better": higher_is_better,
                    "Direction": _metric_direction_label(higher_is_better),
                    "Performance": _performance_band(percentile_value),
                }
            )
    return pd.DataFrame(rows)


def comparison_chart(players: pd.DataFrame, player_names: list[object]) -> go.Figure:
    plot_df = comparison_percentile_rows(players, player_names)
    if plot_df.empty:
        return go.Figure()
    plot_df = plot_df.dropna(subset=["Percentile"]).copy()
    plot_df["Text"] = plot_df["Percentile"].apply(lambda value: charting.metric_text(value, "Percentile"))
    metric_order = plot_df[["Metric", "Metric Label"]].drop_duplicates()["Metric Label"].tolist()
    fig = go.Figure()
    for player_label in plot_df["Player Label"].drop_duplicates().tolist():
        player_df = plot_df[plot_df["Player Label"] == player_label].copy()
        player_df["Metric Label"] = pd.Categorical(player_df["Metric Label"], categories=metric_order, ordered=True)
        player_df = player_df.sort_values("Metric Label")
        customdata = np.stack(
            [
                player_df["Player"],
                player_df["Team"],
                player_df["Role Group"],
                player_df["Metric"],
                player_df["Display Value"],
                player_df["Peer Group"],
                player_df["Direction"],
                player_df["Performance"],
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Bar(
                x=player_df["Metric Label"],
                y=player_df["Percentile"],
                name=player_label,
                offsetgroup=player_label,
                marker=dict(
                    color=[_performance_colour(value) for value in player_df["Percentile"]],
                    line=dict(color="#ffffff", width=1.2),
                ),
                text=player_df["Text"],
                textposition="outside",
                cliponaxis=False,
                customdata=customdata,
                hovertemplate=(
                    "%{customdata[0]}<br>%{customdata[1]}<br>Role: %{customdata[2]}"
                    "<br>%{customdata[3]}: %{customdata[4]}"
                    "<br>Performance percentile: %{y:.0f}"
                    "<br>%{customdata[6]}"
                    "<br>%{customdata[7]}"
                    "<br>Peer group: %{customdata[5]}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    fig.add_hrect(y0=50, y1=100, fillcolor="rgba(21, 128, 61, 0.055)", line_width=0, layer="below")
    fig.add_hrect(y0=0, y1=50, fillcolor="rgba(220, 38, 38, 0.045)", line_width=0, layer="below")
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color=LIGHT_GREY,
        annotation_text="Peer median",
        annotation_position="top left",
        annotation_font=dict(size=11, color=GREY),
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=11, color=PERFORMANCE_GREEN),
            name="Better than peer median",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=11, color=PERFORMANCE_RED),
            name="Worse than peer median",
            showlegend=True,
        )
    )
    fig.update_layout(
        barmode="group",
        height=640,
        yaxis_range=[0, 106],
        yaxis_title="Performance Percentile (higher is better)",
        xaxis_title="",
        bargap=0.18,
    )
    fig.update_yaxes(tickformat=".0f")
    fig.update_xaxes(categoryorder="array", categoryarray=metric_order, tickfont=dict(size=12, color=DARK), tickangle=0)
    fig = polish_figure(fig, "Player Percentile Comparison")
    fig.update_layout(
        margin=dict(l=42, r=42, t=90, b=150),
        legend_title_text="",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
        ),
    )
    return fig


def passing_table(players: pd.DataFrame) -> pd.DataFrame:
    out = players.copy()
    components = []
    for metric in ["Pass %", "Passes to Final 3rd /90", "Bypassed Opponents /90"]:
        if metric in out:
            components.append(percentile(out[metric]))
    out["Passing Impact"] = pd.concat(components, axis=1).mean(axis=1).round(1) if components else np.nan
    return out


def similarity_table(players: pd.DataFrame, player_name: str, top_n: int = 8, metrics: list[str] | None = None) -> pd.DataFrame:
    if metrics is None:
        metrics = _available_metrics(players, data.PLAYER_PROFILE_METRICS)
        if not metrics:
            metrics = metric_columns(players)
    else:
        metrics = [metric for metric in metrics if metric in players]
    if not metrics or len(players) <= 1:
        return pd.DataFrame()
    numeric = players[metrics].apply(pd.to_numeric, errors="coerce")
    filled = numeric.copy()
    for metric in metrics:
        median = filled[metric].median()
        filled[metric] = filled[metric].fillna(0 if pd.isna(median) else median)
    ranges = (filled.max() - filled.min()).replace(0, 1)
    scaled = (filled - filled.min()) / ranges
    target_index = players[players["Player"].astype(str) == str(player_name)].index[0]
    distance = ((scaled - scaled.loc[target_index]) ** 2).sum(axis=1) ** 0.5
    similar = players.assign(_distance=distance).drop(index=target_index).sort_values("_distance").head(top_n).copy()
    max_distance = similar["_distance"].max()
    if pd.isna(max_distance) or max_distance == 0:
        max_distance = 1
    similar["Similarity"] = (100 - (similar["_distance"] / max_distance * 35)).round(1)
    return similar.drop(columns=["_distance"])


def safe_key(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_")
