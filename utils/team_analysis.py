import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import charting, data, team_badges, ui

TEAM_SOURCE = (
    "Team metrics come from CAFC_DB Impect squad-iteration KPI facts, including attacking, "
    "passing, progression, and defensive performance indicators."
)
TEAM_STYLE_SOURCE = (
    "Team style metrics come from CAFC_DB Impect squad-iteration KPI facts. Percentiles "
    "are calculated relative to the league average for the selected season."
)
MATCH_SOURCE = (
    "Match trends use typed CAFC_DB Impect match and squad dimensions; scores are reconstructed "
    "from the provider event feed, including own-goal attribution."
)
ACTION_SOURCE = (
    "Action visuals use CAFC_DB Impect provider events through the app's normalised adapter. "
    "They are counts of event labels, not location-aware event maps."
)

RED = ui.CHARLTON_RED
DARK = ui.CHARLTON_BLACK
GREY = "#7a7f87"
LIGHT_GREY = ui.CHARLTON_BORDER
BLUE = ui.CHARLTON_DEEP_RED
SOFT_BLUE = "#fff1f3"

TEAM_METRIC_META = {
    "Goals /90": ("Attacking", True, "Goals"),
    "Assists /90": ("Attacking", True, "Assists"),
    "xG /90": ("Attacking", True, "xG"),
    "Packing xG /90": ("Attacking", True, "Packing xG"),
    "Shots /90": ("Attacking", True, "Shots"),
    "Pass %": ("Passing", True, "Pass %"),
    "Passes to Final 3rd /90": ("Progression", True, "Final Third Passes"),
    "Bypassed Opponents /90": ("Progression", True, "Opponents Bypassed"),
    "Dribble Progression /90": ("Progression", True, "Dribble Progression"),
    "Ball Wins /90": ("Defensive", True, "Ball Wins"),
    "Ball Win Value /90": ("Defensive", True, "Ball Win Value"),
}


def _inject_team_css() -> None:
    st.markdown(
        """
        <style>
        .ta-hero {
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

        .ta-hero::before {
            content: "";
            position: absolute;
            inset: 0;
            border-top: 5px solid #c30017;
            pointer-events: none;
        }

        .ta-hero-inner {
            align-items: center;
            display: flex;
            gap: 26px;
            justify-content: space-between;
            position: relative;
            z-index: 1;
        }

        .ta-hero-copy {
            min-width: 0;
        }

        .ta-eyebrow {
            color: #ffffff;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .ta-title {
            color: #ffffff;
            font-size: clamp(2rem, 3vw, 2.8rem);
            line-height: 1.05;
            margin: 0 0 12px;
            font-weight: 850;
        }

        .ta-caption {
            color: rgba(255, 255, 255, 0.80);
            max-width: 880px;
            line-height: 1.55;
            font-size: 1rem;
            margin: 0 0 14px;
        }

        .ta-source {
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 8px;
            color: rgba(255, 255, 255, 0.86);
            padding: 10px 12px;
            margin-top: 10px;
            font-size: 0.9rem;
        }

        .ta-limitation {
            color: #ffecd5;
            background: rgba(251, 146, 60, 0.14);
            border-color: rgba(251, 146, 60, 0.34);
        }

        .ta-badge {
            flex: 0 0 auto;
            width: clamp(76px, 10vw, 120px);
            height: clamp(76px, 10vw, 120px);
            object-fit: contain;
            filter: drop-shadow(0 14px 22px rgba(0, 0, 0, 0.35));
        }

        .ta-section {
            margin: 28px 0 12px;
            color: #667085;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .ta-card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 12px;
            margin: 8px 0 18px;
        }

        .ta-card {
            border: 1px solid #e6edf5;
            border-radius: 8px;
            background: #ffffff;
            padding: 16px 18px;
            min-height: 120px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            border-top: 3px solid #c30017;
        }

        .ta-card-icon {
            color: #c30017;
            font-weight: 850;
            margin-bottom: 10px;
        }

        .ta-card-title {
            color: #172033;
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .ta-card-body {
            color: #667085;
            font-size: 0.9rem;
            line-height: 1.45;
        }

        @media (max-width: 760px) {
            .ta-hero-inner {
                align-items: flex-start;
                flex-direction: column-reverse;
            }

            .ta-hero {
                padding: 26px 22px;
            }

            .ta-badge {
                width: 78px;
                height: 78px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, caption: str, basis: str, limitation: str | None = None, visualisation_note: bool = True) -> None:
    ui.apply_statsearch_theme()
    _inject_team_css()
    badge = ui.badge_html("ta-badge", "Charlton Athletic crest")
    limitation_html = ""
    if limitation:
        limitation_html = f'<div class="ta-source ta-limitation"><strong>Limitation:</strong> {ui.esc(limitation)}</div>'
    html = (
        '<div class="ta-hero">'
        '<div class="ta-hero-inner">'
        '<div class="ta-hero-copy">'
        '<div class="ta-eyebrow">Charlton Team Analysis</div>'
        f'<h1 class="ta-title">{ui.esc(title)}</h1>'
        f'<p class="ta-caption">{ui.esc(caption)}</p>'
        f'<div class="ta-source"><strong>Data basis:</strong> {ui.esc(basis)}</div>'
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
    st.markdown(f'<div class="ta-section">{ui.esc(label)}</div>', unsafe_allow_html=True)


def analysis_card_grid(cards: list[dict[str, str]]) -> None:
    card_html = []
    for index, card in enumerate(cards, start=1):
        card_html.append(
            f"""
            <div class="ta-card">
                <div class="ta-card-icon">{index:02d}</div>
                <div class="ta-card-title">{ui.esc(card["title"])}</div>
                <div class="ta-card-body">{ui.esc(card["body"])}</div>
            </div>
            """
        )
    st.markdown(f'<div class="ta-card-grid">{"".join(card_html)}</div>', unsafe_allow_html=True)


def polish_figure(fig: go.Figure, title: str | None = None) -> go.Figure:
    return charting.polish_figure(fig, title)


def select_season(kind: str = "teams", key: str | None = None) -> str | None:
    seasons = data.list_seasons().get(kind, [])
    if not seasons:
        st.caption("No season selector is available from the data source.")
        return None
    default = data.preferred_season(seasons)
    return st.selectbox("Season", seasons, index=seasons.index(default), key=key)


def load_team_data(season: str | None = None) -> pd.DataFrame:
    teams = data.load_teams(season=season).copy()
    for metric in metric_columns(teams):
        teams[metric] = pd.to_numeric(teams[metric], errors="coerce")
    return teams


def load_team_style_data(season: str | None = None) -> pd.DataFrame:
    teams = data.load_team_iteration_rollups(season=season).copy()
    for metric in metric_columns(teams):
        teams[metric] = pd.to_numeric(teams[metric], errors="coerce").round(2)
    return teams


def load_player_data(season: str | None = None) -> pd.DataFrame:
    players = data.load_players(season=season).copy()
    for metric in data.PLAYER_METRICS:
        if metric in players:
            players[metric] = pd.to_numeric(players[metric], errors="coerce")
    if "Minutes" in players:
        players["Minutes"] = pd.to_numeric(players["Minutes"], errors="coerce")
    return players


def metric_columns(df: pd.DataFrame) -> list[str]:
    return [metric for metric in data.TEAM_METRICS if metric in df.columns]


def team_selector(teams: pd.DataFrame, key: str, label: str = "Team") -> str:
    team_names = teams["Team"].dropna().astype(str).tolist()
    if key in st.session_state and st.session_state[key] not in team_names:
        del st.session_state[key]
    charlton_matches = [index for index, team in enumerate(team_names) if "charlton" in team.lower()]
    default = charlton_matches[0] if charlton_matches else 0
    return st.selectbox(label, team_names, index=default, key=key)


def regular_season_fixtures(matches: pd.DataFrame) -> pd.DataFrame:
    """Keep the first two completed meetings for each league opponent pair."""
    required = {"Home", "Away", "Home Goals", "Away Goals"}
    if matches.empty or not required.issubset(matches.columns):
        return pd.DataFrame(columns=matches.columns)

    rows = matches.copy()
    rows = rows[rows["Home Goals"].notna() & rows["Away Goals"].notna()]
    if "Date" in rows:
        rows["Date"] = pd.to_datetime(rows["Date"], errors="coerce", utc=True)
        rows = rows[rows["Date"].le(pd.Timestamp.now(tz="UTC"))]
    sort_columns = [column for column in ["Date", "MatchId"] if column in rows.columns]
    if sort_columns:
        rows = rows.sort_values(sort_columns, kind="stable")

    rows["_Opponent Pair"] = [
        tuple(sorted((str(home), str(away))))
        for home, away in zip(rows["Home"], rows["Away"])
    ]
    rows["_Pair Meeting"] = rows.groupby("_Opponent Pair", sort=False).cumcount()
    return rows[rows["_Pair Meeting"].lt(2)].drop(columns=["_Opponent Pair", "_Pair Meeting"]).reset_index(drop=True)


def percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    ranks = values.rank(ascending=higher_is_better, pct=True)
    return (ranks * 100).round(1)


def metric_rank_table(df: pd.DataFrame, metric: str, higher_is_better: bool = True) -> pd.DataFrame:
    if df.empty or "Team" not in df or metric not in df:
        return pd.DataFrame(columns=["Team", "Value", "Rank", "Percentile"])

    ranked = df[["Team", metric]].copy()
    ranked["Value"] = pd.to_numeric(ranked[metric], errors="coerce")
    ranked = ranked.dropna(subset=["Team", "Value"])
    if ranked.empty:
        return pd.DataFrame(columns=["Team", "Value", "Rank", "Percentile"])

    ranked["Rank"] = ranked["Value"].rank(
        ascending=not higher_is_better,
        method="min",
    ).astype("Int64")
    ranked["Percentile"] = ranked["Value"].rank(
        ascending=higher_is_better,
        pct=True,
    ).mul(100).round(1)
    return ranked[["Team", "Value", "Rank", "Percentile"]].sort_values(["Rank", "Team"])


def add_metric_ranks(teams: pd.DataFrame) -> pd.DataFrame:
    out = teams.copy()
    for metric in metric_columns(out):
        higher = TEAM_METRIC_META.get(metric, (None, True, None))[1]
        out[f"{metric} Rank"] = pd.to_numeric(out[metric], errors="coerce").rank(
            ascending=not higher,
            method="min",
        ).astype("Int64")
        out[f"{metric} Percentile"] = percentile(out[metric], higher_is_better=higher)
    return out


def style_scores(teams: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"Team": teams["Team"]})

    def score(metric: str) -> pd.Series:
        if metric not in teams:
            return pd.Series(np.nan, index=teams.index)
        higher = TEAM_METRIC_META.get(metric, (None, True, None))[1]
        return percentile(teams[metric], higher_is_better=higher)

    out["Scoring"] = score("Goals /90")
    out["Creation"] = score("Assists /90")
    out["Progression"] = pd.concat(
        [score("Bypassed Opponents /90"), score("Passes to Final 3rd /90")],
        axis=1,
    ).mean(axis=1).round(1)
    out["Ball Security"] = score("Pass %")
    out["Final Third"] = score("Passes to Final 3rd /90")
    out["Territory"] = pd.concat(
        [score("Passes to Final 3rd /90"), score("Bypassed Opponents /90")],
        axis=1,
    ).mean(axis=1).round(1)
    out["Control Proxy"] = pd.concat(
        [out["Ball Security"], out["Progression"]],
        axis=1,
    ).mean(axis=1).round(1)
    out["Metric Balance"] = pd.concat(
        [score("Pass %"), score("Bypassed Opponents /90"), score("Passes to Final 3rd /90")],
        axis=1,
    ).mean(axis=1).round(1)
    return out


def selected_team_style(teams: pd.DataFrame, team_name: str) -> pd.Series | None:
    scores = style_scores(teams)
    matched = scores[scores["Team"] == team_name]
    return matched.iloc[0] if not matched.empty else None


def highlight_colors(values: pd.Series, selected: str | None = None) -> list[str]:
    return [RED if str(value) == selected else GREY for value in values]


def ranked_bar(
    df: pd.DataFrame,
    metric: str,
    selected: str | None = None,
    title: str | None = None,
    top_n: int | None = None,
    higher_is_better: bool = True,
    show_rank: bool = False,
) -> go.Figure:
    plot_df = df.copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])
    plot_df["_Rank"] = plot_df[metric].rank(
        ascending=not higher_is_better,
        method="min",
    ).astype("Int64")
    if top_n:
        selected_row = plot_df[plot_df["Team"] == selected] if selected else pd.DataFrame()
        plot_df = plot_df[plot_df["_Rank"].le(top_n)]
        if selected is not None and selected not in plot_df["Team"].astype(str).tolist() and not selected_row.empty:
            plot_df = pd.concat([plot_df, selected_row], ignore_index=True)
    plot_df = plot_df.sort_values("_Rank", ascending=False)
    if show_rank:
        plot_df["_Label"] = [
            f"{int(rank)}. {charting.wrap_label(team, width=18, max_lines=2)}"
            for team, rank in zip(plot_df["Team"], plot_df["_Rank"])
        ]
    else:
        plot_df["_Label"] = plot_df["Team"].apply(lambda value: charting.wrap_label(value, width=18, max_lines=2))
    plot_df["_Text"] = charting.outside_bar_text(plot_df[metric], metric)

    fig = go.Figure(
        go.Bar(
            x=plot_df[metric],
            y=plot_df["_Label"],
            orientation="h",
            marker_color=highlight_colors(plot_df["Team"], selected),
            text=plot_df["_Text"],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([plot_df["Team"], plot_df["_Text"]], axis=-1),
            hovertemplate="%{customdata[0]}<br>" + metric + ": %{customdata[1]}<extra></extra>",
        )
    )
    fig.update_layout(
        height=charting.horizontal_bar_height(len(plot_df), row_height=30),
        xaxis_title=metric,
        yaxis_title="",
        showlegend=False,
    )
    charting.format_xaxis(fig, metric)
    fig = polish_figure(fig, title)
    fig.update_layout(margin=dict(l=36, r=76, t=68 if title else 36, b=54))
    return fig


def team_radar(
    labels: list[str], 
    values: list[float] | list[list[float]], 
    names: str | list[str], 
    height: int = 500,
    colors: list[str] | None = None
) -> go.Figure:
    if not labels:
        return go.Figure()
        
    fig = go.Figure()
    
    # Normalise inputs to lists
    if not isinstance(names, list):
        names = [names]
    if values and not isinstance(values[0], list):
        values = [values] # type: ignore
        
    if not colors:
        colors = [RED, DARK, "#16a34a", "#f59e0b"]
        
    for i, (name, val_set) in enumerate(zip(names, values)):
        color = colors[i % len(colors)]
        # Add opacity for fill
        rgba_color = f"rgba{tuple(list(ui.hex_to_rgb(color)) + [0.22])}" if color.startswith("#") else color
        
        fig.add_trace(
            go.Scatterpolar(
                r=list(val_set) + [val_set[0]],
                theta=[charting.wrap_label(label, width=12, max_lines=3) for label in labels + [labels[0]]],
                fill="toself",
                name=name,
                line=dict(color=color, width=3),
                fillcolor=rgba_color,
                hovertemplate=f"<b>{name}</b><br>%{{theta}}<br>Percentile: %{{r:.0f}}<extra></extra>",
            )
        )

    fig.update_layout(
        polar=dict(
            domain=dict(x=[0.05, 0.95], y=[0.05, 0.95]),
            radialaxis=dict(visible=True, range=[0, 100], tickformat=".0f", ticksuffix=""),
            angularaxis=dict(
                tickfont=dict(
                    size=13,
                    color=DARK,
                    family="Inter SemiBold, Arial, sans-serif",
                )
            ),
        ),
        showlegend=len(names) > 1,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        height=height,
        margin=dict(l=80, r=80, t=55, b=70),
    )
    fig = polish_figure(fig)
    return fig


def metric_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    selected: str | None = None,
    size: str | None = None,
    title: str | None = None,
) -> go.Figure:
    plot_df = df.copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[x, y]).reset_index(drop=True)
    plot_df["_Plot X"] = plot_df[x]
    plot_df["_Plot Y"] = plot_df[y]

    duplicate_count = plot_df.groupby([x, y])["Team"].transform("size")
    duplicate_index = plot_df.groupby([x, y]).cumcount()
    duplicate_mask = duplicate_count.gt(1)
    if duplicate_mask.any():
        x_span = plot_df[x].max() - plot_df[x].min()
        y_span = plot_df[y].max() - plot_df[y].min()
        x_radius = (x_span * 0.016) if x_span else 0.001
        y_radius = (y_span * 0.024) if y_span else 0.001
        angles = 2 * np.pi * duplicate_index / duplicate_count
        plot_df.loc[duplicate_mask, "_Plot X"] = plot_df.loc[duplicate_mask, x] + np.cos(angles[duplicate_mask]) * x_radius
        plot_df.loc[duplicate_mask, "_Plot Y"] = plot_df.loc[duplicate_mask, y] + np.sin(angles[duplicate_mask]) * y_radius

    size_col = None
    if size in plot_df.columns:
        plot_df["_Scatter Size"] = pd.to_numeric(plot_df[size], errors="coerce")
        if plot_df["_Scatter Size"].notna().any():
            fallback_size = plot_df["_Scatter Size"].median()
            plot_df["_Scatter Size"] = plot_df["_Scatter Size"].fillna(fallback_size).clip(lower=0)
            size_col = "_Scatter Size"
    hover_cols = [plot_df["Team"], plot_df[x], plot_df[y]]
    if size in plot_df.columns:
        hover_cols.append(plot_df[size])
    customdata = np.stack(hover_cols, axis=-1)
    hover = f"%{{customdata[0]}}<br>{x}: {charting.hover_value('customdata[1]', x)}<br>{y}: {charting.hover_value('customdata[2]', y)}"
    if size in plot_df.columns:
        hover += f"<br>{size}: {charting.hover_value('customdata[3]', size)}"

    hover_marker_size = pd.Series(8, index=plot_df.index)

    fig = go.Figure(
        go.Scatter(
            x=plot_df["_Plot X"],
            y=plot_df["_Plot Y"],
            mode="markers",
            customdata=customdata,
            hovertemplate=hover + "<extra></extra>",
            marker=dict(
                size=hover_marker_size.tolist(),
                color="rgba(0, 0, 0, 0)",
                line=dict(width=0, color="rgba(0, 0, 0, 0)"),
                opacity=0,
            ),
        )
    )
    badge_df = plot_df[["Team", x, y, "_Plot X", "_Plot Y"]].copy()
    badge_df[x] = pd.to_numeric(badge_df[x], errors="coerce")
    badge_df[y] = pd.to_numeric(badge_df[y], errors="coerce")

    if size in plot_df.columns:
        badge_df[size] = pd.to_numeric(plot_df[size], errors="coerce")
        valid_sizes = badge_df[size].dropna()
        if not valid_sizes.empty and valid_sizes.max() != valid_sizes.min():
            size_range = valid_sizes.max() - valid_sizes.min()
            badge_df["_Badge Scale"] = 0.75 + ((badge_df[size] - valid_sizes.min()) / size_range).clip(0, 1) * 0.85
            badge_df["_Badge Scale"] = badge_df["_Badge Scale"].fillna(1.0)
        else:
            badge_df["_Badge Scale"] = 1.15
    else:
        badge_df["_Badge Scale"] = 1.0

    badge_df = badge_df.dropna(subset=["_Plot X", "_Plot Y"])

    x_span = badge_df["_Plot X"].max() - badge_df["_Plot X"].min()
    y_span = badge_df["_Plot Y"].max() - badge_df["_Plot Y"].min()

    base_badge_width = x_span * 0.045 if x_span else 1
    base_badge_height = y_span * 0.070 if y_span else 1

    for _, row in badge_df.iterrows():
        badge_uri = team_badges.badge_data_uri(row["Team"])
        if not badge_uri:
            continue
        badge_scale = float(row.get("_Badge Scale", 1.0))

        fig.add_layout_image(
            dict(
                source=badge_uri,
                x=row["_Plot X"],
                y=row["_Plot Y"],
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="middle",
                sizex=base_badge_width * badge_scale,
                sizey=base_badge_height * badge_scale,
                sizing="contain",
                layer="above",
            )
        )
    fig.update_layout(height=560, showlegend=False, hovermode="closest", hoverdistance=4)
    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)
    charting.format_xaxis(fig, x)
    charting.format_yaxis(fig, y)
    return polish_figure(fig, title)


def match_rows_for_team(matches: pd.DataFrame, team_name: str) -> pd.DataFrame:
    if matches.empty or not {"Home", "Away", "Home Goals", "Away Goals"}.issubset(matches.columns):
        return pd.DataFrame()
    home = matches["Home"].astype(str) == str(team_name)
    away = matches["Away"].astype(str) == str(team_name)
    team_matches = matches[home | away].copy()
    if team_matches.empty:
        return team_matches

    is_home = team_matches["Home"].astype(str) == str(team_name)
    team_matches["Opponent"] = np.where(is_home, team_matches["Away"], team_matches["Home"])
    team_matches["Goals For"] = np.where(is_home, team_matches["Home Goals"], team_matches["Away Goals"])
    team_matches["Goals Against"] = np.where(is_home, team_matches["Away Goals"], team_matches["Home Goals"])
    team_matches["Goal Difference"] = team_matches["Goals For"] - team_matches["Goals Against"]
    team_matches["Team Result"] = np.select(
        [
            team_matches["Goal Difference"] > 0,
            team_matches["Goal Difference"] < 0,
        ],
        ["Win", "Loss"],
        default="Draw",
    )
    team_matches["Points"] = np.select(
        [
            team_matches["Team Result"] == "Win",
            team_matches["Team Result"] == "Draw",
        ],
        [3, 1],
        default=0,
    )
    if "Date" in team_matches:
        team_matches["Date"] = pd.to_datetime(team_matches["Date"], errors="coerce")
        team_matches = team_matches.sort_values("Date")
    team_matches["Cumulative Points"] = team_matches["Points"].cumsum()
    team_matches["Cumulative Goal Difference"] = team_matches["Goal Difference"].cumsum()
    team_matches["Rolling Points"] = team_matches["Points"].rolling(5, min_periods=1).mean().round(2)
    return team_matches


def match_trend_chart(team_matches: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    x = team_matches["Date"] if "Date" in team_matches else team_matches.index + 1
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
    fig.update_layout(height=500, yaxis_title="Goals", xaxis_title="Match date")
    fig.update_yaxes(tickformat=".0f")
    return polish_figure(fig, title)


def momentum_chart(team_matches: pd.DataFrame, title: str) -> go.Figure:
    plot_df = team_matches.copy().reset_index(drop=True)
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


def load_action_counts(season: str | None = None) -> pd.DataFrame:
    if hasattr(data, "load_team_action_counts"):
        try:
            actions = data.load_team_action_counts(season=season).copy()
        except Exception as exc:
            st.warning(f"Could not load event action counts: {exc}")
            return pd.DataFrame(columns=["Team", "Action", "Actions"])
        if not actions.empty:
            actions["Actions"] = pd.to_numeric(actions["Actions"], errors="coerce").fillna(0)
        return actions
    return pd.DataFrame(columns=["Team", "Action", "Actions"])


def filter_actions(actions: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    if actions.empty or "Action" not in actions:
        return pd.DataFrame(columns=actions.columns)
    pattern = "|".join(re.escape(keyword) for keyword in keywords)
    return actions[actions["Action"].astype(str).str.contains(pattern, case=False, na=False)].copy()


def action_summary(actions: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    filtered = filter_actions(actions, keywords)
    if filtered.empty:
        return pd.DataFrame(columns=["Team", "Actions"])
    return filtered.groupby("Team", as_index=False)["Actions"].sum().sort_values("Actions", ascending=False)


def team_action_breakdown(actions: pd.DataFrame, team_name: str, keywords: list[str]) -> pd.DataFrame:
    filtered = filter_actions(actions, keywords)
    if filtered.empty:
        return pd.DataFrame(columns=["Action", "Actions"])
    return (
        filtered[filtered["Team"].astype(str) == str(team_name)]
        .groupby("Action", as_index=False)["Actions"]
        .sum()
        .sort_values("Actions", ascending=False)
    )


def action_bar(summary: pd.DataFrame, selected: str | None, title: str) -> go.Figure:
    if summary.empty:
        return go.Figure()
    return ranked_bar(summary.rename(columns={"Actions": "Action Count"}), "Action Count", selected=selected, title=title)


def action_donut(breakdown: pd.DataFrame, title: str) -> go.Figure:
    if breakdown.empty:
        return go.Figure()
    plot_df = breakdown.sort_values("Actions", ascending=False).copy()
    plot_df["_Label"] = plot_df["Action"].apply(lambda value: charting.wrap_label(value, width=18, max_lines=2))
    fig = go.Figure(
        go.Pie(
            labels=plot_df["_Label"],
            values=plot_df["Actions"],
            hole=0.58,
            marker=dict(colors=[RED, BLUE, DARK, GREY, "#e6b400", "#b42318"]),
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


def player_contribution(players: pd.DataFrame, team_name: str, metric: str, top_n: int = 12) -> pd.DataFrame:
    if players.empty or metric not in players:
        return pd.DataFrame()
    team_players = players[players["Team"].astype(str) == str(team_name)].copy()
    if team_players.empty:
        return team_players
    team_players[metric] = pd.to_numeric(team_players[metric], errors="coerce")
    return team_players.sort_values(metric, ascending=False).head(top_n)


def player_contribution_bar(players: pd.DataFrame, metric: str, title: str) -> go.Figure:
    if players.empty:
        return go.Figure()
    plot_df = players.sort_values(metric, ascending=True)
    plot_df["_Label"] = plot_df["Player"].apply(lambda value: charting.wrap_label(value, width=18, max_lines=2))
    plot_df["_Text"] = charting.outside_bar_text(plot_df[metric], metric)
    fig = go.Figure(
        go.Bar(
            x=plot_df[metric],
            y=plot_df["_Label"],
            orientation="h",
            marker_color=RED,
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
    )
    charting.format_xaxis(fig, metric)
    fig = polish_figure(fig, title)
    fig.update_layout(margin=dict(l=36, r=76, t=68, b=54))
    return fig


def cluster_passing_profiles(teams: pd.DataFrame) -> pd.DataFrame:
    out = teams.copy()
    pass_pct = percentile(out["Pass %"]) if "Pass %" in out else pd.Series(50, index=out.index)
    progression = pd.concat(
        [
            percentile(out["Passes to Final 3rd /90"]) if "Passes to Final 3rd /90" in out else pd.Series(50, index=out.index),
            percentile(out["Bypassed Opponents /90"]) if "Bypassed Opponents /90" in out else pd.Series(50, index=out.index),
        ],
        axis=1,
    ).mean(axis=1)

    out["Pass Security Percentile"] = pass_pct
    out["Progression Percentile"] = progression.round(1)

    conditions = [
        (out["Pass Security Percentile"] >= 55) & (out["Progression Percentile"] >= 55),
        (out["Pass Security Percentile"] < 55) & (out["Progression Percentile"] >= 55),
        (out["Pass Security Percentile"] >= 55) & (out["Progression Percentile"] < 55),
    ]
    labels = ["Controlled Progressors", "Direct Progressors", "Secure Circulators"]
    out["Cluster"] = np.select(conditions, labels, default="Lower Volume")
    return out


def cluster_chart(clustered: pd.DataFrame, selected: str | None) -> go.Figure:
    clustered = clustered.copy()
    clustered["_Selected Size"] = np.where(clustered["Team"].astype(str) == str(selected), 18, 11)
    clustered["_Text"] = charting.selected_text(clustered["Team"], selected)
    fig = px.scatter(
        clustered,
        x="Pass Security Percentile",
        y="Progression Percentile",
        color="Cluster",
        text="_Text",
        size="_Selected Size",
        size_max=18,
        hover_data=["Pass %", "Passes to Final 3rd /90", "Bypassed Opponents /90"],
    )
    fig.add_hline(y=55, line_dash="dash", line_color=LIGHT_GREY)
    fig.add_vline(x=55, line_dash="dash", line_color=LIGHT_GREY)
    fig.update_traces(textposition="top center", marker=dict(line=dict(width=1.2, color="#ffffff"), opacity=0.9))
    fig.update_layout(height=560, showlegend=True)
    fig.update_xaxes(tickformat=".0f", range=[0, 105])
    fig.update_yaxes(tickformat=".0f", range=[0, 105])
    return polish_figure(fig, "Pass profile clusters")
