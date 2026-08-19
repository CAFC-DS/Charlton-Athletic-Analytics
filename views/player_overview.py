# =============================================================================
# PLAYER OVERVIEW - player analysis hub and selected-player snapshot
# =============================================================================
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charting
from utils import player_analysis as pa
from utils import ui


GREEN = "#16a34a"
AMBER = "#f59e0b"
GREY = "#98a2b3"


def _overview_css() -> None:
    st.markdown(
        """
        <style>
        .po-card-grid {
            display: grid;
            gap: 10px;
            grid-template-columns: repeat(auto-fit, minmax(142px, 1fr));
            margin: 10px 0 18px;
        }

        .po-card {
            background: #ffffff;
            border: 1px solid #e6edf5;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-height: 78px;
            padding: 10px 12px;
        }

        .po-card-label {
            color: #667085;
            font-size: 0.62rem;
            font-weight: 850;
            letter-spacing: 0.04em;
            line-height: 1.18;
            text-transform: uppercase;
        }

        .po-card-value {
            color: #111111;
            font-size: 1.02rem;
            font-weight: 800;
            line-height: 1.2;
            margin-top: 5px;
            overflow-wrap: anywhere;
        }

        .po-card-sub {
            color: #667085;
            font-size: 0.72rem;
            line-height: 1.25;
            margin-top: 4px;
            overflow-wrap: anywhere;
        }

        .po-link-grid {
            display: grid;
            gap: 10px;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            margin: 2px 0 22px;
        }

        .po-link-grid [data-testid="stPageLink"] a {
            border: 1px solid #d8dde6;
            border-radius: 8px;
            padding: 10px 12px;
        }

        .po-chart-legend {
            align-items: center;
            color: #475467;
            display: flex;
            flex-wrap: wrap;
            gap: 10px 16px;
            font-size: 0.82rem;
            line-height: 1.25;
            margin: 2px 0 8px;
        }

        .po-legend-item {
            align-items: center;
            display: inline-flex;
            gap: 6px;
        }

        .po-legend-swatch {
            border-radius: 2px;
            display: inline-block;
            height: 10px;
            width: 18px;
        }

        .po-legend-line {
            border-top: 2px dashed #f59e0b;
            display: inline-block;
            height: 0;
            width: 22px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _clean_text(value: object, fallback: str = "N/A") -> str:
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text and text.lower() != "nan" else fallback


def _snapshot_card(label: str, value: object, sub: object | None = None) -> str:
    sub_html = f'<div class="po-card-sub">{ui.esc(_clean_text(sub))}</div>' if sub is not None else ""
    return (
        '<div class="po-card">'
        f'<div class="po-card-label">{ui.esc(label)}</div>'
        f'<div class="po-card-value">{ui.esc(_clean_text(value))}</div>'
        f"{sub_html}"
        "</div>"
    )


def _snapshot_grid(row: pd.Series, metrics: list[str]) -> None:
    metric_cards = [
        _snapshot_card(metric, pa.metric_value(row.get(metric), metric))
        for metric in metrics[:2]
    ]
    cards = [
        _snapshot_card("Team", row.get("Team", "Unknown")),
        _snapshot_card("Position", row.get("_Position Display", "Unknown")),
        _snapshot_card("Minutes", pa.metric_value(row.get("Minutes"), "Minutes")),
        *metric_cards,
    ]
    st.markdown(f'<div class="po-card-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def _ranking_summary(row: pd.Series, metric: str, players: pd.DataFrame) -> None:
    value = pd.to_numeric(row.get(metric), errors="coerce")
    average = pd.to_numeric(players[metric], errors="coerce").mean()
    rank = row.get(f"{metric} Rank")
    percentile = row.get(f"{metric} Percentile")
    cards = [
        _snapshot_card("Rank", f"{rank} / {len(players)}"),
        _snapshot_card("Player value", pa.metric_value(value, metric)),
        _snapshot_card("League average", pa.metric_value(average, metric)),
        _snapshot_card("Percentile", charting.metric_text(percentile, "Percentile")),
    ]
    st.markdown(f'<div class="po-card-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def _ranking_legend(metric: str, players: pd.DataFrame) -> None:
    average = pd.to_numeric(players[metric], errors="coerce").mean()
    html = (
        '<div class="po-chart-legend">'
        '<span class="po-legend-item"><span class="po-legend-swatch" style="background:#c30017"></span>Selected player</span>'
        '<span class="po-legend-item"><span class="po-legend-swatch" style="background:#16a34a"></span>At or above average</span>'
        '<span class="po-legend-item"><span class="po-legend-swatch" style="background:#98a2b3"></span>Below average</span>'
        f'<span class="po-legend-item"><span class="po-legend-line"></span>Average {ui.esc(pa.metric_value(average, metric))}</span>'
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _ranking_chart(players: pd.DataFrame, metric: str, selected: str, top_n: int = 18) -> go.Figure:
    full_df = players.copy()
    full_df[metric] = pd.to_numeric(full_df[metric], errors="coerce")
    full_df = full_df.dropna(subset=[metric]).copy()
    average = full_df[metric].mean()

    plot_df = full_df.sort_values(metric, ascending=True).tail(top_n).copy()
    selected_row = full_df[full_df["Player"].astype(str) == str(selected)]
    if not selected_row.empty and selected not in plot_df["Player"].astype(str).tolist():
        plot_df = pd.concat([plot_df.iloc[1:], selected_row], ignore_index=True)
        plot_df = plot_df.sort_values(metric, ascending=True)

    plot_df["_Label"] = plot_df["Player"].apply(lambda value: charting.wrap_label(value, width=19, max_lines=2))
    plot_df["_Text"] = charting.outside_bar_text(plot_df[metric], metric)
    plot_df["_Colour"] = [
        ui.CHARLTON_RED
        if str(player) == str(selected)
        else GREEN
        if value >= average
        else GREY
        for player, value in zip(plot_df["Player"], plot_df[metric])
    ]

    fig = go.Figure(
        go.Bar(
            x=plot_df[metric],
            y=plot_df["_Label"],
            orientation="h",
            marker=dict(color=plot_df["_Colour"], line=dict(color="#ffffff", width=1)),
            text=plot_df["_Text"],
            textposition="outside",
            cliponaxis=False,
            customdata=pd.concat(
                [
                    plot_df["Player"],
                    plot_df["Team"] if "Team" in plot_df else pd.Series("", index=plot_df.index),
                    plot_df["_Position Display"] if "_Position Display" in plot_df else pd.Series("", index=plot_df.index),
                    plot_df["_Text"],
                ],
                axis=1,
            ).to_numpy(),
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>%{customdata[2]}<br>"
            + metric
            + ": %{customdata[3]}<extra></extra>",
        )
    )
    selected_plot = plot_df[plot_df["Player"].astype(str) == str(selected)]
    if not selected_plot.empty:
        fig.add_trace(
            go.Scatter(
                x=selected_plot[metric],
                y=selected_plot["_Label"],
                mode="markers",
                marker=dict(symbol="diamond", size=13, color=ui.CHARLTON_RED, line=dict(color="#ffffff", width=1.5)),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_vline(
        x=average,
        line_width=2,
        line_dash="dash",
        line_color=AMBER,
    )

    max_x = max(plot_df[metric].max(), average)
    min_x = min(plot_df[metric].min(), average)
    if min_x >= 0:
        fig.update_xaxes(range=[0, max_x * 1.18 if max_x else 1])
    charting.format_xaxis(fig, metric)
    fig.update_layout(
        height=charting.horizontal_bar_height(len(plot_df), row_height=34),
        margin=dict(l=40, r=104, t=82, b=58),
        showlegend=False,
        xaxis_title=metric,
        yaxis_title="",
    )
    fig = pa.polish_figure(fig, f"{metric} ranking with league average")
    return fig


pa.page_header(
    "Player Overview",
    "Use this page as the Player Analysis hub: select a player, review the supported metrics and jump into the main visual pages.",
)
_overview_css()

season = pa.select_season(key="player_overview_season")
players = pa.add_metric_ranks(pa.load_player_data(season))
if players.empty:
    st.warning("No players are available for the selected season.")
    st.stop()

pa.section_heading("Selected player snapshot")
st.caption("Use the dropdown to set the player context for the summary cards below.")
player_name = pa.player_selector(players, key="player_overview_player")
row = pa.player_row(players, player_name)
metrics = pa.metric_columns(players)
_snapshot_grid(row, metrics)

pa.section_heading("Supported player analysis pages")
pa.analysis_card_grid(
    [
        {"title": "Player Profiles", "body": "Role-aware profile, standard stats, radar shape, detailed breakdowns and similar players."},
        {"title": "Player Radar", "body": "Focused percentile radar for the selected player across the core comparison metrics."},
        {"title": "Comparison", "body": "Side-by-side percentile comparison for two to four players."},
        {"title": "Scatter and Rankings", "body": "Distribution plots and leaderboards for spotting outliers and peer groups."},
    ]
)

link_cols = st.columns(4)
with link_cols[0]:
    st.page_link("views/player_profiles.py", label="Open Player Profiles")
with link_cols[1]:
    st.page_link("views/player_radar.py", label="Open Player Radar")
with link_cols[2]:
    st.page_link("views/player_comparison.py", label="Open Player Comparison")
with link_cols[3]:
    st.page_link("views/player_scatter.py", label="Open Player Scatter")

pa.section_heading("Selected player ranking")
rank_cols = st.columns([2.4, 1])
metric = rank_cols[0].selectbox("Ranking metric", metrics, key="player_overview_metric")
top_n = rank_cols[1].slider("Players shown", min_value=10, max_value=30, value=18, step=2)
row = pa.player_row(players, player_name)
_ranking_summary(row, metric, players)
_ranking_legend(metric, players)
st.plotly_chart(
    _ranking_chart(players, metric, selected=player_name, top_n=top_n),
    use_container_width=True,
    config={"displayModeBar": False, "responsive": True},
    key=f"player_overview_ranking_{pa.safe_key(player_name)}_{pa.safe_key(metric)}_{top_n}",
)
