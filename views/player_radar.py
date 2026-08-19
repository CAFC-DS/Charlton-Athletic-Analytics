# =============================================================================
# PLAYER RADAR - configurable single-player and comparison radars
# =============================================================================
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charting
from utils import player_analysis as pa
from utils import ui


CATEGORY_ORDER = ["Attacking", "Passing", "Progression", "Defending", "Possession", "Goalkeeping"]
CATEGORY_LABELS = {"Defending": "Defensive"}
RADAR_COLORS = [ui.CHARLTON_RED, ui.CHARLTON_BLACK, "#16a34a", "#f59e0b"]


def _radar_css() -> None:
    st.markdown(
        """
        <style>
        .pr-control-note {
            color: #667085;
            font-size: 0.84rem;
            line-height: 1.4;
            margin: 4px 0 14px;
        }

        .pr-summary-grid {
            display: grid;
            gap: 10px;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            margin: 6px 0 16px;
        }

        .pr-summary-card {
            background: #ffffff;
            border: 1px solid #e6edf5;
            border-radius: 8px;
            min-height: 72px;
            padding: 10px 12px;
        }

        .pr-summary-label {
            color: #667085;
            font-size: 0.62rem;
            font-weight: 850;
            letter-spacing: 0.04em;
            line-height: 1.18;
            text-transform: uppercase;
        }

        .pr-summary-value {
            color: #111111;
            font-size: 1.05rem;
            font-weight: 820;
            line-height: 1.2;
            margin-top: 5px;
            overflow-wrap: anywhere;
        }

        .pr-radar-note {
            color: #667085;
            font-size: 0.82rem;
            line-height: 1.35;
            margin: 0 0 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sort_key(row: pd.Series) -> tuple[str, str, str]:
    last_name = row.get("Last Name")
    if pd.notna(last_name) and str(last_name).strip():
        sort_last = str(last_name).strip().casefold()
    else:
        player = str(row.get("Player", "")).strip()
        sort_last = player.split()[-1].casefold() if player else ""
    return (sort_last, str(row.get("Player", "")).casefold(), str(row.get("Team", "")).casefold())


def _player_options(players: pd.DataFrame) -> list[int]:
    sortable = players.copy()
    sortable["_sort"] = sortable.apply(_sort_key, axis=1)
    return sortable.sort_values("_sort", kind="mergesort").index.tolist()


def _player_label(players: pd.DataFrame, idx: int) -> str:
    row = players.loc[idx]
    return f"{row.get('Player', 'Unknown')} | {row.get('Team', 'Unknown')} | {row.get('_Position Display', 'Unknown')}"


def _preferred_index(players: pd.DataFrame, options: list[int]) -> int:
    selected = st.session_state.get("selected_player")
    if selected:
        found = players.index[players["Player"].astype(str) == str(selected)].tolist()
        if found and found[0] in options:
            return options.index(found[0])
    return 0


def _available_metrics_by_category(players: pd.DataFrame) -> dict[str, list[str]]:
    grouped = {category: [] for category in CATEGORY_ORDER}
    for metric, (category, _, _) in pa.PROFILE_METRIC_META.items():
        if metric not in players or category not in grouped:
            continue
        values = pd.to_numeric(players[metric], errors="coerce")
        if values.notna().sum() >= 2:
            grouped[category].append(metric)
    return grouped


def _metric_display(metric: str) -> str:
    _, _, label = pa.PROFILE_METRIC_META.get(metric, ("General", True, metric))
    return f"{label} ({metric})" if label != metric else metric


def _default_metrics(players: pd.DataFrame, selected_indices: list[int]) -> list[str]:
    if not selected_indices:
        return []
    role = str(players.loc[selected_indices[0]].get("Role Group", "Outfield"))
    defaults = pa.profile_metrics_for_role(players, role)
    return [metric for metric in defaults if metric in pa.PROFILE_METRIC_META]


def _selected_metric_controls(players: pd.DataFrame, selected_indices: list[int]) -> list[str]:
    by_category = _available_metrics_by_category(players)
    defaults = _default_metrics(players, selected_indices)
    selected_metrics: list[str] = []
    st.markdown('<div class="pr-control-note">Choose the axes to include. Six to ten metrics usually gives the cleanest radar.</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    for index, category in enumerate(CATEGORY_ORDER):
        options = by_category.get(category, [])
        if not options:
            continue
        label = CATEGORY_LABELS.get(category, category)
        default = [metric for metric in defaults if metric in options]
        with cols[index % 3]:
            chosen = st.multiselect(
                label,
                options,
                default=default,
                format_func=_metric_display,
                key=f"player_radar_metric_{category}",
            )
        selected_metrics.extend(chosen)

    return list(dict.fromkeys(selected_metrics))


def _percentile_rows(players: pd.DataFrame, selected_indices: list[int], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        category, higher_is_better, label = pa.PROFILE_METRIC_META.get(metric, ("General", True, metric))
        values = pd.to_numeric(players[metric], errors="coerce")
        percentiles = pa.percentile(values, higher_is_better=higher_is_better)
        ranks = values.rank(method="min", ascending=not higher_is_better)
        for idx in selected_indices:
            row = players.loc[idx]
            selection_label = f"{row.get('Player', 'Unknown')} | {row.get('Team', 'Unknown')}"
            rows.append(
                {
                    "Selection": selection_label,
                    "Player": row.get("Player", "Unknown"),
                    "Team": row.get("Team", "Unknown"),
                    "Position": row.get("_Position Display", "Unknown"),
                    "Category": CATEGORY_LABELS.get(category, category),
                    "Metric": metric,
                    "Radar Label": label,
                    "Value": row.get(metric),
                    "Display Value": pa.metric_value(row.get(metric), metric),
                    "Percentile": float(percentiles.loc[idx]) if idx in percentiles.index and pd.notna(percentiles.loc[idx]) else pd.NA,
                    "Rank": int(ranks.loc[idx]) if idx in ranks.index and pd.notna(ranks.loc[idx]) else pd.NA,
                    "Higher Is Better": higher_is_better,
                }
            )
    return pd.DataFrame(rows)


def _radar_chart(rows: pd.DataFrame, selected_labels: list[str], metrics: list[str], mode: str) -> go.Figure:
    fig = go.Figure()
    labels = [
        f"<b>{charting.wrap_label(pa.PROFILE_METRIC_META.get(metric, ('', True, metric))[2], width=13, max_lines=2)}</b>"
        for metric in metrics
    ]
    closed_labels = labels + labels[:1]

    for index, player in enumerate(selected_labels):
        player_rows = rows[rows["Selection"].astype(str) == str(player)].set_index("Metric").reindex(metrics).reset_index()
        values = pd.to_numeric(player_rows["Percentile"], errors="coerce").fillna(0).tolist()
        closed_values = values + values[:1]
        customdata = [
            [
                metric,
                display,
                None if pd.isna(percentile) else float(percentile),
                rank,
                "Higher" if higher else "Lower",
            ]
            for metric, display, percentile, rank, higher in zip(
                player_rows["Metric"],
                player_rows["Display Value"],
                player_rows["Percentile"],
                player_rows["Rank"],
                player_rows["Higher Is Better"],
            )
        ]
        closed_customdata = customdata + customdata[:1]
        color = RADAR_COLORS[index % len(RADAR_COLORS)]
        fig.add_trace(
            go.Scatterpolar(
                r=closed_values,
                theta=closed_labels,
                mode="lines+markers",
                name=str(player),
                line=dict(color=color, width=3 if index == 0 else 2.4),
                fill="toself" if len(selected_labels) == 1 else None,
                fillcolor="rgba(195, 0, 23, 0.16)" if len(selected_labels) == 1 else None,
                marker=dict(size=8, color=color, line=dict(color="#ffffff", width=1.2)),
                customdata=closed_customdata,
                hovertemplate=(
                    "%{fullData.name}"
                    "<br>%{customdata[0]}"
                    "<br>Value: %{customdata[1]}"
                    "<br>Percentile: %{customdata[2]:.0f}"
                    "<br>Rank: %{customdata[3]}"
                    "<br>%{customdata[4]} is better<extra></extra>"
                ),
            )
        )

    title = "Single-Player Radar" if mode == "Single player" else "Player Radar Comparison"
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Arial, sans-serif", color=pa.DARK, size=12),
        height=700,
        margin=dict(l=72, r=72, t=88, b=150),
        polar=dict(
            bgcolor="#ffffff",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80],
                tickfont=dict(size=10, color=pa.GREY),
                gridcolor="#e9edf3",
                linecolor="#e9edf3",
                angle=90,
            ),
            angularaxis=dict(
                rotation=90,
                direction="clockwise",
                tickfont=dict(size=12, color=pa.DARK),
                gridcolor="#eef2f6",
                linecolor="#eef2f6",
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.13,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
            itemwidth=30,
        ),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=pa.LIGHT_GREY, font_size=13, font_color=pa.DARK),
        showlegend=len(selected_labels) > 1,
    )
    fig = pa.polish_figure(fig, title)
    fig.update_layout(
        height=700,
        margin=dict(l=72, r=72, t=88, b=150),
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


def _summary_cards(selected_rows: pd.DataFrame, metrics: list[str]) -> None:
    players_text = f"{len(selected_rows)} player" if len(selected_rows) == 1 else f"{len(selected_rows)} players"
    roles = selected_rows["Role Group"].dropna().astype(str).drop_duplicates().tolist() if "Role Group" in selected_rows else []
    cards = [
        ("Mode", players_text),
        ("Radar axes", len(metrics)),
        ("Primary role", roles[0] if roles else "Unknown"),
        ("Minutes range", f"{pa.metric_value(selected_rows['Minutes'].min(), 'Minutes')} - {pa.metric_value(selected_rows['Minutes'].max(), 'Minutes')}" if "Minutes" in selected_rows else "N/A"),
    ]
    html = "".join(
        '<div class="pr-summary-card">'
        f'<div class="pr-summary-label">{ui.esc(label)}</div>'
        f'<div class="pr-summary-value">{ui.esc(value)}</div>'
        "</div>"
        for label, value in cards
    )
    st.markdown(f'<div class="pr-summary-grid">{html}</div>', unsafe_allow_html=True)


pa.page_header(
    "Player Radar",
    "Build a percentile radar for one player or compare up to four players across selected metric categories.",
)
_radar_css()

season = pa.select_season(key="player_radar_season")
players = pa.add_position_groups(pa.load_player_data(season))
if players.empty:
    st.warning("No players are available for the selected season.")
    st.stop()

options = _player_options(players)

pa.section_heading("Radar controls")
mode = st.radio("Radar type", ["Single player", "Player comparison"], horizontal=True, key="player_radar_mode")
control_cols = st.columns([2.6, 1])
if mode == "Single player":
    default_index = _preferred_index(players, options)
    selected_index = control_cols[0].selectbox(
        "Player",
        options,
        index=default_index,
        format_func=lambda idx: _player_label(players, idx),
        key="player_radar_single_player",
    )
    selected_indices = [selected_index]
else:
    default_start = _preferred_index(players, options)
    default_options = [options[default_start], *[idx for idx in options if idx != options[default_start]][:1]]
    selected_indices = control_cols[0].multiselect(
        "Players",
        options,
        default=default_options[:2],
        max_selections=4,
        format_func=lambda idx: _player_label(players, idx),
        key="player_radar_compare_players",
    )

min_minutes = control_cols[1].number_input("Minimum minutes", min_value=0, value=0, step=250)
if not selected_indices:
    st.info("Select at least one player to build the radar.")
    st.stop()
if mode == "Player comparison" and len(selected_indices) < 2:
    st.info("Select at least two players to build a comparison radar.")
    st.stop()

st.session_state["selected_player"] = str(players.loc[selected_indices[0], "Player"])
filtered_players = players[pd.to_numeric(players["Minutes"], errors="coerce").fillna(0) >= min_minutes].copy()
selected_rows = players.loc[selected_indices].copy()
filtered_players = pd.concat([filtered_players, selected_rows])
filtered_players = filtered_players.loc[~filtered_players.index.duplicated(keep="first")].copy()

pa.section_heading("Metric categories")
selected_metrics = _selected_metric_controls(filtered_players, selected_indices)
if len(selected_metrics) < 3:
    st.info("Select at least three metrics across the category dropdowns to draw a radar.")
    st.stop()
if len(selected_metrics) > 12:
    st.warning("More than 12 radar axes can become hard to read. Consider removing a few metrics for presentation use.")

radar_rows = _percentile_rows(filtered_players, selected_indices, selected_metrics)
selected_player_labels = [f"{row.get('Player', 'Unknown')} | {row.get('Team', 'Unknown')}" for _, row in selected_rows.iterrows()]
_summary_cards(selected_rows, selected_metrics)

pa.section_heading("Radar output")
st.markdown(
    '<div class="pr-radar-note">Percentiles are calculated across the selected season dataset after the minimum-minutes filter. Higher percentile is always better; only lower-is-better raw metrics are inverted before scoring.</div>',
    unsafe_allow_html=True,
)
st.plotly_chart(
    _radar_chart(radar_rows, selected_player_labels, selected_metrics, mode),
    use_container_width=True,
    config={"displayModeBar": False, "responsive": True},
    key=f"player_radar_chart_{pa.safe_key(mode)}_{len(selected_metrics)}_{'_'.join(pa.safe_key(name) for name in selected_player_labels)}",
)
