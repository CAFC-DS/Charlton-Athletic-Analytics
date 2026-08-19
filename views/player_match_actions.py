# =============================================================================
# PLAYER MATCH ACTIONS - real selected-match player event summary
# =============================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charting, data, match_analysis as ma, ui


def _player_actions_css() -> None:
    st.markdown(
        """
        <style>
        .pma-context-grid,
        .pma-spotlight-grid {
            display: grid;
            gap: 12px;
            margin: 8px 0 18px;
        }

        .pma-context-grid {
            grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
        }

        .pma-spotlight-grid {
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        }

        .pma-card,
        .pma-spotlight-card,
        .pma-control-note {
            background: var(--ss-panel);
            border: 1px solid var(--ss-border);
            border-radius: 10px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }

        .pma-card {
            border-top: 3px solid var(--ss-accent);
            min-height: 104px;
            padding: 14px 16px;
        }

        .pma-card-dark {
            background:
                radial-gradient(circle at 94% 14%, rgba(255, 255, 255, 0.13), transparent 24%),
                linear-gradient(135deg, #111111 0%, #271115 60%, #9c0214 148%);
            border-color: rgba(255, 255, 255, 0.14);
            color: #ffffff;
        }

        .pma-card-dark .pma-label,
        .pma-card-dark .pma-helper {
            color: rgba(255, 255, 255, 0.72);
        }

        .pma-card-dark .pma-value {
            color: #ffffff;
        }

        .pma-label {
            color: var(--ss-muted);
            font-size: 0.74rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            line-height: 1.25;
            margin-bottom: 10px;
            text-transform: uppercase;
        }

        .pma-value {
            color: var(--ss-ink);
            display: block;
            font-size: clamp(1.45rem, 1.9vw, 1.85rem);
            font-weight: 500;
            letter-spacing: -0.04em;
            line-height: 1.05;
            overflow-wrap: anywhere;
        }

        .pma-value-text {
            font-size: clamp(0.9rem, 1.08vw, 1.08rem);
            font-weight: 850;
            letter-spacing: -0.02em;
            line-height: 1.22;
        }

        .pma-helper {
            color: var(--ss-muted);
            font-size: 0.82rem;
            font-weight: 650;
            line-height: 1.35;
            margin-top: 8px;
        }

        .pma-control-note {
            background: #f8fafc;
            border-left: 4px solid var(--ss-accent);
            color: var(--ss-muted);
            font-size: 0.88rem;
            line-height: 1.45;
            margin: 4px 0 14px;
            padding: 12px 14px;
        }

        .pma-spotlight-card {
            min-height: 126px;
            overflow: hidden;
            padding: 0;
        }

        .pma-spotlight-top {
            background: #111111;
            color: #ffffff;
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            padding: 9px 13px;
            text-transform: uppercase;
        }

        .pma-spotlight-body {
            padding: 13px 14px 14px;
        }

        .pma-spotlight-name {
            color: var(--ss-ink);
            font-size: 1.03rem;
            font-weight: 900;
            line-height: 1.16;
            margin-bottom: 10px;
            overflow-wrap: anywhere;
        }

        .pma-spotlight-value {
            color: var(--ss-accent);
            font-size: 1.26rem;
            font-weight: 900;
            letter-spacing: -0.03em;
            line-height: 1.05;
        }

        .pma-spotlight-meta {
            color: var(--ss-muted);
            font-size: 0.83rem;
            font-weight: 650;
            line-height: 1.35;
            margin-top: 8px;
        }

        .pma-layout-note {
            color: var(--ss-muted);
            font-size: 0.84rem;
            line-height: 1.35;
            margin: -4px 0 10px;
        }

        @media (max-width: 760px) {
            .pma-card {
                min-height: 92px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _metric_label(metric: str) -> str:
    labels = {
        "Actions": "Actions",
        "Positive PXT": "Positive PXT",
        "Team xT": "Team xT",
        "Shot xG": "Shot xG",
        "Action Share": "Action Share",
    }
    return labels.get(metric, str(metric).title())


def _signed_metric_text(value: object, metric: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    prefix = "+" if number > 0 else ""
    return f"{prefix}{charting.metric_text(number, metric)}"


def _performance_scores(values: pd.Series, average_value: float) -> pd.Series:
    if values.empty:
        return pd.Series(dtype="float64")
    values = pd.to_numeric(values, errors="coerce").fillna(0)
    if np.isfinite(average_value) and average_value > 0:
        return (values / (average_value * 2)).clip(lower=0, upper=1)
    value_range = float(values.max() - values.min())
    if value_range > 0:
        return ((values - values.min()) / value_range).clip(lower=0, upper=1)
    return pd.Series(0.5, index=values.index, dtype="float64")


def _mode(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    clean = clean[clean.str.len() > 0]
    if clean.empty:
        return "Unknown"
    return str(clean.value_counts().index[0])


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["Action Type"] = out["Action Type"].fillna("Unknown").astype(str) if "Action Type" in out else "Unknown"
    out["Action"] = out["Action"].fillna(out["Action Type"]).astype(str) if "Action" in out else out["Action Type"]
    out["Result"] = out["Result"].fillna("No result").astype(str) if "Result" in out else "No result"

    out["Positive PXT"] = (
        pd.concat(
            [
                _numeric(out, "PXT Pass"),
                _numeric(out, "PXT Shot"),
                _numeric(out, "Team xT"),
            ],
            axis=1,
        )
        .clip(lower=0)
        .max(axis=1)
        .fillna(0)
    )
    out["Team xT"] = _numeric(out, "Team xT")
    out["Shot xG"] = _numeric(out, "Shot xG")
    out["Minute"] = _numeric(out, "Minute", np.nan)
    return out


def _summarise_players(filtered: pd.DataFrame) -> pd.DataFrame:
    if filtered.empty:
        return pd.DataFrame(columns=["Player", "Actions", "Action Share", "Positive PXT", "Team xT", "Shot xG", "Primary Action"])

    summary = filtered.groupby("Player", as_index=False).agg(
        Actions=("Action Type", "size"),
        **{
            "Positive PXT": ("Positive PXT", "sum"),
            "Team xT": ("Team xT", "sum"),
            "Shot xG": ("Shot xG", "sum"),
            "Primary Action": ("Action Type", _mode),
        },
    )
    total_actions = float(summary["Actions"].sum())
    summary["Action Share"] = np.where(total_actions > 0, summary["Actions"] / total_actions * 100, 0)
    for column in ["Positive PXT", "Team xT", "Shot xG", "Action Share"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0)
    return summary


def _card(label: str, value: object, helper: str | None = None, *, text_value: bool = False, dark: bool = False) -> str:
    value_class = "pma-value pma-value-text" if text_value else "pma-value"
    card_class = "pma-card pma-card-dark" if dark else "pma-card"
    helper_html = f'<div class="pma-helper">{ui.esc(helper)}</div>' if helper else ""
    return (
        f'<div class="{card_class}">'
        f'<div class="pma-label">{ui.esc(label)}</div>'
        f'<div class="{value_class}">{ui.esc(value)}</div>'
        f"{helper_html}"
        "</div>"
    )


def _render_context_cards(match_row: pd.Series, season: str | None, team_name: str, events: pd.DataFrame) -> None:
    match_date = pd.to_datetime(match_row.get("Date"), errors="coerce")
    date_text = match_date.strftime("%d %b %Y") if pd.notna(match_date) else "Date unavailable"
    fixture = str(match_row.get("Match", "Unknown fixture"))
    html = "".join(
        [
            _card("Fixture", fixture, date_text, text_value=True, dark=True),
            _card("Team", team_name, f"Season {season or 'N/A'}", text_value=True),
            _card("Player Event Rows", f"{len(events):,}", "Rows With a Named Player"),
            _card("Players Involved", f"{events['Player'].nunique():,}" if not events.empty else "0", "Unique Players in Selected Rows"),
        ]
    )
    st.markdown(f'<div class="pma-context-grid">{html}</div>', unsafe_allow_html=True)


def _top_player(summary: pd.DataFrame, metric: str) -> pd.Series | None:
    if summary.empty or metric not in summary:
        return None
    ranked = summary.sort_values([metric, "Actions"], ascending=False).reset_index(drop=True)
    return ranked.iloc[0] if not ranked.empty else None


def _render_selection_cards(filtered: pd.DataFrame, summary: pd.DataFrame, value_metric: str) -> None:
    metric_label = _metric_label(value_metric)
    top_ranked = _top_player(summary, value_metric)
    top_ranked_name = str(top_ranked["Player"]) if top_ranked is not None else "N/A"
    top_ranked_value = charting.metric_text(top_ranked[value_metric], value_metric) if top_ranked is not None else "N/A"
    primary_action = _mode(filtered["Action Type"]) if not filtered.empty and "Action Type" in filtered else "N/A"
    html = "".join(
        [
            _card("Selected Actions", f"{len(filtered):,}", f"{summary['Player'].nunique():,} Players After Filters"),
            _card("Primary Action Type", primary_action.replace("_", " ").title(), "Most Common Selected Action", text_value=True),
            _card(f"Top By {metric_label}", top_ranked_name, top_ranked_value, text_value=True, dark=True),
            _card("Positive PXT", charting.metric_text(filtered["Positive PXT"].sum(), "Positive PXT"), "Total Positive Possession Value"),
            _card("Team xT", charting.metric_text(filtered["Team xT"].sum(), "Team xT"), "Total Expected Threat"),
            _card("Shot xG", charting.metric_text(filtered["Shot xG"].sum(), "Shot xG"), "Total Shot Quality"),
        ]
    )
    st.markdown(f'<div class="pma-context-grid">{html}</div>', unsafe_allow_html=True)


def _spotlight_card(label: str, row: pd.Series | None, metric: str) -> str:
    if row is None:
        name = "N/A"
        value = "N/A"
        meta = "No selected data"
    else:
        name = str(row.get("Player", "Unknown"))
        value = charting.metric_text(row.get(metric), metric)
        meta = (
            f'{charting.metric_text(row.get("Actions"), "Actions")} actions · '
            f'{charting.metric_text(row.get("Action Share"), "Action Share %")} share · '
            f'{ui.esc(row.get("Primary Action", "Unknown")).replace("_", " ").title()}'
        )
    return (
        '<div class="pma-spotlight-card">'
        f'<div class="pma-spotlight-top">{ui.esc(label)}</div>'
        '<div class="pma-spotlight-body">'
        f'<div class="pma-spotlight-name">{ui.esc(name)}</div>'
        f'<div class="pma-spotlight-value">{ui.esc(value)}</div>'
        f'<div class="pma-spotlight-meta">{meta}</div>'
        "</div>"
        "</div>"
    )


def _render_spotlights(summary: pd.DataFrame, value_metric: str) -> None:
    metric_label = _metric_label(value_metric)
    cards = [
        _spotlight_card(f"Leader: {metric_label}", _top_player(summary, value_metric), value_metric),
        _spotlight_card("Volume Leader", _top_player(summary, "Actions"), "Actions"),
        _spotlight_card("PXT Leader", _top_player(summary, "Positive PXT"), "Positive PXT"),
        _spotlight_card("Shot Threat Leader", _top_player(summary, "Shot xG"), "Shot xG"),
    ]
    st.markdown(f'<div class="pma-spotlight-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def _player_contribution_chart(
    summary: pd.DataFrame,
    value_metric: str,
    team_name: str,
    benchmark_summary: pd.DataFrame | None = None,
) -> go.Figure:
    plot_df = summary.sort_values(value_metric, ascending=True).copy()
    metric_label = _metric_label(value_metric)
    labels = plot_df["Player"].apply(lambda value: charting.wrap_label(value, width=19, max_lines=2))
    values = pd.to_numeric(plot_df[value_metric], errors="coerce").fillna(0)
    benchmark = benchmark_summary if benchmark_summary is not None and value_metric in benchmark_summary else plot_df
    benchmark_values = pd.to_numeric(benchmark[value_metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    average_value = float(benchmark_values.mean()) if not benchmark_values.empty else 0.0
    display_scale = max(float(values.max()) if len(values) else 0.0, average_value, 1.0)
    performance_scores = _performance_scores(values, average_value)
    delta_text = [_signed_metric_text(value - average_value, value_metric) for value in values]

    customdata = np.stack(
        [
            plot_df["Player"],
            plot_df["Actions"],
            plot_df["Action Share"],
            plot_df["Positive PXT"],
            plot_df["Team xT"],
            plot_df["Shot xG"],
            plot_df["Primary Action"],
            delta_text,
        ],
        axis=-1,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[display_scale] * len(plot_df),
            y=labels,
            orientation="h",
            marker=dict(color="rgba(17, 17, 17, 0.055)", line=dict(width=0)),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color=performance_scores,
                colorscale=[
                    [0.0, "#d92d20"],
                    [0.5, "#f2c94c"],
                    [1.0, "#12b76a"],
                ],
                cmin=0,
                cmax=1,
                colorbar=dict(
                    tickmode="array",
                    tickvals=[0, 0.5, 1],
                    ticktext=["Below", "Average", "Above"],
                    thickness=12,
                    len=0.72,
                ),
                line=dict(color="rgba(255,255,255,0.82)", width=0.7),
            ),
            text=[charting.metric_text(value, value_metric) for value in values],
            textposition="outside",
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b>"
                "<br>Actions: %{customdata[1]:.0f}"
                "<br>Action Share: %{customdata[2]:.1f}%"
                "<br>Positive PXT: %{customdata[3]:.2f}"
                "<br>Team xT: %{customdata[4]:.2f}"
                "<br>Shot xG: %{customdata[5]:.2f}"
                "<br>Primary Action: %{customdata[6]}"
                "<br>Vs Selected Average: %{customdata[7]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.update_layout(
        barmode="overlay",
        height=charting.horizontal_bar_height(len(plot_df), min_height=480, row_height=37, max_height=780),
        xaxis_title=value_metric,
        yaxis_title="",
        showlegend=False,
    )
    fig.update_xaxes(range=[0, display_scale * 1.18])
    if average_value > 0 or display_scale > 0:
        fig.add_shape(
            type="line",
            x0=average_value,
            x1=average_value,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color=ui.CHARLTON_BLACK, width=2, dash="dash"),
        )
        fig.add_annotation(
            x=average_value,
            y=1.045,
            xref="x",
            yref="paper",
            text=f"Selected Average: {charting.metric_text(average_value, value_metric)}",
            showarrow=False,
            xanchor="left" if average_value <= display_scale * 0.72 else "right",
            yanchor="bottom",
            bgcolor="#ffffff",
            bordercolor=ui.CHARLTON_BORDER,
            borderpad=4,
            font=dict(size=12, color=ui.CHARLTON_BLACK),
        )
    charting.format_xaxis(fig, value_metric)
    fig = charting.polish_figure(fig, f"{team_name}: Player Contribution Ranked by {metric_label}")
    fig.update_layout(margin=dict(l=30, r=118, t=92, b=54))
    return fig


def _action_mix_chart(filtered: pd.DataFrame, title: str) -> go.Figure:
    action_mix = (
        filtered.groupby("Action Type", as_index=False)
        .size()
        .rename(columns={"size": "Actions"})
        .sort_values("Actions", ascending=False)
    )
    if len(action_mix) > 8:
        top = action_mix.head(7).copy()
        other = pd.DataFrame([{"Action Type": "Other", "Actions": action_mix.iloc[7:]["Actions"].sum()}])
        action_mix = pd.concat([top, other], ignore_index=True)

    action_mix["Action Label"] = action_mix["Action Type"].astype(str).str.replace("_", " ", regex=False).str.title()
    total_actions = float(action_mix["Actions"].sum())
    action_mix["Action Share"] = np.where(total_actions > 0, action_mix["Actions"] / total_actions * 100, 0)
    fig = go.Figure(
        go.Pie(
            labels=action_mix["Action Label"],
            values=action_mix["Actions"],
            hole=0.62,
            marker=dict(colors=[ui.CHARLTON_RED, ui.CHARLTON_BLACK, "#7a7f87", ui.CHARLTON_DEEP_RED, "#c69214", "#b42318", "#475467", "#98a2b3"]),
            textinfo="percent",
            textposition="inside",
            customdata=np.stack([action_mix["Action Type"], action_mix["Actions"], action_mix["Action Share"]], axis=-1),
            hovertemplate=(
                "<b>%{label}</b>"
                "<br>Action Type: %{customdata[0]}"
                "<br>Actions: %{customdata[1]:,.0f}"
                "<br>Share: %{customdata[2]:.1f}%<extra></extra>"
            ),
        )
    )
    fig = charting.polish_figure(fig, title)
    fig.update_layout(
        height=470,
        hoverlabel=dict(
            align="left",
            bgcolor=ui.CHARLTON_BLACK,
            bordercolor=ui.CHARLTON_RED,
            font=dict(color="#ffffff", size=15, family="Inter, Arial, sans-serif"),
        ),
        margin=dict(l=20, r=30, t=72, b=30),
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
    )
    return fig


ma.page_header(
    "Player Match Actions",
    "Summarise real event actions, PXT and xG by player for a selected fixture and team.",
    "CAFC_DB Impect provider events supply player-level rows for event-covered match seasons.",
)
_player_actions_css()

season = ma.select_match_season(key="player_actions_match_season")
matches = ma.load_matches(season)
if matches.empty:
    st.warning("No match data is available for this season.")
    st.stop()

match_row = ma.match_selector(matches, key="player_actions_match")
team_name = ma.team_selector_for_match(match_row, key="player_actions_team")
events = data.load_match_events(season=season, match_id=match_row.get("MatchId"), team=team_name, limit=9000)
events = events[events["Player"].notna()].copy()
events = _prepare_events(events)

ma.section_heading("Selected Fixture Summary")
_render_context_cards(match_row, season, team_name, events)

if events.empty:
    st.info("No player-level event rows are available for this selected match and team.")
    st.stop()

ma.section_heading("Action Controls")
st.markdown(
    """
    <div class="pma-control-note">
        Use the filters to change the profile of the chart. Leaving Action Types empty keeps the full selected fixture.
    </div>
    """,
    unsafe_allow_html=True,
)
action_types = sorted(events["Action Type"].dropna().astype(str).unique().tolist())
control_cols = st.columns([2.2, 1, 1, 1])
selected_action_types = control_cols[0].multiselect("Action Types", action_types, default=action_types)
value_metric = control_cols[1].selectbox("Rank By", ["Actions", "Positive PXT", "Team xT", "Shot xG"])
top_n = control_cols[2].slider("Players Shown", 5, 22, 14)
min_actions = control_cols[3].slider("Minimum Actions", 1, 25, 1)

filtered = events.copy()
if selected_action_types:
    filtered = filtered[filtered["Action Type"].astype(str).isin(selected_action_types)].copy()

summary_all = _summarise_players(filtered)
summary_all = summary_all[summary_all["Actions"].ge(min_actions)].copy()

ma.section_heading("Selection Snapshot")
if filtered.empty or summary_all.empty:
    st.info("No player actions match the current filters.")
    st.stop()

_render_selection_cards(filtered, summary_all, value_metric)

summary = summary_all.sort_values(value_metric, ascending=True).tail(top_n)

ma.section_heading("Player Leaders")
_render_spotlights(summary_all, value_metric)

ma.section_heading("Player Event Contribution")
st.markdown(
    '<div class="pma-layout-note">Faint bars show the current scale. The dashed line is the selected-player average for the current action filters; colour runs red to green against that benchmark.</div>',
    unsafe_allow_html=True,
)
st.plotly_chart(_player_contribution_chart(summary, value_metric, team_name, summary_all), width="stretch")

ma.section_heading("Action Profile")
st.plotly_chart(_action_mix_chart(filtered, "Selected Action Type Mix"), width="stretch")
