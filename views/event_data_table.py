# =============================================================================
# EVENT DATA TABLE - selected-match raw event feed
# =============================================================================
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils import charting, data, match_analysis as ma, ui


def _event_table_css() -> None:
    st.markdown(
        """
        <style>
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            min-height: 2.35rem;
        }

        .stSelectbox div[data-baseweb="select"] *,
        .stMultiSelect div[data-baseweb="select"] * {
            font-size: 0.78rem !important;
            line-height: 1.2 !important;
        }

        .stSelectbox div[data-baseweb="select"] [data-baseweb="tag"],
        .stSelectbox div[data-baseweb="select"] span,
        .stSelectbox div[data-baseweb="select"] div {
            overflow-wrap: anywhere !important;
            text-overflow: clip !important;
            white-space: normal !important;
        }

        .stMultiSelect span[data-baseweb="tag"] {
            margin-bottom: 2px;
            margin-top: 2px;
        }

        div[data-baseweb="popover"] [role="listbox"] *,
        div[data-baseweb="popover"] [role="option"] * {
            font-size: 0.78rem !important;
            line-height: 1.25 !important;
        }

        .edt-summary-grid {
            display: grid;
            gap: 12px;
            grid-template-columns: minmax(280px, 1.7fr) repeat(3, minmax(130px, 1fr));
            margin: 8px 0 18px;
        }

        .edt-summary-card {
            background: #ffffff;
            border: 1px solid var(--ss-border);
            border-radius: 10px;
            border-top: 3px solid var(--ss-accent);
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-height: 92px;
            padding: 13px 15px;
        }

        .edt-summary-label {
            color: var(--ss-muted);
            font-size: 0.74rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            line-height: 1.25;
            margin-bottom: 9px;
            text-transform: uppercase;
        }

        .edt-summary-value {
            color: var(--ss-ink);
            font-size: clamp(1.25rem, 1.65vw, 1.6rem);
            font-weight: 650;
            letter-spacing: -0.035em;
            line-height: 1.08;
            overflow-wrap: anywhere;
        }

        .edt-summary-value-text {
            font-size: clamp(0.82rem, 0.96vw, 1rem);
            font-weight: 850;
            letter-spacing: -0.015em;
            line-height: 1.2;
        }

        .edt-chart-note {
            color: var(--ss-muted);
            font-size: 0.86rem;
            line-height: 1.4;
            margin: -2px 0 10px;
        }

        @media (max-width: 860px) {
            .edt-summary-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _summary_card(label: str, value: object, *, text_value: bool = False) -> str:
    value_class = "edt-summary-value edt-summary-value-text" if text_value else "edt-summary-value"
    return (
        '<div class="edt-summary-card">'
        f'<div class="edt-summary-label">{ui.esc(label)}</div>'
        f'<div class="{value_class}">{ui.esc(value)}</div>'
        "</div>"
    )


def _render_fixture_summary(match_row, events, season: str | None) -> None:
    try:
        score = f"{match_row['Home Goals']:.0f} - {match_row['Away Goals']:.0f}"
    except (KeyError, TypeError, ValueError):
        score = "N/A"
    html = "".join(
        [
            _summary_card("Fixture", str(match_row.get("Match", "Unknown")), text_value=True),
            _summary_card("Score", score),
            _summary_card("Normalised Event Rows", f"{len(events):,}"),
            _summary_card("Event Season", season or "All", text_value=True),
        ]
    )
    st.markdown(f'<div class="edt-summary-grid">{html}</div>', unsafe_allow_html=True)


def _team_color_map(teams: list[str]) -> dict[str, str]:
    unique_teams = sorted(set(str(team) for team in teams if str(team)))
    charlton = [team for team in unique_teams if "charlton" in team.lower()]
    others = [team for team in unique_teams if team not in charlton]
    ordered = charlton + others
    if not ordered:
        return {}
    color_map = {}
    for index, team in enumerate(ordered):
        color_map[team] = ui.CHARLTON_RED if index == 0 else ui.CHARLTON_BLACK
    return color_map


def _action_summary_chart(summary) -> go.Figure:
    if summary.empty:
        return go.Figure()

    action_totals = (
        summary.groupby("Action Type", as_index=False)["Actions"]
        .sum()
        .sort_values("Actions", ascending=False)
        .head(12)
    )
    action_order = action_totals.sort_values("Actions", ascending=True)["Action Type"].tolist()
    plot_df = summary[summary["Action Type"].isin(action_order)].copy()
    plot_df["Action Label"] = plot_df["Action Type"].astype(str).str.replace("_", " ", regex=False).str.title()
    label_map = dict(zip(plot_df["Action Type"], plot_df["Action Label"]))
    team_colors = _team_color_map(plot_df["Team"].dropna().astype(str).tolist())
    team_order = list(team_colors.keys())
    y_positions = list(range(len(action_order)))
    action_labels = [charting.wrap_label(label_map.get(action, action), width=18, max_lines=2) for action in action_order]
    max_value = 1.0

    if len(team_order) < 2:
        team = team_order[0] if team_order else "Team"
        team_df = plot_df[plot_df["Team"].astype(str) == team].set_index("Action Type")
        raw_values = [float(team_df.loc[action, "Actions"]) if action in team_df.index else 0 for action in action_order]
        max_value = max(max_value, max(raw_values) if raw_values else 0)
        fig = go.Figure(
            go.Bar(
                x=raw_values,
                y=action_labels,
                name=team,
                orientation="h",
                marker=dict(color=team_colors.get(team, ui.CHARLTON_RED), line=dict(color="#ffffff", width=0.8)),
                text=[f"<b>{charting.metric_text(value, 'Actions')}</b>" if value else "" for value in raw_values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=f"<b>{team}</b><br>%{{y}}: %{{x:.0f}}<extra></extra>",
            )
        )
        fig.update_xaxes(range=[0, max_value * 1.22])
        fig.update_layout(
            height=charting.horizontal_bar_height(len(action_order), min_height=440, row_height=44, max_height=720),
            xaxis_title="Events",
            yaxis_title="Event Type",
            showlegend=False,
        )
        fig = charting.polish_figure(fig, "Top Selected Event Types")
        fig.update_layout(margin=dict(l=146, r=72, t=84, b=54))
        fig.update_yaxes(tickfont=dict(size=13, color=ui.CHARLTON_BLACK, family="Inter SemiBold, Arial, sans-serif"))
        return fig

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.24,
        column_widths=[0.5, 0.5],
    )

    team_values: dict[str, list[float]] = {}
    for team in team_order[:2]:
        team_df = plot_df[plot_df["Team"].astype(str) == team].set_index("Action Type")
        raw_values = [float(team_df.loc[action, "Actions"]) if action in team_df.index else 0 for action in action_order]
        team_values[team] = raw_values
        max_value = max(max_value, max(raw_values) if raw_values else 0)
        customdata = np.stack(
            [
                [team] * len(action_order),
                [label_map.get(action, action) for action in action_order],
                raw_values,
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Bar(
                x=raw_values,
                y=y_positions,
                name=team,
                orientation="h",
                marker=dict(color=team_colors[team], line=dict(color="#ffffff", width=0.8)),
                cliponaxis=False,
                customdata=customdata,
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}: %{customdata[2]:.0f}<extra></extra>",
            ),
            row=1,
            col=1 if team == team_order[0] else 2,
        )

    label_gap = max(max_value * 0.045, 1.0)
    range_limit = max_value + max(max_value * 0.22, label_gap * 4.0)
    tick_limit = float(np.ceil(max_value / 10) * 10) if max_value >= 10 else max_value
    tick_values = [0, tick_limit / 2, tick_limit]
    tick_text = [charting.metric_text(value, "Actions") for value in tick_values]

    fig.update_xaxes(range=[range_limit, 0], tickvals=tick_values, ticktext=tick_text, row=1, col=1)
    fig.update_xaxes(range=[0, range_limit], tickvals=tick_values, ticktext=tick_text, row=1, col=2)

    for col_index, team in enumerate(team_order[:2], start=1):
        is_left_team = col_index == 1
        xref = "x" if col_index == 1 else "x2"
        yref = "y" if col_index == 1 else "y2"
        for y_value, raw_value in zip(y_positions, team_values[team]):
            if raw_value <= 0:
                continue
            fig.add_annotation(
                x=raw_value + label_gap,
                y=y_value,
                xref=xref,
                yref=yref,
                text=f"<b>{charting.metric_text(raw_value, 'Actions')}</b>",
                showarrow=False,
                xanchor="right" if is_left_team else "left",
                yanchor="middle",
                xshift=-4 if is_left_team else 4,
                font=dict(
                    color=team_colors.get(team, ui.CHARLTON_BLACK),
                    family="Inter SemiBold, Arial, sans-serif",
                    size=14,
                ),
            )

    for y_value, label in zip(y_positions, action_labels):
        fig.add_annotation(
            x=0.5,
            y=y_value,
            xref="paper",
            yref="y",
            text=f"<b>{label}</b>",
            showarrow=False,
            xanchor="center",
            yanchor="middle",
            align="center",
            font=dict(color=ui.CHARLTON_BLACK, family="Inter SemiBold, Arial, sans-serif", size=15),
        )

    fig.update_layout(
        barmode="overlay",
        height=charting.horizontal_bar_height(len(action_order), min_height=440, row_height=44, max_height=720),
        xaxis_title="Events",
        xaxis2_title="Events",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        showlegend=True,
    )
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_xaxes(tickformat=".0f", zeroline=False)
    fig = charting.polish_figure(fig, "Top Selected Event Types")
    fig.update_layout(margin=dict(l=72, r=72, t=92, b=54))
    return fig


ma.page_header(
    "Event Data Table",
    "Inspect, filter and export the normalised selected-match Impect event rows available to the app.",
    "CAFC_DB Impect provider events supply timestamps, players, teams, actions, results, adjusted coordinates, PXT and xG through the app's event adapter.",
)
_event_table_css()

season = ma.select_match_season(key="event_table_season")
matches = ma.load_matches(season)
if matches.empty:
    st.warning("No match data is available for this season.")
    st.stop()

match_row = ma.match_selector(matches, key="event_table_match")
events = data.load_match_events(season=season, match_id=match_row.get("MatchId"), limit=12000)

ma.section_heading("Selected fixture")
_render_fixture_summary(match_row, events, season)

if events.empty:
    st.info("No event rows are available for this selected fixture.")
    st.stop()

ma.section_heading("Filters")
teams = sorted(events["Team"].dropna().astype(str).unique().tolist())
action_types = sorted(events["Action Type"].dropna().astype(str).unique().tolist())
players = sorted(events["Player"].dropna().astype(str).unique().tolist())
filter_cols = st.columns(3)
selected_teams = filter_cols[0].multiselect("Teams", teams, default=teams)
selected_action_types = filter_cols[1].multiselect("Action types", action_types, default=action_types)
selected_players = filter_cols[2].multiselect("Players", players, default=[])

filtered = events.copy()
if selected_teams:
    filtered = filtered[filtered["Team"].astype(str).isin(selected_teams)]
if selected_action_types:
    filtered = filtered[filtered["Action Type"].astype(str).isin(selected_action_types)]
if selected_players:
    filtered = filtered[filtered["Player"].astype(str).isin(selected_players)]

ma.section_heading("Action summary")
summary = filtered.groupby(["Team", "Action Type"], as_index=False).size().rename(columns={"size": "Actions"})
if summary.empty:
    st.caption("The current filters remove all event rows.")
else:
    st.markdown(
        '<div class="edt-chart-note">Mirrored bars compare the two teams. Action names sit in the centre, removing the cluttered side-axis labels.</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(_action_summary_chart(summary), width="stretch")

ma.section_heading("Event rows")
if filtered.empty:
    st.caption("The current filters remove all event rows.")
else:
    st.caption(f"{len(filtered):,} of {len(events):,} event rows match the current filters.")
    table_cols = ma.available_columns(
        filtered,
        [
            "Minute",
            "Period",
            "Team",
            "Player",
            "Position",
            "Action Type",
            "Action",
            "Body Part",
            "Result",
            "Pressure",
            "Start X",
            "Start Y",
            "End X",
            "End Y",
            "Receiver",
            "Team xT",
            "PXT Pass",
            "PXT Shot",
            "Shot xG",
            "Set Piece",
        ],
    )
    st.dataframe(filtered[table_cols].sort_values("Minute"), width="stretch", hide_index=True)
    st.download_button(
        "Download filtered event rows (CSV)",
        filtered[table_cols].to_csv(index=False),
        file_name=f"event_data_{str(match_row.get('MatchId', 'match'))}.csv",
        mime="text/csv",
    )
