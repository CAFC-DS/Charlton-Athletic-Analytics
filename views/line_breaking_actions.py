# =============================================================================
# LINE-BREAKING ACTIONS - event-level Impect bypassed-player map
# =============================================================================
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charting, data, ui
from utils import match_analysis as ma
from utils import pitch


# Impect's raw ACTION values that can carry bypassed-opponent/defender value,
# grouped into the technique families the analytics team thinks in.
TECHNIQUE_GROUPS: dict[str, str] = {
    "LOW_PASS": "Ground Pass",
    "DIAGONAL_PASS": "Diagonal Pass",
    "CHIPPED_PASS": "Chipped Pass",
    "SHORT_AERIAL_PASS": "Short Aerial Pass",
    "HIGH_CROSS": "Cross (High)",
    "LOW_CROSS": "Cross (Low)",
    "HEADER": "Header",
}
TECHNIQUE_COLOURS: dict[str, str] = {
    "Ground Pass": ui.CHARLTON_RED,
    "Diagonal Pass": ui.CHARLTON_BLACK,
    "Chipped Pass": "#c69214",
    "Short Aerial Pass": "#12b76a",
    "Cross (High)": "#2563eb",
    "Cross (Low)": "#7a4dc4",
    "Header": "#7a7f87",
    "Carry / Dribble": "#e04f9f",
    "Other": "#98a2b3",
}


def _line_breaking_css() -> None:
    st.markdown(
        """
        <style>
        .lb-summary-card {
            background: var(--ss-panel);
            border: 1px solid var(--ss-border);
            border-radius: 8px;
            border-top: 3px solid var(--ss-accent);
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-height: 102px;
            padding: 14px 16px;
        }

        .lb-summary-label {
            color: var(--ss-muted);
            font-size: 0.875rem;
            font-weight: 650;
            line-height: 1.25;
            margin-bottom: 14px;
        }

        .lb-summary-value {
            color: var(--ss-ink);
            display: block;
            font-size: clamp(1.45rem, 1.85vw, 1.85rem);
            font-weight: 400;
            letter-spacing: -0.03em;
            line-height: 1.1;
        }

        .lb-summary-value-text {
            font-size: clamp(0.78rem, 0.92vw, 0.98rem);
            letter-spacing: -0.01em;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _summary_card(label: str, value: object, *, text_value: bool = False) -> None:
    value_class = "lb-summary-value lb-summary-value-text" if text_value else "lb-summary-value"
    st.markdown(
        f"""
        <div class="lb-summary-card">
            <div class="lb-summary-label">{ui.esc(label)}</div>
            <div class="{value_class}">{ui.esc(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _numeric(events: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in events:
        return pd.Series(default, index=events.index, dtype="float64")
    return pd.to_numeric(events[col], errors="coerce").fillna(default)


def _technique_label(action: object, action_type: object) -> str:
    action_key = "" if action is None else str(action).strip().upper()
    if action_key in TECHNIQUE_GROUPS:
        return TECHNIQUE_GROUPS[action_key]
    if str(action_type or "").strip().upper() == "DRIBBLE":
        return "Carry / Dribble"
    return "Other"


def _add_technique(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["Technique"] = [
        _technique_label(action, action_type)
        for action, action_type in zip(out.get("Action", pd.Series(dtype=object)), out.get("Action Type", pd.Series(dtype=object)), strict=False)
    ]
    return out


def _technique_breakdown_figure(events: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if events.empty or "Technique" not in events:
        return charting.polish_figure(fig, title, height=380)

    summary = (
        events.assign(**{
            "Bypassed Opponents": _numeric(events, "Bypassed Opponents"),
            "Bypassed Defenders": _numeric(events, "Bypassed Defenders"),
        })
        .groupby("Technique", as_index=False)
        .agg(Actions=("Technique", "size"), **{
            "Bypassed Opponents": ("Bypassed Opponents", "sum"),
            "Bypassed Defenders": ("Bypassed Defenders", "sum"),
        })
        .sort_values("Actions", ascending=True)
    )
    if summary.empty:
        return charting.polish_figure(fig, title, height=380)

    colours = [TECHNIQUE_COLOURS.get(technique, "#98a2b3") for technique in summary["Technique"]]
    customdata = np.stack([summary["Bypassed Opponents"], summary["Bypassed Defenders"]], axis=-1)
    fig.add_trace(
        go.Bar(
            x=summary["Actions"],
            y=summary["Technique"],
            orientation="h",
            marker=dict(color=colours, line=dict(color="#ffffff", width=1)),
            text=[f"{value:.0f}" for value in summary["Actions"]],
            textposition="outside",
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                "<b>%{y}</b><br>Actions: %{x:.0f}"
                "<br>Bypassed opponents: %{customdata[0]:.1f}"
                "<br>Bypassed defenders: %{customdata[1]:.1f}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="Line-breaking actions")
    fig.update_yaxes(title="")
    fig.update_layout(showlegend=False)
    return charting.polish_figure(fig, title, height=charting.horizontal_bar_height(len(summary), min_height=320, row_height=38, max_height=520))


def _line_breaking_events(
    events: pd.DataFrame,
    min_bypassed_opponents: float,
    min_bypassed_defenders: float,
) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    required = [col for col in ["Start X", "Start Y"] if col in events]
    if len(required) < 2:
        return pd.DataFrame(columns=events.columns)

    out = events.dropna(subset=required).copy()
    out["Bypassed Opponents"] = _numeric(out, "Bypassed Opponents")
    out["Bypassed Defenders"] = _numeric(out, "Bypassed Defenders")
    out = out[
        out["Bypassed Opponents"].ge(float(min_bypassed_opponents))
        & out["Bypassed Defenders"].ge(float(min_bypassed_defenders))
    ].copy()
    return out


def _rank_actions(events: pd.DataFrame, max_actions: int | None = None) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    ranked = events.copy()
    for col in ["Packing xG", "PXT Pass", "PXT Shot", "Minute"]:
        ranked[col] = _numeric(ranked, col)
    ranked["_Threat Sort"] = ranked[["Packing xG", "PXT Pass", "PXT Shot"]].clip(lower=0).max(axis=1).fillna(0)
    ranked = ranked.sort_values(
        ["Bypassed Defenders", "Bypassed Opponents", "_Threat Sort", "Minute"],
        ascending=[False, False, False, True],
    )
    if max_actions is not None and len(ranked) > max_actions:
        ranked = ranked.head(max(int(max_actions), 1)).copy()
    return ranked.drop(columns=["_Threat Sort"], errors="ignore")


def _display_table(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    out = _rank_actions(events).copy()
    for col in ["Bypassed Opponents", "Bypassed Defenders", "Packing xG", "PXT Pass", "PXT Shot"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    display_cols = ma.available_columns(
        out,
        [
            "Minute",
            "Player",
            "Position",
            "Technique",
            "Receiver",
            "Result",
            "Bypassed Opponents",
            "Bypassed Defenders",
            "Packing xG",
            "PXT Pass",
            "PXT Shot",
            "Start Lane",
            "End Lane",
            "Start X",
            "Start Y",
            "End X",
            "End Y",
        ],
    )
    return out[display_cols]


ma.page_header(
    "Line-Breaking Actions",
    "Map the selected team's event-level actions that bypass opponents or defenders, broken down by technique and the players on both ends of the action.",
    "CAFC_DB Impect provider events supply opponents and defenders bypassed. These packing metrics count players taken out of the game by an action.",
)
_line_breaking_css()

def _render_match_tab() -> None:
    season = ma.select_match_season(key="line_breaking_actions_season")
    matches = ma.load_matches(season)
    if matches.empty:
        st.warning("No match data is available for this season.")
        return

    match_row = ma.match_selector(matches, key="line_breaking_actions_match")
    team_name = ma.team_selector_for_match(match_row, key="line_breaking_actions_team")
    events = data.load_match_events(season=season, match_id=match_row.get("MatchId"), team=team_name, limit=20000)
    if events.empty:
        st.info("No event-level rows are available for this selected match and team.")
        return
    events = _add_technique(events)

    base_breaks = _line_breaking_events(events, min_bypassed_opponents=0.01, min_bypassed_defenders=0)
    if base_breaks.empty:
        st.info("No positive bypassed-player values are available for this selected match and team.")
        return

    ma.section_heading("Line-breaking controls")
    control_cols = st.columns([1, 1, 2, 1])
    min_opponents = control_cols[0].slider("Minimum bypassed opponents", 1, 8, 1)
    min_defenders = control_cols[1].slider("Minimum bypassed defenders", 0, 5, 0)
    action_types = sorted(base_breaks["Action Type"].dropna().astype(str).unique().tolist())
    selected_action_types = control_cols[2].multiselect("Action types", action_types, default=action_types)
    max_actions = control_cols[3].slider("Maximum plotted actions", 25, 500, min(250, max(len(base_breaks), 25)), step=25)
    control_cols[3].caption(
        "If the selection exceeds this limit, the map keeps the highest bypassed defenders first, "
        "then highest bypassed opponents, then highest Packing xG/PXT."
    )

    filtered = _line_breaking_events(events, min_opponents, min_defenders)
    if selected_action_types:
        filtered = filtered[filtered["Action Type"].astype(str).isin(selected_action_types)].copy()
    plotted = _rank_actions(filtered, max_actions)

    ma.section_heading("Selected fixture summary")
    metric_cols = st.columns(5)
    with metric_cols[0]:
        _summary_card("Fixture", str(match_row.get("Match", "Unknown")), text_value=True)
    with metric_cols[1]:
        _summary_card("Team", team_name, text_value=True)
    with metric_cols[2]:
        _summary_card("Actions selected", f"{len(filtered):,}")
    with metric_cols[3]:
        _summary_card("Bypassed opponents", f"{_numeric(filtered, 'Bypassed Opponents').sum():.1f}")
    with metric_cols[4]:
        _summary_card("Bypassed defenders", f"{_numeric(filtered, 'Bypassed Defenders').sum():.1f}")

    ma.section_heading(f"{team_name}: Line-Breaking Actions")
    if filtered.empty:
        st.info("No actions match the current filters.")
    else:
        st.plotly_chart(
            pitch.line_breaking_actions_map(
                filtered,
                team_name,
                f"{team_name}: Line-Breaking Actions",
                min_bypassed_opponents=min_opponents,
                min_bypassed_defenders=min_defenders,
                max_actions=max_actions,
            ),
            width="stretch",
        )
        st.caption(
            "Lines show available action start-to-end paths. Markers use the end location when available, otherwise the start location. "
            "Marker size reflects bypassed opponents; the defender filter narrows this to actions that also bypass recognised defenders."
        )

    ma.section_heading("Line-Break Variations")
    if filtered.empty:
        st.caption("No actions match the current filters.")
    else:
        st.plotly_chart(
            _technique_breakdown_figure(filtered, f"{team_name}: Line Breaks by Technique"),
            width="stretch",
        )
        st.caption(
            "Technique groups Impect's raw action label (ground pass, diagonal, chipped pass, cross, header, carry/dribble) "
            "so the same line-breaking total can be read by how the ball was moved."
        )

    ma.section_heading("Line-breaking action table")
    if plotted.empty:
        st.caption("No line-breaking actions are available for the current selection.")
    else:
        st.dataframe(_display_table(plotted), width="stretch", hide_index=True)
        st.caption(
            "Player is the player who broke the line; Receiver is who the pass reached (blank for dribbles/carries "
            "and passes without a recorded receiver)."
        )


match_tab, leaderboard_tab = st.tabs(["Match Map", "Season Leaderboard"])

with match_tab:
    _render_match_tab()

with leaderboard_tab:
    ma.section_heading("Season Line-Break Leaderboard")
    st.caption(
        "Ranks every league player by season line-breaking output. 'Initiating' totals the player's own bypassed-opponent "
        "actions (passes, carries, dribbles); 'Receiving' uses Impect's receiving-progression KPIs to show who was most "
        "often the target of a line-breaking pass — the other player involved in the action."
    )
    leaderboard_season = ma.select_player_season(key="line_breaking_leaderboard_season")
    leaderboard_players = data.load_players(leaderboard_season)
    if leaderboard_players.empty:
        st.info("No player season data is available for this season.")
    else:
        team_options = sorted(leaderboard_players["Team"].dropna().astype(str).unique().tolist())
        default_team_index = next(
            (index for index, team in enumerate(team_options) if "charlton" in team.lower()), 0
        )
        leaderboard_controls = st.columns([1.2, 1.0, 0.9])
        team_filter = leaderboard_controls[0].selectbox(
            "Team filter", ["All Teams", *team_options], index=default_team_index + 1, key="line_breaking_leaderboard_team"
        )
        role = leaderboard_controls[1].radio(
            "Role", ["Initiating", "Receiving"], horizontal=True, key="line_breaking_leaderboard_role"
        )
        leaderboard_top_n = leaderboard_controls[2].slider(
            "Players shown", 5, 30, 15, key="line_breaking_leaderboard_top_n"
        )

        role_metric = "Bypassed Opponents /90" if role == "Initiating" else "Receiving Progression /90"
        role_secondary = "Bypassed Defenders /90" if role == "Initiating" else "Receiving Defenders Bypassed /90"
        pool = leaderboard_players.copy()
        if team_filter != "All Teams":
            pool = pool[pool["Team"].astype(str).eq(team_filter)].copy()
        pool = pool[pd.to_numeric(pool["Minutes"], errors="coerce").fillna(0).ge(45)].copy()
        pool[role_metric] = pd.to_numeric(pool.get(role_metric), errors="coerce")
        pool = pool.dropna(subset=[role_metric]).nlargest(leaderboard_top_n, role_metric).sort_values(role_metric)
        if pool.empty:
            st.info("No players meet the minimum-minutes threshold for this selection.")
        else:
            colours = [ui.CHARLTON_RED if "charlton" in str(team).lower() else ui.CHARLTON_BLACK for team in pool["Team"]]
            customdata = np.stack(
                [pool["Team"], pd.to_numeric(pool.get(role_secondary), errors="coerce").fillna(0), pool["Minutes"]],
                axis=-1,
            )
            fig = go.Figure(
                go.Bar(
                    x=pool[role_metric],
                    y=pool["Player"],
                    orientation="h",
                    marker=dict(color=colours, line=dict(color="#ffffff", width=1)),
                    text=[f"{value:.2f}" for value in pool[role_metric]],
                    textposition="outside",
                    cliponaxis=False,
                    customdata=customdata,
                    hovertemplate=(
                        "<b>%{y}</b> · %{customdata[0]}"
                        f"<br>{role_metric}: " + "%{x:.2f}"
                        f"<br>{role_secondary}: " + "%{customdata[1]:.2f}"
                        "<br>Minutes: %{customdata[2]:.0f}<extra></extra>"
                    ),
                )
            )
            fig.update_xaxes(title=role_metric)
            fig.update_yaxes(title="")
            fig.update_layout(showlegend=False)
            fig = charting.polish_figure(
                fig,
                f"{leaderboard_season}: Top {role.lower()} line-breakers"
                + ("" if team_filter == "All Teams" else f" · {team_filter}"),
                height=charting.horizontal_bar_height(len(pool), min_height=380, row_height=34, max_height=760),
            )
            st.plotly_chart(fig, width="stretch")

            table_cols = ma.available_columns(
                pool, ["Player", "Team", "Position", "Minutes", role_metric, role_secondary]
            )
            st.dataframe(
                pool[table_cols].sort_values(role_metric, ascending=False),
                width="stretch",
                hide_index=True,
            )
