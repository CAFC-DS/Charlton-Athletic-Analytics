# =============================================================================
# PASSING IDENTITY - league passing profile comparison
# =============================================================================
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charting
from utils import data
from utils import team_analysis as ta


PASSING_SOURCE = (
    "Team passing metrics come from CAFC_DB Impect squad-iteration KPI facts. Pass % is "
    "recalculated from the provider's successful and unsuccessful pass components."
)


def _page_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stMetricValue"] {
            font-size: 1.45rem;
            line-height: 1.18;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _value(frame: pd.DataFrame, metric: str, team_name: str) -> float:
    if frame.empty or metric not in frame:
        return np.nan
    row = frame[frame["Team"].astype(str).eq(str(team_name))]
    if row.empty:
        return np.nan
    value = pd.to_numeric(row[metric], errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else np.nan


def _ordinal_suffix(rounded: int) -> str:
    if 11 <= abs(rounded) % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(abs(rounded) % 10, "th")


def _percentile_text(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    rounded = round(number)
    return f"{rounded:.0f}{_ordinal_suffix(rounded)} percentile"


def _metric_text(value: object, metric: str) -> str:
    return charting.metric_text(value, metric)


def _profile_cards(clustered: pd.DataFrame, team_name: str) -> None:
    selected = clustered[clustered["Team"].astype(str).eq(str(team_name))].iloc[0]
    league_average_security = pd.to_numeric(clustered["Pass Security Percentile"], errors="coerce").mean()
    league_average_progression = pd.to_numeric(clustered["Progression Percentile"], errors="coerce").mean()

    cols = st.columns(4)
    cols[0].metric("Passing Identity", selected["Cluster"])
    cols[1].metric(
        "Pass Security",
        _percentile_text(selected["Pass Security Percentile"]),
        f"{float(selected['Pass Security Percentile']) - league_average_security:+.0f} vs league midpoint",
        delta_color="off",
    )
    cols[2].metric(
        "Progression",
        _percentile_text(selected["Progression Percentile"]),
        f"{float(selected['Progression Percentile']) - league_average_progression:+.0f} vs league midpoint",
        delta_color="off",
    )
    cols[3].metric(
        "Pass Completion",
        _metric_text(selected.get("Pass %"), "Pass %"),
        f"League avg {_metric_text(pd.to_numeric(clustered['Pass %'], errors='coerce').mean(), 'Pass %')}",
        delta_color="off",
    )


def _passing_map(clustered: pd.DataFrame, selected: str) -> go.Figure:
    fig = ta.cluster_chart(clustered, selected=selected)
    fig.update_layout(
        title=dict(text="League Passing Identity Map", x=0.01, xanchor="left"),
        height=620,
        margin=dict(l=54, r=34, t=82, b=68),
        legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="left", x=0, title_text="Identity"),
    )
    fig.update_xaxes(title="Pass Security Percentile - completion and ball retention proxy")
    fig.update_yaxes(title="Progression Percentile - final-third access and opponents bypassed")
    fig.add_annotation(
        x=82,
        y=96,
        text="Controlled progressors",
        showarrow=False,
        font=dict(size=12, color="#16803c"),
        bgcolor="rgba(255,255,255,0.78)",
        bordercolor="#d9eadf",
        borderwidth=1,
        borderpad=5,
    )
    fig.add_annotation(
        x=20,
        y=96,
        text="Direct progressors",
        showarrow=False,
        font=dict(size=12, color="#c30017"),
        bgcolor="rgba(255,255,255,0.78)",
        bordercolor="#f0ccd1",
        borderwidth=1,
        borderpad=5,
    )
    fig.add_annotation(
        x=82,
        y=10,
        text="Secure circulators",
        showarrow=False,
        font=dict(size=12, color="#344054"),
        bgcolor="rgba(255,255,255,0.78)",
        bordercolor="#d0d5dd",
        borderwidth=1,
        borderpad=5,
    )
    return fig


def _metric_comparison_chart(clustered: pd.DataFrame, team_name: str) -> go.Figure:
    metrics = ["Pass %", "Passes to Final 3rd /90", "Bypassed Opponents /90"]
    rows = []
    for metric in metrics:
        if metric not in clustered:
            continue
        selected_value = _value(clustered, metric, team_name)
        league_average = pd.to_numeric(clustered[metric], errors="coerce").mean()
        percentile_series = ta.percentile(pd.to_numeric(clustered[metric], errors="coerce"), higher_is_better=True)
        team_row = clustered["Team"].astype(str).eq(str(team_name))
        selected_percentile = float(percentile_series[team_row].iloc[0]) if team_row.any() else np.nan
        rows.append(
            {
                "Metric": metric,
                "Selected": selected_value,
                "League Average": league_average,
                "Percentile": selected_percentile,
                "Percentile Gap": selected_percentile - 50.0,
            }
        )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        fig = ta.polish_figure(go.Figure(), "Selected Team vs League Average")
        fig.update_layout(height=360)
        return fig

    plot_df["Colour"] = np.where(plot_df["Percentile Gap"].ge(0), "#16803c", "#c30017")
    plot_df["Label"] = [
        f"{row['Percentile']:.0f}th percentile ({charting.metric_text(row['Selected'], row['Metric'])})"
        for _, row in plot_df.iterrows()
    ]
    fig = go.Figure(
        go.Bar(
            x=plot_df["Percentile Gap"],
            y=plot_df["Metric"].map(lambda value: charting.wrap_label(value, width=22, max_lines=1)),
            orientation="h",
            marker_color=plot_df["Colour"],
            text=plot_df["Label"],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([plot_df["Metric"], plot_df["Selected"], plot_df["League Average"], plot_df["Percentile"]], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Selected: %{customdata[1]:.2f}<br>"
                "League average: %{customdata[2]:.2f}<br>League percentile: %{customdata[3]:.0f}th<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line=dict(color="#7a7f87", width=1.4, dash="dash"))
    max_gap = float(plot_df["Percentile Gap"].abs().max()) if not plot_df.empty else 50.0
    fig.update_layout(
        height=130 * len(plot_df) + 120,
        xaxis_title="League Percentile Gap (0 = league median)",
        yaxis_title="",
        showlegend=False,
        margin=dict(l=140, r=90, t=72, b=58),
    )
    fig.update_xaxes(range=[-max(max_gap, 20) * 1.35, max(max_gap, 20) * 1.35])
    return ta.polish_figure(fig, f"{team_name}: Passing Metric Gaps vs League Median")


def _closest_matches(clustered: pd.DataFrame, team_name: str, limit: int = 6) -> pd.DataFrame:
    rows = clustered.copy()
    selected = rows[rows["Team"].astype(str).eq(str(team_name))]
    if selected.empty:
        return pd.DataFrame()
    security = float(selected["Pass Security Percentile"].iloc[0])
    progression = float(selected["Progression Percentile"].iloc[0])
    rows["Similarity Distance"] = np.sqrt(
        (pd.to_numeric(rows["Pass Security Percentile"], errors="coerce") - security) ** 2
        + (pd.to_numeric(rows["Progression Percentile"], errors="coerce") - progression) ** 2
    )
    rows = rows[~rows["Team"].astype(str).eq(str(team_name))].copy()
    return (
        rows.sort_values("Similarity Distance")
        .head(limit)[
            [
                "Team",
                "Cluster",
                "Pass Security Percentile",
                "Progression Percentile",
                "Pass %",
                "Passes to Final 3rd /90",
                "Bypassed Opponents /90",
            ]
        ]
        .round(2)
    )


def _cluster_summary(clustered: pd.DataFrame) -> pd.DataFrame:
    grouped = clustered.groupby("Cluster", as_index=False).agg(
        Teams=("Team", "count"),
        **{
            "Avg Pass Security": ("Pass Security Percentile", "mean"),
            "Avg Progression": ("Progression Percentile", "mean"),
            "Avg Pass %": ("Pass %", "mean"),
            "Avg Final Third Passes /90": ("Passes to Final 3rd /90", "mean"),
            "Avg Bypassed Opponents /90": ("Bypassed Opponents /90", "mean"),
        },
    )
    return grouped.round(2).sort_values(["Avg Progression", "Avg Pass Security"], ascending=False)


def _terminology_key() -> None:
    st.markdown(
        """
        <div style="border:1px solid #e6edf5; border-radius:8px; padding:14px 16px; background:#ffffff;">
            <div style="font-weight:800; color:#172033; margin-bottom:8px;">How to read this page</div>
            <div style="color:#475467; line-height:1.45; font-size:0.9rem;">
                <b>Pass Security</b> is the team's league percentile for Pass %. 
                <b>Progression</b> is the average percentile of Passes to Final 3rd /90 and Bypassed Opponents /90.
                The cluster names are rule-based labels, not machine-learning outputs.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


ta.page_header(
    "Passing Identity",
    "Compare how each team progresses the ball relative to how securely they keep it.",
    PASSING_SOURCE,
    "This is a league-relative profile. It explains broad passing identity, not exact tactical structure or individual pass patterns.",
)
_page_css()

season = ta.select_season("players", key="pass_clusters_season")
teams = ta.load_team_style_data(season)
if teams.empty:
    st.warning("No team data is available for this season.")
    st.stop()

team_name = ta.team_selector(teams, key="pass_clusters_team")
clustered = ta.cluster_passing_profiles(teams)

ta.section_heading("Selected Team Passing Identity")
_profile_cards(clustered, team_name)
st.caption(
    "The point of this page is to separate teams that progress through secure possession from teams that progress more directly, "
    "and to flag teams that keep the ball without turning that possession into territory."
)

ta.section_heading("League Passing Identity Map")
_terminology_key()
st.plotly_chart(_passing_map(clustered, team_name), width="stretch")

ta.section_heading("Selected Team vs League")
st.plotly_chart(_metric_comparison_chart(clustered, team_name), width="stretch")
st.caption(
    "Bars show league-percentile points above or below the median (0), not raw metric gaps -- Pass %, Passes to "
    "Final 3rd /90 and Bypassed Opponents /90 sit on very different numeric scales, so a shared percentile axis is "
    "what makes them comparable in a single chart. Raw values sit in the outside label and hover."
)

ta.section_heading(f"{team_name} Crossing Profile")
event_seasons = data.list_seasons().get("matches", [])
if season not in event_seasons:
    st.caption(f"No event-level pass data is available for {season}, so a crossing breakdown can't be built for this season.")
else:
    season_matches = data.load_matches(season=season)
    team_fixtures = ta.match_rows_for_team(season_matches, team_name)
    match_ids = team_fixtures["MatchId"].dropna().tolist() if "MatchId" in team_fixtures else []
    if not match_ids:
        st.caption(f"No fixtures are available for {team_name} in {season}.")
    else:
        team_season_passes = data.load_match_events(
            season=season, team=team_name, match_ids=match_ids, action_types=["PASS"], limit=120000,
        )
        team_crosses = team_season_passes[data.is_cross(team_season_passes)].copy() if not team_season_passes.empty else team_season_passes
        st.caption(
            f"Built from {team_name}'s own {len(match_ids)} event-level {season} fixtures -- not part of the "
            "league-wide KPI-fact comparison above, since Impect does not publish a season-aggregate crossing KPI."
        )
        if team_crosses.empty:
            st.info(f"No cross events are available for {team_name} in {season}.")
        else:
            completed_crosses = int(team_crosses["Result"].astype(str).str.upper().eq("SUCCESS").sum())
            matches_played = len(match_ids)
            crossing_cols = st.columns(4)
            crossing_cols[0].metric("Crosses / 90", f"{len(team_crosses) / matches_played:.1f}" if matches_played else "N/A")
            crossing_cols[1].metric("Completion %", f"{completed_crosses / len(team_crosses) * 100:.1f}%")
            low_crosses = int(team_crosses["Action"].astype(str).str.upper().eq("LOW_CROSS").sum())
            high_crosses = int(team_crosses["Action"].astype(str).str.upper().eq("HIGH_CROSS").sum())
            crossing_cols[2].metric("Low / High Split", f"{low_crosses} / {high_crosses}")
            total_pxt = pd.to_numeric(team_crosses.get("PXT Pass"), errors="coerce").fillna(0).sum()
            crossing_cols[3].metric("Total PXT From Crosses", f"{total_pxt:.2f}")

            top_crossers = (
                team_crosses.groupby("Player", as_index=False)
                .agg(Attempts=("Player", "size"), Completed=("Result", lambda s: int(s.astype(str).str.upper().eq("SUCCESS").sum())))
                .sort_values("Attempts", ascending=False)
                .head(10)
            )
            top_crossers["Completion %"] = (top_crossers["Completed"] / top_crossers["Attempts"] * 100).round(1)
            st.dataframe(top_crossers, width="stretch", hide_index=True)

ta.section_heading("Closest Passing Profiles")
closest = _closest_matches(clustered, team_name)
if closest.empty:
    st.caption("No comparable teams found.")
else:
    st.dataframe(closest, width="stretch", hide_index=True)

ta.section_heading("Cluster Summary")
st.dataframe(_cluster_summary(clustered), width="stretch", hide_index=True)

with st.expander("Show Full League Passing Table"):
    display_columns = [
        "Team",
        "Cluster",
        "Pass Security Percentile",
        "Progression Percentile",
        "Pass %",
        "Passes to Final 3rd /90",
        "Bypassed Opponents /90",
    ]
    st.dataframe(
        clustered[display_columns].sort_values(["Progression Percentile", "Pass Security Percentile"], ascending=False),
        width="stretch",
        hide_index=True,
    )
