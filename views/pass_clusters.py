# =============================================================================
# PASSING IDENTITY - league passing profile comparison
# =============================================================================
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charting
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


def _percentile_text(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.0f}th percentile"


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
        rows.append(
            {
                "Metric": metric,
                "Selected": selected_value,
                "League Average": league_average,
                "Gap": selected_value - league_average,
            }
        )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        fig = ta.polish_figure(go.Figure(), "Selected Team vs League Average")
        fig.update_layout(height=360)
        return fig

    plot_df["Colour"] = np.where(plot_df["Gap"].ge(0), "#16803c", "#c30017")
    plot_df["Text"] = [
        f"{charting.metric_text(row['Selected'], row['Metric'])} vs avg {charting.metric_text(row['League Average'], row['Metric'])}"
        for _, row in plot_df.iterrows()
    ]
    fig = go.Figure(
        go.Bar(
            x=plot_df["Gap"],
            y=plot_df["Metric"].map(lambda value: charting.wrap_label(value, width=24, max_lines=2)),
            orientation="h",
            marker_color=plot_df["Colour"],
            text=plot_df["Text"],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([plot_df["Metric"], plot_df["Selected"], plot_df["League Average"]], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Selected: %{customdata[1]:.2f}<br>"
                "League average: %{customdata[2]:.2f}<br>Gap: %{x:+.2f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line=dict(color="#7a7f87", width=1.4, dash="dash"))
    fig.update_layout(height=390, xaxis_title="Gap to League Average", yaxis_title="", showlegend=False)
    return ta.polish_figure(fig, f"{team_name}: Passing Metric Gaps")


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
