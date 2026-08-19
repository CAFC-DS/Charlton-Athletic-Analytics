# =============================================================================
# SIMILAR PLAYER SEARCH - recruitment-style profile matching
# =============================================================================
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charting, player_analysis as pa, ui


PROFILE_PRESETS = {
    "Role Specific": "Uses the metrics most relevant to the selected player's role.",
    "Attacking": "Goal threat, chance creation and shot profile.",
    "Passing & Progression": "Pass quality, volume, progression and final-third contribution.",
    "Defending": "Ball-winning, duel and defensive value metrics.",
    "Possession Security": "Retention, ball-loss and circulation profile.",
    "Goalkeeping": "Shot-stopping and goalkeeper distribution profile.",
    "Custom": "Choose the exact metrics used in the similarity calculation.",
}

PRESET_METRICS = {
    "Attacking": ["Goals /90", "Assists /90", "xG /90", "Post-Shot xG /90", "Shots /90"],
    "Passing & Progression": [
        "Pass %",
        "Successful Passes /90",
        "Passes to Final 3rd /90",
        "Pass Progression /90",
        "Cross Progression /90",
        "Bypassed Opponents /90",
        "Bypassed Defenders /90",
        "Receiving Progression /90",
        "Dribble Progression /90",
    ],
    "Defending": ["Ball Wins /90", "Ball Win Value /90", "Ground Duel Win %", "Aerial Duel Win %"],
    "Possession Security": [
        "Ball Security %",
        "Ball Losses /90",
        "Losses Per 100 Actions",
        "Critical Ball Losses /90",
        "Ball Loss Threat /90",
        "Team-Mates Bypassed By Losses /90",
        "Neutral Passes /90",
    ],
    "Goalkeeping": [
        "Goals Prevented /90",
        "Save Actions /90",
        "Post-Shot xG Faced /90",
        "Goals Conceded /90",
        "Pass %",
        "Successful Passes /90",
        "Pass Progression /90",
    ],
}


def _metric_meta(metric: str) -> tuple[str, bool, str]:
    return pa.PROFILE_METRIC_META.get(metric, ("General", True, metric))


def _metric_label(metric: str) -> str:
    return _metric_meta(metric)[2]


def _available_metrics(players: pd.DataFrame, metrics: list[str], min_non_null: int = 3) -> list[str]:
    available = []
    for metric in metrics:
        if metric in players and pd.to_numeric(players[metric], errors="coerce").notna().sum() >= min_non_null:
            available.append(metric)
    return available


def _age_years(birthdate: object) -> float:
    date = pd.to_datetime(birthdate, errors="coerce")
    if pd.isna(date):
        return np.nan
    today = pd.Timestamp.today().normalize()
    return round((today - date).days / 365.25, 1)


def _prepare_players(raw_players: pd.DataFrame) -> pd.DataFrame:
    players = pa.add_position_groups(raw_players).copy()
    players["_Age"] = players["Birthdate"].apply(_age_years) if "Birthdate" in players else np.nan
    players["_Minutes"] = pd.to_numeric(players["Minutes"], errors="coerce") if "Minutes" in players else np.nan
    players["_Search Label"] = players.apply(
        lambda row: (
            f"{row.get('Player', 'Unknown player')} - "
            f"{row.get('_Position Display', row.get('Position', 'Unknown position'))} - "
            f"{row.get('Team', 'Unknown team')} - "
            f"{charting.metric_text(row.get('_Minutes'), 'Minutes')} mins"
        ),
        axis=1,
    )
    return players


def _reference_default_index(players: pd.DataFrame, options: list[int]) -> int:
    selected_name = st.session_state.get("selected_player")
    if selected_name:
        matches = players.index[players["Player"].astype(str) == str(selected_name)].tolist()
        for match in matches:
            if match in options:
                return options.index(match)
    return 0


def _player_lookup_select(players: pd.DataFrame) -> int:
    options = sorted(
        players.index.tolist(),
        key=lambda idx: (
            str(players.at[idx, "Player"]).strip().split()[-1].casefold(),
            str(players.at[idx, "Player"]).casefold(),
            str(players.at[idx, "Team"]).casefold() if "Team" in players else "",
        ),
    )
    return st.selectbox(
        "Reference Player",
        options,
        index=_reference_default_index(players, options),
        format_func=lambda idx: players.at[idx, "_Search Label"],
        key="similar_player_reference_v2",
    )


def _metrics_for_profile(players: pd.DataFrame, role_group: str, profile: str) -> list[str]:
    if profile == "Role Specific":
        metrics = pa.profile_metrics_for_role(players, role_group)
    elif profile == "Custom":
        metrics = _available_metrics(players, list(pa.PROFILE_METRIC_META.keys()), min_non_null=2)
    else:
        metrics = PRESET_METRICS.get(profile, [])

    metrics = _available_metrics(players, metrics, min_non_null=2)
    if len(metrics) >= 3:
        return metrics

    fallback = _available_metrics(players, list(pa.PROFILE_METRIC_META.keys()), min_non_null=2)
    return fallback[:10]


def _peer_mask(players: pd.DataFrame, reference: pd.Series, peer_pool: str) -> pd.Series:
    mask = pd.Series(True, index=players.index)
    if peer_pool == "Same Role Group":
        mask &= players["Role Group"].astype(str) == str(reference.get("Role Group", ""))
    elif peer_pool == "Same Listed Position":
        mask &= players["_Position Display"].astype(str) == str(reference.get("_Position Display", ""))
    elif peer_pool == "All Outfield Players":
        mask &= players["Role Group"].astype(str) != "Goalkeeper"
    return mask


def _percentile_frame(universe: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    percentiles = pd.DataFrame(index=universe.index)
    for metric in metrics:
        higher_is_better = _metric_meta(metric)[1]
        percentiles[metric] = pa.percentile(universe[metric], higher_is_better=higher_is_better)
    return percentiles


def _format_metric_list(metric_gaps: pd.Series, best: bool) -> str:
    if metric_gaps.empty:
        return "N/A"
    ordered = metric_gaps.sort_values(ascending=best)
    parts = []
    for metric, gap in ordered.head(3).items():
        parts.append(f"{_metric_label(metric)} ({gap:.0f})")
    return ", ".join(parts)


def _similarity_results(
    players: pd.DataFrame,
    reference_index: int,
    candidate_pool: pd.DataFrame,
    universe: pd.DataFrame,
    metrics: list[str],
    min_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(candidate_pool) == 0 or len(metrics) < 3:
        return pd.DataFrame(), pd.DataFrame()

    percentiles = _percentile_frame(universe, metrics)
    if reference_index not in percentiles.index:
        return pd.DataFrame(), percentiles

    target = percentiles.loc[reference_index]
    rows = []
    min_valid = max(3, math.ceil(len(metrics) * min_coverage))

    for idx, row in candidate_pool.iterrows():
        if idx not in percentiles.index:
            continue
        gaps = (percentiles.loc[idx] - target).abs().dropna()
        gaps = gaps[target[gaps.index].notna()]
        if len(gaps) < min_valid:
            continue

        rmse = float(np.sqrt(np.mean(np.square(gaps))))
        similarity = max(0.0, min(100.0, 100.0 - rmse))
        rows.append(
            {
                "_Index": idx,
                "Player": row.get("Player", "Unknown player"),
                "Team": row.get("Team", "Unknown team"),
                "Position": row.get("_Position Display", row.get("Position", "Unknown position")),
                "Role Group": row.get("Role Group", "Unknown role"),
                "Minutes": row.get("_Minutes", row.get("Minutes", np.nan)),
                "Age": row.get("_Age", np.nan),
                "Foot": row.get("Foot", "Unknown"),
                "Nationality": row.get("Nationality", "Unknown"),
                "Similarity": round(similarity, 1),
                "Profile Difference": round(rmse, 1),
                "Metric Coverage %": round(len(gaps) / len(metrics) * 100, 0),
                "Closest Matching Metrics": _format_metric_list(gaps, best=True),
                "Biggest Profile Gaps": _format_metric_list(gaps, best=False),
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        return results, percentiles
    results = results.sort_values(["Similarity", "Minutes"], ascending=[False, False]).reset_index(drop=True)
    results.insert(0, "Rank", np.arange(1, len(results) + 1))
    return results, percentiles


def _summary_metric(label: str, value: object, metric: str | None = None) -> None:
    st.metric(label, charting.metric_text(value, metric))


def _inject_metric_value_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stMetricValue"] > div {
            font-size: 1.45rem;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _reference_card(reference: pd.Series) -> None:
    st.markdown(
        f"""
        <div class="pa-card">
            <div class="pa-card-icon">Reference Profile</div>
            <div class="pa-card-title">{ui.esc(reference.get('Player', 'Unknown player'))}</div>
            <div class="pa-card-body">
                {ui.esc(reference.get('Team', 'Unknown team'))}<br>
                {ui.esc(reference.get('_Position Display', reference.get('Position', 'Unknown position')))} |
                {ui.esc(reference.get('Role Group', 'Unknown role'))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _similarity_bar(results: pd.DataFrame, title: str) -> go.Figure:
    plot_df = results.head(12).sort_values("Similarity", ascending=True).copy()
    plot_df["_Label"] = plot_df["Player"].apply(lambda value: charting.wrap_label(value, 18, 2))
    colors = np.where(plot_df["Similarity"] >= 85, ui.CHARLTON_RED, "#7a7f87")
    fig = go.Figure(
        go.Bar(
            x=plot_df["Similarity"],
            y=plot_df["_Label"],
            orientation="h",
            marker_color=colors,
            text=[f"{value:.1f}" for value in plot_df["Similarity"]],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack(
                [
                    plot_df["Player"],
                    plot_df["Team"],
                    plot_df["Position"],
                    plot_df["Minutes"].fillna(0),
                    plot_df["Profile Difference"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "%{customdata[0]}<br>"
                "%{customdata[1]} | %{customdata[2]}<br>"
                "Minutes: %{customdata[3]:.0f}<br>"
                "Similarity: %{x:.1f}<br>"
                "Profile difference: %{customdata[4]:.1f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(height=charting.horizontal_bar_height(len(plot_df), min_height=430), xaxis_title="Similarity Score", yaxis_title="")
    fig.update_xaxes(range=[0, 105], tickformat=".0f")
    return charting.polish_figure(fig, title)


def _comparison_radar(
    percentiles: pd.DataFrame,
    reference_index: int,
    comparison_index: int,
    metrics: list[str],
    reference_name: str,
    comparison_name: str,
) -> go.Figure:
    available = [metric for metric in metrics if metric in percentiles and pd.notna(percentiles.at[reference_index, metric])]
    available = [metric for metric in available if comparison_index in percentiles.index and pd.notna(percentiles.at[comparison_index, metric])]
    if not available:
        return charting.polish_figure(go.Figure(), "Profile Percentile Comparison")

    labels = [charting.wrap_label(_metric_label(metric), 14, 2) for metric in available]
    theta = labels + [labels[0]]
    ref_values = [float(percentiles.at[reference_index, metric]) for metric in available]
    comp_values = [float(percentiles.at[comparison_index, metric]) for metric in available]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=ref_values + [ref_values[0]],
            theta=theta,
            fill="toself",
            name=reference_name,
            line=dict(color=ui.CHARLTON_RED, width=3),
            fillcolor="rgba(195, 0, 23, 0.18)",
            hovertemplate="%{theta}<br>Percentile: %{r:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=comp_values + [comp_values[0]],
            theta=theta,
            fill="toself",
            name=comparison_name,
            line=dict(color="#344054", width=3),
            fillcolor="rgba(52, 64, 84, 0.12)",
            hovertemplate="%{theta}<br>Percentile: %{r:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickformat=".0f"), angularaxis=dict(tickfont=dict(size=12))),
        height=560,
        showlegend=True,
    )
    return charting.polish_figure(fig, "Profile Percentile Comparison")


def _gap_heatmap(results: pd.DataFrame, percentiles: pd.DataFrame, reference_index: int, metrics: list[str]) -> go.Figure:
    top = results.head(8).copy()
    indexes = top["_Index"].tolist()
    available_metrics = [
        metric
        for metric in metrics
        if metric in percentiles and reference_index in percentiles.index and percentiles.loc[indexes, metric].notna().any()
    ]
    if not available_metrics:
        return charting.polish_figure(go.Figure(), "Profile Gap Map")

    z_values = []
    hover = []
    ref_values = percentiles.loc[reference_index, available_metrics]
    for _, row in top.iterrows():
        idx = row["_Index"]
        diff = percentiles.loc[idx, available_metrics] - ref_values
        z_values.append(diff.astype(float).tolist())
        hover.append(
            [
                f"{row['Player']}<br>{_metric_label(metric)}<br>Candidate minus reference: {diff[metric]:+.0f} percentile points"
                for metric in available_metrics
            ]
        )

    fig = go.Figure(
        go.Heatmap(
            z=z_values,
            x=[charting.wrap_label(_metric_label(metric), 13, 2) for metric in available_metrics],
            y=[charting.wrap_label(player, 18, 2) for player in top["Player"]],
            colorscale=[
                [0, "#2166ac"],
                [0.5, "#f7f7f7"],
                [1, ui.CHARLTON_RED],
            ],
            zmid=0,
            zmin=-60,
            zmax=60,
            colorbar=dict(title="Percentile<br>Gap"),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(height=520, xaxis_title="", yaxis_title="")
    return charting.polish_figure(fig, "Profile Gap Map")


def _metric_breakdown(
    players: pd.DataFrame,
    percentiles: pd.DataFrame,
    reference_index: int,
    comparison_index: int,
    metrics: list[str],
) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in percentiles or reference_index not in percentiles.index or comparison_index not in percentiles.index:
            continue
        ref_pct = percentiles.at[reference_index, metric]
        comp_pct = percentiles.at[comparison_index, metric]
        if pd.isna(ref_pct) or pd.isna(comp_pct):
            continue
        rows.append(
            {
                "Metric": _metric_label(metric),
                "Reference Value": charting.metric_text(players.at[reference_index, metric], metric),
                "Reference Percentile": round(float(ref_pct), 1),
                "Comparison Value": charting.metric_text(players.at[comparison_index, metric], metric),
                "Comparison Percentile": round(float(comp_pct), 1),
                "Gap": round(float(comp_pct - ref_pct), 1),
            }
        )
    return pd.DataFrame(rows)


def _open_profile(player_name: str) -> None:
    st.session_state["selected_player"] = player_name
    st.switch_page("views/player_profiles.py")


pa.page_header(
    "Similar Player Search",
    "Build a recruitment-style shortlist by comparing a reference player against role-aware metric profiles.",
    limitation=(
        "Similarity uses available Impect player metrics only. It does not include contract status, wage, scouting grades, "
        "injury history or full event-location fingerprints."
    ),
)
_inject_metric_value_css()

season = pa.select_season(key="similar_player_season")
players = _prepare_players(pa.load_player_data(season))
if players.empty:
    st.warning("No players are available for the selected season.")
    st.stop()

pa.section_heading("Reference Player")
st.caption("Choose the player whose profile you want to match. The search then compares candidates against this reference profile.")
reference_index = _player_lookup_select(players)
reference = players.loc[reference_index]
st.session_state["selected_player"] = str(reference.get("Player", ""))

ref_card, ref_m1, ref_m2, ref_m3, ref_m4 = st.columns([2.2, 1, 1, 1, 1])
with ref_card:
    _reference_card(reference)
with ref_m1:
    _summary_metric("Minutes", reference.get("_Minutes"), "Minutes")
with ref_m2:
    _summary_metric("Age", reference.get("_Age"), "Age")
with ref_m3:
    st.metric("Foot", str(reference.get("Foot", "Unknown")).title())
with ref_m4:
    st.metric("Nationality", str(reference.get("Nationality", "Unknown")))

pa.section_heading("Search Filters")
st.caption("Use these controls to define the candidate market before calculating similarity.")
filter_cols = st.columns([1.2, 1.1, 1, 1])
peer_pool = filter_cols[0].selectbox(
    "Candidate Pool",
    ["Same Role Group", "Same Listed Position", "All Outfield Players", "All Players"],
    help="Role-aware pools generally produce more useful matches than comparing every player together.",
)
max_minutes = int(max(1, math.ceil(float(players["_Minutes"].max(skipna=True) if players["_Minutes"].notna().any() else 1))))
# Defaulting to min(600, max_minutes) sets the floor at the pool's own highest
# minutes-played value once a season is young (e.g. 1 game week in, max_minutes
# might be ~97) -- that excludes almost every candidate by construction, not
# because they're genuinely unproven. Scaling by half the current maximum keeps
# the same 600-minute cap once a season is well underway, but stays usable early on.
default_minutes = min(600, max(0, round(max_minutes * 0.5)))
min_minutes = filter_cols[1].slider("Minimum Minutes", min_value=0, max_value=max_minutes, value=default_minutes, step=50 if max_minutes >= 500 else 10)
exclude_same_team = filter_cols[2].checkbox("Exclude Same Team", value=False)
top_n = filter_cols[3].slider("Shortlist Size", min_value=3, max_value=min(30, max(3, len(players) - 1)), value=min(12, max(3, len(players) - 1)))

team_options = sorted(players["Team"].dropna().astype(str).unique()) if "Team" in players else []
selected_teams = st.multiselect(
    "Restrict to Teams",
    team_options,
    default=[],
    help="Leave blank to search across every available team in the selected pool.",
)

pa.section_heading("Similarity Profile")
st.caption("Choose which football profile the search should match. Role Specific is the default because a useful match for a centre-back is different from a useful match for a winger.")
profile_cols = st.columns([1, 1])
profile = profile_cols[0].selectbox(
    "Metric Profile",
    list(PROFILE_PRESETS.keys()),
    help="\n".join([f"{name}: {description}" for name, description in PROFILE_PRESETS.items()]),
)
metric_basis = _metrics_for_profile(players, str(reference.get("Role Group", "Outfield")), profile)
all_metric_options = _available_metrics(players, list(pa.PROFILE_METRIC_META.keys()), min_non_null=2)
if profile == "Custom":
    selected_metrics = profile_cols[1].multiselect(
        "Similarity Metrics",
        all_metric_options,
        default=metric_basis[:8],
        format_func=lambda metric: f"{_metric_label(metric)} ({metric})",
    )
else:
    selected_metrics = profile_cols[1].multiselect(
        "Similarity Metrics",
        all_metric_options,
        default=metric_basis,
        format_func=lambda metric: f"{_metric_label(metric)} ({metric})",
        help="You can still adjust the metric set after choosing a preset.",
    )
min_coverage = st.slider("Minimum Metric Coverage", min_value=50, max_value=100, value=70, step=5) / 100

if len(selected_metrics) < 3:
    st.info("Select at least three metrics to calculate a useful similarity score.")
    st.stop()

base_mask = _peer_mask(players, reference, peer_pool)
candidate_mask = base_mask.copy()
candidate_mask &= players.index != reference_index
candidate_mask &= players["_Minutes"].fillna(0) >= min_minutes
if exclude_same_team and "Team" in players:
    candidate_mask &= players["Team"].astype(str) != str(reference.get("Team", ""))
if selected_teams and "Team" in players:
    candidate_mask &= players["Team"].astype(str).isin(selected_teams)

candidate_pool = players[candidate_mask].copy()
universe_mask = base_mask.copy()
universe_mask |= players.index == reference_index
universe = players[universe_mask].copy()

results, percentiles = _similarity_results(
    players=players,
    reference_index=reference_index,
    candidate_pool=candidate_pool,
    universe=universe,
    metrics=selected_metrics,
    min_coverage=min_coverage,
)

market_cols = st.columns(4)
market_cols[0].metric("Candidate Pool", len(candidate_pool))
market_cols[1].metric("Metrics Used", len(selected_metrics))
market_cols[2].metric("Reference Role", str(reference.get("Role Group", "Unknown")))
market_cols[3].metric("Comparable Universe", len(universe))

pa.section_heading("Closest Profile Matches")
if results.empty:
    if len(candidate_pool) == 0:
        st.info(
            "No candidates meet the selected pool, minutes and metric-coverage filters -- the Minimum Minutes filter "
            f"(currently {min_minutes}) is excluding every candidate. This is common early in a season when even the "
            "most-used players have played relatively few minutes; try lowering Minimum Minutes or picking a season "
            "that's further along."
        )
    else:
        st.info("No candidates meet the selected pool, minutes and metric-coverage filters.")
    st.stop()

top_results = results.head(top_n).copy()
display_cols = [
    "Rank",
    "Similarity",
    "Player",
    "Team",
    "Position",
    "Role Group",
    "Minutes",
    "Age",
    "Foot",
    "Nationality",
    "Metric Coverage %",
    "Closest Matching Metrics",
    "Biggest Profile Gaps",
]
st.dataframe(
    top_results[display_cols],
    width="stretch",
    hide_index=True,
    column_config={
        "Similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=100, format="%.1f"),
        "Metric Coverage %": st.column_config.ProgressColumn("Metric Coverage", min_value=0, max_value=100, format="%.0f%%"),
        "Minutes": st.column_config.NumberColumn("Minutes", format="%.0f"),
        "Age": st.column_config.NumberColumn("Age", format="%.1f"),
    },
)

best_cards = top_results.head(3)
card_cols = st.columns(3)
for col, (_, row) in zip(card_cols, best_cards.iterrows()):
    with col:
        st.markdown(
            f"""
            <div class="pa-card">
                <div class="pa-card-icon">Match #{int(row['Rank'])}</div>
                <div class="pa-card-title">{ui.esc(row['Player'])}</div>
                <div class="pa-card-body">
                    {ui.esc(row['Team'])} | {ui.esc(row['Position'])}<br>
                    Similarity: {row['Similarity']:.1f}<br>
                    Closest: {ui.esc(row['Closest Matching Metrics'])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Profile", key=f"similar_open_{pa.safe_key(row['Player'])}_{int(row['_Index'])}"):
            _open_profile(str(row["Player"]))

st.plotly_chart(_similarity_bar(top_results, "Similarity Shortlist"), width="stretch")

pa.section_heading("Why These Players Match")
tabs = st.tabs(["Profile Comparison", "Gap Map", "Metric Breakdown"])

comparison_options = top_results["_Index"].tolist()
comparison_index = tabs[0].selectbox(
    "Compare Against Reference",
    comparison_options,
    format_func=lambda idx: str(players.at[idx, "Player"]),
    key="similar_compare_player",
)
comparison_name = str(players.at[comparison_index, "Player"])

with tabs[0]:
    st.plotly_chart(
        _comparison_radar(
            percentiles,
            reference_index,
            comparison_index,
            selected_metrics,
            str(reference.get("Player", "Reference")),
            comparison_name,
        ),
        width="stretch",
    )

with tabs[1]:
    st.caption("Red means the candidate ranks higher than the reference on that metric; blue means lower. For lower-is-better metrics, percentiles are already direction-adjusted.")
    st.plotly_chart(_gap_heatmap(top_results, percentiles, reference_index, selected_metrics), width="stretch")

with tabs[2]:
    breakdown = _metric_breakdown(players, percentiles, reference_index, comparison_index, selected_metrics)
    if breakdown.empty:
        st.info("No shared metric values are available for this comparison.")
    else:
        st.dataframe(
            breakdown,
            width="stretch",
            hide_index=True,
            column_config={
                "Reference Percentile": st.column_config.NumberColumn("Reference Percentile", format="%.1f"),
                "Comparison Percentile": st.column_config.NumberColumn("Comparison Percentile", format="%.1f"),
                "Gap": st.column_config.NumberColumn("Gap", format="%+.1f"),
            },
        )
