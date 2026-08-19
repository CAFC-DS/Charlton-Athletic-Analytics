# =============================================================================
# PASS MAP - real Impect pass start/end locations
# =============================================================================
import pandas as pd
import streamlit as st

from utils import data
from utils import match_analysis as ma
from utils import pitch
from utils import ui


def _pass_map_css() -> None:
    st.markdown(
        """
        <style>
        .pm-summary-card {
            background: var(--ss-panel);
            border: 1px solid var(--ss-border);
            border-radius: 8px;
            border-top: 3px solid var(--ss-accent);
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-height: 102px;
            padding: 14px 16px;
        }

        .pm-summary-label {
            color: var(--ss-muted);
            font-size: 0.875rem;
            font-weight: 650;
            line-height: 1.25;
            margin-bottom: 14px;
        }

        .pm-summary-value {
            color: var(--ss-ink);
            display: block;
            font-size: clamp(1.5rem, 1.9vw, 1.9rem);
            font-weight: 400;
            letter-spacing: -0.03em;
            line-height: 1.1;
        }

        .pm-summary-value-text {
            font-size: clamp(0.80rem, 0.94vw, 0.96rem);
            letter-spacing: -0.01em;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _summary_card(label: str, value: object, *, text_value: bool = False) -> None:
    value_class = "pm-summary-value pm-summary-value-text" if text_value else "pm-summary-value"
    st.markdown(
        f"""
        <div class="pm-summary-card">
            <div class="pm-summary-label">{ui.esc(label)}</div>
            <div class="{value_class}">{ui.esc(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


ma.page_header(
    "Pass Map",
    "Plot selected-match pass start and end locations using adjusted Impect coordinates.",
    "CAFC_DB Impect provider events supply pass actions, start/end coordinates, receivers, outcomes, distances and pass PXT values.",
)
_pass_map_css()

season = ma.select_match_season(key="pass_map_season")
matches = ma.load_matches(season)
if matches.empty:
    st.warning("No match data is available for this season.")
    st.stop()

match_row = ma.match_selector(matches, key="pass_map_match")
team_name = ma.team_selector_for_match(match_row, key="pass_map_team")
events = data.load_match_events(
    season=season,
    match_id=match_row.get("MatchId"),
    team=team_name,
    action_types=["PASS"],
    limit=2500,
)
passes = events.dropna(subset=["Start X", "Start Y", "End X", "End Y"]).copy()

ma.section_heading("Selected fixture summary")
completed = int((passes["Result"].astype(str).str.upper() == "SUCCESS").sum()) if not passes.empty else 0
failed = int((passes["Result"].astype(str).str.upper() == "FAIL").sum()) if not passes.empty else 0
completion = completed / (completed + failed) * 100 if completed + failed else 0
cross_mask = data.is_cross(passes)
crosses_attempted = int(cross_mask.sum())
crosses_completed = int((cross_mask & passes["Result"].astype(str).str.upper().eq("SUCCESS")).sum()) if not passes.empty else 0
metric_cols = st.columns(5)
with metric_cols[0]:
    _summary_card("Fixture", str(match_row.get("Match", "Unknown")), text_value=True)
with metric_cols[1]:
    _summary_card("Passes plotted", len(passes))
with metric_cols[2]:
    _summary_card("Completion", f"{completion:.1f}%")
with metric_cols[3]:
    _summary_card("Total PXT pass", ma.metric_value(passes["PXT Pass"].sum() if not passes.empty else 0, "PXT Pass"))
with metric_cols[4]:
    _summary_card("Crosses", f"{crosses_completed}/{crosses_attempted} completed", text_value=True)

ma.section_heading("Pass map controls")
control_cols = st.columns(4)
outcomes = sorted(passes["Result"].dropna().astype(str).unique().tolist()) if not passes.empty else []
selected_outcomes = control_cols[0].multiselect("Outcomes", outcomes, default=outcomes)
min_distance = control_cols[1].number_input("Minimum distance", min_value=0.0, value=0.0, step=5.0)
pass_type = control_cols[2].selectbox("Pass type", ["All Passes", "Crosses Only", "Crosses Excluded"])
max_passes = control_cols[3].slider("Maximum plotted passes", 50, 700, min(450, max(len(passes), 50)), step=50)
control_cols[3].caption(
    "When the selection exceeds this limit, the map keeps the highest PXT Passes first. "
    "If PXT Pass is unavailable, it falls back to longest pass distance."
)

filtered = passes.copy()
if selected_outcomes:
    filtered = filtered[filtered["Result"].astype(str).isin(selected_outcomes)]
if "Pass Distance" in filtered:
    filtered = filtered[pd.to_numeric(filtered["Pass Distance"], errors="coerce").fillna(0) >= min_distance]
if pass_type == "Crosses Only":
    filtered = filtered[data.is_cross(filtered)]
elif pass_type == "Crosses Excluded":
    filtered = filtered[~data.is_cross(filtered)]

map_title_suffix = {"Crosses Only": "Cross Delivery Map", "Crosses Excluded": "Non-Cross Pass Map"}.get(pass_type, "Pass Start/End Map")
ma.section_heading(f"{team_name} Pass Map")
if filtered.empty:
    st.info("No pass locations match the current filters.")
else:
    st.plotly_chart(pitch.pass_map(filtered, team_name, f"{team_name}: {map_title_suffix}", max_passes=max_passes), width="stretch")

ma.section_heading("Pass event table")
display_cols = ma.available_columns(
    filtered,
    ["Minute", "Player", "Receiver", "Action", "Result", "Pass Distance", "PXT Pass", "Start Lane", "End Lane", "Start X", "Start Y", "End X", "End Y"],
)
if filtered.empty:
    st.caption("No pass rows are available for the current selection.")
else:
    st.dataframe(filtered[display_cols].sort_values(["Minute", "Player"]), width="stretch", hide_index=True)
