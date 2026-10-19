# =============================================================================
# SHOT MAP - real Impect shot locations and xG
# =============================================================================
import streamlit as st

from utils import data
from utils import match_analysis as ma
from utils import pitch
from utils import ui


def _shot_map_css() -> None:
    st.markdown(
        """
        <style>
        </style>
        """,
        unsafe_allow_html=True,
    )


def _summary_card(label: str, value: object, *, text_value: bool = False) -> None:
    ui.figure_tile(label, value, text_value=text_value)


ma.page_header(
    "Shot Map",
    "Plot shot locations, outcomes and xG for a selected fixture and team.",
    "CAFC_DB Impect provider events supply shot coordinates, xG, post-shot xG, distance, angle, body part and result values.",
)
_shot_map_css()

season, matches, match_row, team_name = ma.fixture_bar("shot_map_season", "shot_map_match", "shot_map_team")
events = data.load_match_events(
    season=season,
    match_id=match_row.get("MatchId"),
    team=team_name,
    action_types=["SHOT"],
    limit=600,
)
shots = events.dropna(subset=["Start X", "Start Y"]).copy()

ma.section_heading("Selected fixture summary")
goals = int(
    (
        (shots["Result"].astype(str).str.upper() == "SUCCESS")
        | (shots["Action"].astype(str).str.upper() == "GOAL")
    ).sum()
) if not shots.empty else 0
metric_cols = st.columns(5)
with metric_cols[0]:
    _summary_card("Score", f"{match_row['Home Goals']:.0f} - {match_row['Away Goals']:.0f}")
with metric_cols[1]:
    _summary_card("Team", team_name, text_value=True)
with metric_cols[2]:
    _summary_card("Shots", len(shots))
with metric_cols[3]:
    _summary_card("Goals", goals)
with metric_cols[4]:
    _summary_card("xG", ma.metric_value(shots["Shot xG"].sum() if not shots.empty else 0, "Shot xG"))

ma.section_heading("Shot filters")
if shots.empty:
    st.info("No shot locations are available for this selected match and team.")
    st.stop()

control_cols = st.columns(3)
body_parts = sorted(shots["Body Part"].dropna().astype(str).unique().tolist())
actions = sorted(shots["Action"].dropna().astype(str).unique().tolist())
selected_body_parts = control_cols[0].multiselect("Body parts", body_parts, default=body_parts)
selected_actions = control_cols[1].multiselect("Shot types", actions, default=actions)
min_xg = control_cols[2].number_input("Minimum xG", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

filtered = shots.copy()
if selected_body_parts:
    filtered = filtered[filtered["Body Part"].astype(str).isin(selected_body_parts)]
if selected_actions:
    filtered = filtered[filtered["Action"].astype(str).isin(selected_actions)]
filtered = filtered[filtered["Shot xG"].fillna(0) >= min_xg]

ma.section_heading(f"{team_name} Shot Map")
if filtered.empty:
    st.info("No shots match the current filters.")
else:
    st.plotly_chart(pitch.shot_map_half_pitch(filtered, team_name, f"{team_name}: Shot Map and xG"), width="stretch")
    st.caption(
        "Every shot is normalised toward the goal at the top. Chart left and right are the attacker's perspective; "
        "Impect's adjusted coordinates already account for the team changing ends between halves."
    )

ma.section_heading("Goalmouth: Where Shots Were Aimed")
st.caption(
    "Plotted from the shooter's view using each shot's target coordinates on the goal face, coloured by player. "
    "Marker size is Post-Shot xG (falls back to pre-shot xG when a shot wasn't on target). Only shots with a "
    "recorded target location can be placed here, so the shot count may be lower than the map above."
)
if filtered.empty:
    st.info("No shots match the current filters.")
else:
    goalmouth_players = sorted(filtered["Player"].dropna().astype(str).unique().tolist())
    st.plotly_chart(
        pitch.goalmouth_shot_map(
            filtered,
            f"{team_name}: Goalmouth Shot Placement",
            group_col="Player",
            group_order=goalmouth_players,
            height=620,
        ),
        width="stretch",
    )

ma.section_heading("Shot event table")
display_cols = ma.available_columns(
    filtered,
    ["Minute", "Player", "Action", "Body Part", "Result", "Shot xG", "Post-Shot xG", "Shot Distance", "Shot Angle", "Start X", "Start Y"],
)
st.dataframe(filtered[display_cols].sort_values("Minute"), width="stretch", hide_index=True)
