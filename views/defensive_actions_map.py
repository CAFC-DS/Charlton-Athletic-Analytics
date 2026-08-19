# =============================================================================
# DEFENSIVE ACTIONS MAP - real Impect defensive locations
# =============================================================================
import pandas as pd
import streamlit as st

from utils import data
from utils import match_analysis as ma
from utils import pitch


DEFENSIVE_ACTION_TYPES = ["LOOSE_BALL_REGAIN", "INTERCEPTION", "CLEARANCE", "BLOCK", "GROUND_DUEL", "REFEREE_INTERCEPTION"]
SECOND_BALL_WIN_ACTION_TYPES = ["LOOSE_BALL_REGAIN", "INTERCEPTION", "GROUND_DUEL", "GK_CATCH"]


def _category_label(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return "Unknown"
    return text.replace("_", " ").title()


def _defensive_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()

    actions = events.dropna(subset=["Start X", "Start Y"]).copy()
    if actions.empty:
        return actions

    action_type = actions["Action Type"].fillna("").astype(str).str.upper()
    phase = actions["Phase"].fillna("").astype(str).str.upper() if "Phase" in actions else pd.Series("", index=actions.index)
    result = actions["Result"].fillna("").astype(str).str.upper() if "Result" in actions else pd.Series("", index=actions.index)

    base_defensive = action_type.isin(DEFENSIVE_ACTION_TYPES)
    second_ball_win = phase.eq("SECOND_BALL") & (
        action_type.isin(SECOND_BALL_WIN_ACTION_TYPES)
        | (action_type.eq("DRIBBLE") & result.eq("SUCCESS"))
    )
    actions = actions[base_defensive | second_ball_win].copy()
    if actions.empty:
        return actions

    actions["Defensive Category"] = action_type.loc[actions.index].map(_category_label)
    actions.loc[second_ball_win.loc[actions.index], "Defensive Category"] = "Second Ball Win"
    actions["Result Label"] = actions["Result"].where(actions["Result"].notna(), "No Result")
    actions["Result Label"] = actions["Result Label"].astype(str).replace({"None": "No Result", "nan": "No Result"})
    return actions


ma.page_header(
    "Defensive Actions Map",
    "Plot defensive action and second-ball-win locations for a selected fixture and team.",
    "CAFC_DB Impect provider events supply adjusted coordinates, action labels, phase labels and result values through the app's event adapter.",
)

season = ma.select_match_season(key="def_map_season")
matches = ma.load_matches(season)
if matches.empty:
    st.warning("No match data is available for this season.")
    st.stop()

match_row = ma.match_selector(matches, key="def_map_match")
team_name = ma.team_selector_for_match(match_row, key="def_map_team")
events = data.load_match_events(
    season=season,
    match_id=match_row.get("MatchId"),
    team=team_name,
    limit=20000,
)
actions = _defensive_events(events)

ma.section_heading("Selected fixture summary")
score = ma.team_match_summary(match_row, team_name)
second_ball_wins = int((actions["Defensive Category"].astype(str) == "Second Ball Win").sum()) if not actions.empty else 0
metric_cols = st.columns(5)
metric_cols[0].metric("Opponent", str(score["Opponent"]))
metric_cols[1].metric("Goals against", ma.metric_value(score["Goals Against"]))
metric_cols[2].metric("Defensive actions", len(actions))
metric_cols[3].metric("Second ball wins", second_ball_wins)
metric_cols[4].metric("Action categories", actions["Defensive Category"].nunique() if not actions.empty else 0)

ma.section_heading("Defensive filters")
if actions.empty:
    st.info("No defensive event locations are available for this selected match and team.")
    st.stop()

control_cols = st.columns(2)
action_types = sorted(actions["Defensive Category"].dropna().astype(str).unique().tolist())
results = sorted(actions["Result Label"].dropna().astype(str).unique().tolist())
selected_action_types = control_cols[0].multiselect("Action categories", action_types, default=action_types)
selected_results = control_cols[1].multiselect("Results", results, default=results)

filtered = actions.copy()
if selected_action_types:
    filtered = filtered[filtered["Defensive Category"].astype(str).isin(selected_action_types)]
if selected_results:
    filtered = filtered[filtered["Result Label"].astype(str).isin(selected_results)]

ma.section_heading(f"{team_name} defensive map")
if filtered.empty:
    st.info("No defensive actions match the current filters.")
else:
    st.plotly_chart(pitch.defensive_action_map(filtered, team_name, f"{team_name}: Defensive Action Locations"), width="stretch")
st.caption(
    "Second ball wins are derived from SECOND_BALL phase events where the selected team regains or wins the loose ball "
    "(loose-ball regains, interceptions, ground duels, goalkeeper catches, or successful dribbles)."
)

ma.section_heading("Defensive event table")
display_cols = ma.available_columns(
    filtered,
    ["Minute", "Player", "Defensive Category", "Action Type", "Action", "Result Label", "Phase", "Pressure", "Start Lane", "Start Pitch Position", "Start X", "Start Y"],
)
st.dataframe(filtered[display_cols].sort_values(["Minute", "Defensive Category"]), width="stretch", hide_index=True)
