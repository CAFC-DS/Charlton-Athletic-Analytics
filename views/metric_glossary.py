# =============================================================================
# DATA HUB · METRIC GLOSSARY — for football analysts and other stakeholders
# =============================================================================
# Two parts:
#   1. "Used in this app" — the exact metrics wired into data.py, tied to
#      PLAYER_METRICS / TEAM_METRICS / MATCH_METRICS so it can't drift out
#      of sync with what the app actually shows.
#   2. A broader football-analytics glossary — terms a stakeholder would
#      want defined even if this app doesn't compute them directly, grouped
#      by category with a filter (same polish as Cleaned Tables).
# =============================================================================
import pandas as pd
import streamlit as st
from utils import data

st.title("📖 Metric Glossary")
st.caption("Look up what a metric or football-analytics term means before reading too much into a number.")

DEFINITIONS = {
    "Goals /90": "Goals scored, normalised to a rate per 90 minutes played (a squad player with few minutes can show a fractional value like 0.57).",
    "Assists /90": "Passes that directly led to a goal, per 90 minutes played.",
    "Bypassed Opponents /90": "Impect packing metric — opponents taken out of the game by a pass, carry or dribble, per 90 minutes. A proxy for progression and creativity.",
    "Pass %": "Percentage of attempted passes that reach a teammate.",
    "Passes to Final 3rd /90": "Passes into the attacking third, per 90 minutes played.",
    "Home Goals": "Goals scored by the provider-identified home team in a match (an actual match total, not a per-90 rate).",
    "Away Goals": "Goals scored by the provider-identified away team in a match (an actual match total, not a per-90 rate).",
}

sections = {
    "Player Analysis": data.PLAYER_METRICS,
    "Team Analysis": data.TEAM_METRICS,
    "Match Analysis": data.MATCH_METRICS,
}

st.subheader("Used in this app")
rows = []
for section, metrics in sections.items():
    for metric in metrics:
        rows.append({
            "Metric": metric,
            "Used in": section,
            "Definition": DEFINITIONS.get(metric, "No definition yet."),
        })
app_glossary = pd.DataFrame(rows)
st.dataframe(app_glossary, use_container_width=True, hide_index=True)

undefined = app_glossary.loc[app_glossary["Definition"] == "No definition yet.", "Metric"].unique()
if len(undefined):
    st.warning(f"Missing definitions for: {', '.join(undefined)}", icon="⚠️")

st.divider()

# ---- Wider football-analytics glossary ---------------------------------------
st.subheader("Football analytics glossary")
st.write("Broader terms analysts, coaches and other stakeholders commonly use — not all are computed in this app yet.")

GLOSSARY = [
    # Attacking & Finishing
    ("xG (Expected Goals)", "Attacking & Finishing", "The likelihood a shot results in a goal, based on shot location, angle, body part and defensive pressure at the moment of the shot."),
    ("xA (Expected Assists)", "Attacking & Finishing", "The likelihood a completed pass becomes a goal assist, based on the quality of the chance it creates."),
    ("Big Chance", "Attacking & Finishing", "A shooting opportunity a player would be expected to score from in most situations."),
    ("Shot Quality", "Attacking & Finishing", "A general term for how good a scoring chance was, usually approximated by its xG value."),
    ("Non-Penalty Goals", "Attacking & Finishing", "Goals scored excluding penalties — used to judge open-play finishing separately from penalty-taking."),
    # Possession & Passing
    ("Possession %", "Possession & Passing", "Share of total match time a team spends in control of the ball."),
    ("Field Tilt %", "Possession & Passing", "Share of final-third possession between two teams — a proxy for territorial dominance that isn't skewed by harmless side-to-side passing."),
    ("Progressive Pass", "Possession & Passing", "A pass that moves the ball meaningfully closer to the opponent's goal, regardless of whether the move succeeds."),
    ("Progressive Carry", "Possession & Passing", "A dribble or run that advances the ball meaningfully upfield."),
    ("Key Pass", "Possession & Passing", "A pass that directly leads to a shot."),
    ("Line-Breaking Pass", "Possession & Passing", "A pass that goes through or past a line of opposition players, disrupting their defensive or midfield structure."),
    ("Final Third Entry", "Possession & Passing", "A pass or carry that brings the ball into the attacking third of the pitch."),
    ("Switch of Play", "Possession & Passing", "A long pass that shifts the ball from one side of the pitch to the other, usually to exploit space."),
    # Defending
    ("PPDA (Passes Per Defensive Action)", "Defending", "The number of opposition passes allowed before a team makes a tackle, interception, foul or challenge — lower values mean more intense pressing."),
    ("Tackle", "Defending", "An attempt to win the ball from an opponent in a physical duel."),
    ("Interception", "Defending", "Reading and cutting out an opposition pass before it reaches its target."),
    ("Ground Duel", "Defending", "A physical contest for the ball with both players on the ground, as opposed to an aerial duel."),
    ("Aerial Duel", "Defending", "A physical contest for the ball in the air, usually from a cross, long pass or clearance."),
    ("Recovery", "Defending", "Regaining possession of the ball, whether through a tackle, interception or picking up a loose ball."),
    ("Clearance", "Defending", "A defensive action to remove the ball from a dangerous area, usually without a specific intended recipient."),
    # Packing & Possession Value
    ("Packing", "Packing & Possession Value (Impect)", "Impect's term for taking opponents 'out of the game' with an action — the more opponents bypassed, the more the action disrupts the opposition's shape."),
    ("Bypassed Opponents", "Packing & Possession Value (Impect)", "The count of opposition players taken out of the game by a single pass, carry or dribble."),
    ("Bypassed Defenders", "Packing & Possession Value (Impect)", "The same idea as Bypassed Opponents, counting only recognised defenders."),
    ("Possession Value / Threat", "Packing & Possession Value (Impect)", "A family of models that assign a scoring-chance value to every on-ball action, not just shots — used to credit build-up play as well as the final pass or shot."),
    # General Analytics Concepts
    ("/90 (Per 90 Minutes)", "General Analytics Concepts", "A stat normalised to a full match's worth of playing time, so players with different amounts of game time can be compared fairly."),
    ("Percentile Rank", "General Analytics Concepts", "Where a player or team sits relative to their peers on a given metric, from 0 (bottom) to 100 (top)."),
    ("Sample Size / Minutes Played", "General Analytics Concepts", "The amount of data behind a stat — per-90 numbers from very few minutes played can be misleading and should be treated cautiously."),
    ("Season / Iteration", "General Analytics Concepts", "Impect's term for a specific competition-and-season instance (e.g. 'Championship 25/26') that data is grouped by."),
    ("Squad", "General Analytics Concepts", "Impect's term for a club or team roster."),
]

glossary_df = pd.DataFrame(GLOSSARY, columns=["Term", "Category", "Definition"])

categories = ["All"] + sorted(glossary_df["Category"].unique())
category = st.selectbox("Category", categories)
filtered = glossary_df if category == "All" else glossary_df[glossary_df["Category"] == category]

st.dataframe(filtered.sort_values("Term").reset_index(drop=True), use_container_width=True, hide_index=True)
st.caption(f"{len(filtered)} terms shown.")
