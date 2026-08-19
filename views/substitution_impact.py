# =============================================================================
# SUBSTITUTION IMPACT - substitution and rotation proxy
# =============================================================================
import pandas as pd
import streamlit as st

from utils import match_analysis as ma


ma.page_header(
    "Substitution Impact",
    "Review substitution action labels where available and a player rotation contribution proxy for the fixture teams.",
    f"{ma.ACTION_SOURCE} {ma.PLAYER_PROXY_SOURCE}",
    "The audited CAFC_DB Impect event feed has no substitution action rows, so substitution timing and before/after impact are not currently available. Low-minute player contribution is used as a transparent rotation proxy.",
)

match_season = ma.select_match_season(key="subs_match_season")
matches = ma.load_matches(match_season)
if matches.empty:
    st.warning("No match data is available for this season.")
    st.stop()

match_row = ma.match_selector(matches, key="subs_match")
team_name = ma.team_selector_for_match(match_row, key="subs_team")
actions = ma.load_match_actions(match_season, match_row)
sub_summary = ma.action_summary(actions, ma.SUBSTITUTION_KEYWORDS)
sub_breakdown = ma.action_breakdown(actions, team_name, ma.SUBSTITUTION_KEYWORDS)

player_season = ma.select_player_season(key="subs_player_season")
players = ma.player_rows_for_match(match_row, player_season)
ratings = ma.player_rating_table(players)

ma.section_heading("Substitution labels")
if sub_summary.empty:
    st.info("No substitution action labels were found for this selected match and season.")
else:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(ma.action_bar(sub_summary, team_name, "Substitution action totals"), width="stretch")
    with c2:
        st.dataframe(sub_breakdown, width="stretch", hide_index=True)

ma.section_heading("Rotation contribution proxy")
if ratings.empty or "Minutes" not in ratings:
    st.info("Player minutes and aggregate metrics are not available for a rotation proxy.")
else:
    minutes = pd.to_numeric(ratings["Minutes"], errors="coerce")
    threshold = minutes.median()
    rotation_players = ratings[minutes <= threshold].copy()
    metric_cols = st.columns(3)
    metric_cols[0].metric("Low-minute threshold", ma.metric_value(threshold, "Minutes"))
    metric_cols[1].metric("Low-minute players", len(rotation_players))
    metric_cols[2].metric("Player season", player_season or "All")
    st.plotly_chart(ma.player_rating_bar(rotation_players, "Low-minute player contribution proxy"), width="stretch")
    display_cols = ma.available_columns(
        rotation_players,
        ["Player", "Team", "_Position Display", "Minutes", "Rating Proxy", "Goals /90", "Assists /90", "Bypassed Opponents /90"],
    )
    st.dataframe(rotation_players[display_cols], width="stretch", hide_index=True)
