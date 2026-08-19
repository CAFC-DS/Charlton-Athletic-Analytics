# =============================================================================
# APP ENTRY POINT - run this file with: streamlit run app.py
# =============================================================================
# Wires every page into the sidebar, grouped as Team Analysis, Player Analysis,
# Match Analysis, and Data Hub.
# =============================================================================
import streamlit as st

from utils import data, ui


page_icon = str(ui.BADGE_PATH) if ui.BADGE_PATH.exists() else ":material/shield:"
st.set_page_config(page_title="Charlton Analytics", page_icon=page_icon, layout="wide")

ui.apply_statsearch_theme()

if ui.BADGE_PATH.exists():
    st.sidebar.markdown(
        f'<div class="cafc-sidebar-brand">{ui.badge_html("cafc-sidebar-badge", "Charlton Athletic crest")}</div>',
        unsafe_allow_html=True,
    )

if data.USE_MOCK_DATA:
    st.sidebar.error("DEMO DATA MODE")
    st.sidebar.caption("Set CHARLTON_DATA_MODE=production to use CAFC_DB.")
else:
    st.sidebar.caption("CAFC_DB · Impect analysis · Opta F24/F7 feeds")

home = st.Page("views/home.py", title="Home", icon=":material/home:", default=True)

team_pages = [
    st.Page("views/team_overview.py", title="Team Overview", icon=":material/dashboard:"),
    st.Page("views/team_style_profile.py", title="Team Style Profile", icon=":material/analytics:"),
    st.Page("views/team_scatter.py", title="Team Scatter Graphs", icon=":material/scatter_plot:"),
    st.Page("views/team_radar.py", title="Team Radar", icon=":material/radar:"),
    st.Page("views/league_rankings.py", title="League Rankings", icon=":material/leaderboard:"),
    st.Page("views/final_third_analysis.py", title="Final Third Analysis", icon=":material/flag:"),
    st.Page("views/defensive_actions.py", title="Defensive Actions", icon=":material/shield:"),
    st.Page("views/pass_clusters.py", title="Pass Clusters", icon=":material/grain:"),
    st.Page("views/set_piece_analysis.py", title="Set Piece Analysis", icon=":material/sports_soccer:"),
    st.Page("views/team_data_table.py", title="Team Data Table", icon=":material/table:"),
]

player_pages = [
    st.Page("views/player_search.py", title="Player Search", icon=":material/person_search:"),
    st.Page("views/player_overview.py", title="Player Overview", icon=":material/dashboard:"),
    st.Page("views/player_profiles.py", title="Player Profiles", icon=":material/badge:"),
    st.Page("views/player_radar.py", title="Player Radar", icon=":material/radar:"),
    st.Page("views/player_comparison.py", title="Player Comparison", icon=":material/compare_arrows:"),
    st.Page("views/player_scatter.py", title="Player Scatter Graphs", icon=":material/scatter_plot:"),
    st.Page("views/percentile_rankings.py", title="Percentile Rankings", icon=":material/leaderboard:"),
    st.Page("views/passing_dashboard.py", title="Passing Dashboard", icon=":material/sync_alt:"),
    st.Page("views/shooting_dashboard.py", title="Shooting Dashboard", icon=":material/sports_soccer:"),
    st.Page("views/similar_player_search.py", title="Similar Player Search", icon=":material/manage_search:"),
    st.Page("views/player_build_up_involvement.py", title="Build-Up Involvement", icon=":material/timeline:"),
    st.Page("views/player_data_table.py", title="Player Data Table", icon=":material/table:"),
]

match_pages = [
    st.Page("views/match_overview.py", title="Match Overview", icon=":material/stadium:"),
    st.Page("views/set_piece_review.py", title="Set Piece Review", icon=":material/sports_soccer:"),
    st.Page("views/xg_timeline.py", title="xG Timeline", icon=":material/timeline:"),
    st.Page("views/expected_threat.py", title="Expected Threat", icon=":material/bolt:"),
    st.Page("views/game_control_momentum.py", title="Game Control / Momentum", icon=":material/query_stats:"),
    st.Page("views/shot_map.py", title="Shot Map", icon=":material/my_location:"),
    st.Page("views/pass_map.py", title="Pass Map", icon=":material/route:"),
    st.Page("views/line_breaking_actions.py", title="Line-Breaking Actions", icon=":material/conversion_path:"),
    st.Page("views/transitional_data.py", title="Transitional Data", icon=":material/compare_arrows:"),
    st.Page("views/build_up_involvement.py", title="Build-Up Involvement", icon=":material/timeline:"),
    st.Page("views/passing_network.py", title="Passing Network", icon=":material/hub:"),
    st.Page("views/phase_average_positions.py", title="Phase Average Positions", icon=":material/group_work:"),
    st.Page("views/final_third_passes_shots.py", title="Final Third / Box Entries", icon=":material/flag:"),
    st.Page("views/defensive_actions_map.py", title="Defensive Actions Map", icon=":material/shield:"),
    st.Page("views/player_match_actions.py", title="Player Match Actions", icon=":material/person:"),
    st.Page("views/player_match_ratings.py", title="Player Match Ratings", icon=":material/star:"),
    st.Page("views/substitution_impact.py", title="Substitution Impact", icon=":material/swap_horiz:"),
    st.Page("views/event_data_table.py", title="Event Data Table", icon=":material/table:"),
]

data_hub_pages = [
    st.Page("views/data_raw.py", title="Loader Output", icon=":material/database:"),
    st.Page("views/opta_data.py", title="Opta Data", icon=":material/sports_soccer:"),
    st.Page("views/cleaned_tables.py", title="Cleaned Tables", icon=":material/cleaning_services:"),
    st.Page("views/data_quality.py", title="Data Quality Checks", icon=":material/task_alt:"),
    st.Page("views/metric_glossary.py", title="Metric Glossary", icon=":material/menu_book:"),
    st.Page("views/export_data.py", title="Export Data", icon=":material/download:"),
]

pg = st.navigation(
    {
        "": [home],
        "Team Analysis": team_pages,
        "Player Analysis": player_pages,
        "Match Analysis": match_pages,
        "Data Hub": data_hub_pages,
    },
    expanded=True,
)
pg.run()
