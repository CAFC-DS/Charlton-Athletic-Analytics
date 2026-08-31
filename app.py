# =============================================================================
# APP ENTRY POINT - run this file with: streamlit run app.py
# =============================================================================
# Wires every page into the sidebar, grouped as Team Analysis, Player Analysis,
# Match Analysis, and Data Hub.
# =============================================================================

import streamlit as st

from utils import auth, data, ui


page_icon = str(ui.BADGE_PATH) if ui.BADGE_PATH.exists() else ":material/shield:"
st.set_page_config(page_title="Charlton Analytics", page_icon=page_icon, layout="wide")

ui.apply_statsearch_theme()
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] a[href$="substitution-impact"] {
        pointer-events: none !important;
        opacity: 0.38 !important;
        filter: grayscale(1) !important;
        text-decoration: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

AUTHENTICATED_KEY = "cafc_authenticated"
AUTH_REVISION_KEY = "cafc_auth_revision"


def clear_cafc_auth_session() -> None:
    for key in (
        AUTHENTICATED_KEY,
        AUTH_REVISION_KEY,
        "cafc_login_username",
        "cafc_login_password",
        "cafc_login_username_v2",
        "cafc_login_password_v2",
    ):
        st.session_state.pop(key, None)


def require_cafc_login() -> None:
    try:
        auth_config = auth.load_auth_config(st.secrets)
        config_error = None
    except auth.AuthConfigError as exc:
        auth_config = None
        config_error = str(exc)

    if auth_config is not None:
        authenticated = st.session_state.get(AUTHENTICATED_KEY, False)
        session_revision = st.session_state.get(AUTH_REVISION_KEY)
        if authenticated and session_revision == auth_config.revision:
            return

        # A password change must invalidate sessions authenticated with the old one.
        if authenticated or session_revision is not None:
            clear_cafc_auth_session()
    else:
        clear_cafc_auth_session()

    st.markdown(
        """
        <style>
        .cafc-login-brand {
            max-width: 560px;
            margin: 7vh auto 0;
            padding: 30px 28px 20px;
            text-align: center;
            color: #ffffff;
            border-top: 6px solid #c30017;
            border-radius: 14px 14px 0 0;
            background: linear-gradient(135deg, #111111 0%, #4b1118 58%, #9c0214 145%);
            box-shadow: 0 14px 34px rgba(16, 24, 40, 0.18);
        }

        .cafc-login-badge {
            width: 92px;
            height: 92px;
            object-fit: contain;
            margin-bottom: 12px;
        }
        .cafc-login-eyebrow {
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.12em;
        }

        .cafc-login-title {
            margin-top: 8px;
            font-size: 2rem;
            font-weight: 900;
        }

        .cafc-login-subtitle {
            margin-top: 8px;
            color: rgba(255, 255, 255, 0.78);
        }

        div[data-testid="stForm"] {
            border: 1px solid #d8dde6;
            border-top: 0;
            border-radius: 0 0 14px 14px;
            padding: 1rem 1.5rem 1.5rem;
            background: #ffffff;
            box-shadow: 0 14px 34px rgba(16, 24, 40, 0.10);
        }

        div[data-testid="stForm"] [data-testid="InputInstructions"] {
            display: none !important;
        }

        .cafc-login-note {
            margin-top: 14px;
            text-align: center;
            color: #667085;
            font-size: 0.82rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, login_col, _ = st.columns([1, 1.1, 1])

    with login_col:
        st.markdown(
            f"""
            <div class="cafc-login-brand">
                {ui.badge_html("cafc-login-badge", "Charlton Athletic crest")}
                <div class="cafc-login-eyebrow">CHARLTON ATHLETIC</div>
                <div class="cafc-login-title">CAFC Analytics</div>
                <div class="cafc-login-subtitle">Private analyst access</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if auth_config is None:
            st.error("Login is unavailable because authentication is not configured correctly.")
            st.caption(
                f"Owner diagnostics: {config_error}. Add quoted username and password "
                f"values under [auth] in Streamlit secrets, then reboot the app."
            )
        else:
            with st.form("cafc_login_form_v2", clear_on_submit=True):
                # This is a single-account app. Keeping the configured username fixed
                # prevents placeholder text from being mistaken for an entered value.
                st.text_input(
                    "Username",
                    value=auth_config.username,
                    disabled=True,
                    key="cafc_login_username_v2",
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    key="cafc_login_password_v2",
                )
                submitted = st.form_submit_button(
                    "Sign in",
                    type="primary",
                    width="stretch",
                )

            if submitted:
                if not password.strip():
                    st.warning("Enter your password.")
                elif auth.credentials_match(auth_config, auth_config.username, password):
                    st.session_state[AUTHENTICATED_KEY] = True
                    st.session_state[AUTH_REVISION_KEY] = auth_config.revision
                    st.rerun()
                else:
                    st.error("Incorrect password.")

        st.markdown(
            '<div class="cafc-login-note">Authorised CAFC analysts only.</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{auth.AUTH_BUILD} · secrets {'loaded' if auth_config else 'not loaded'}")

    st.stop()

require_cafc_login()

if ui.BADGE_PATH.exists():
    st.sidebar.markdown(
        f'<div class="cafc-sidebar-brand">{ui.badge_html("cafc-sidebar-badge", "Charlton Athletic crest")}</div>',
        unsafe_allow_html=True,
    )

if st.sidebar.button("Log out", icon=":material/logout:", width="stretch"):
    clear_cafc_auth_session()
    st.rerun()

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
    st.Page(
        "views/substitution_impact.py",
        title="Substitution Impact",
        icon=":material/construction:",
        url_path="substitution-impact",
    ),
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

# Search page titles and navigation sections from a native fuzzy-search box.
# Work-in-progress pages remain visible in navigation but are excluded here.
disabled_search_pages = {"substitution-impact"}
page_search_options = {}

for section, pages in {
    "Home": [home],
    "Team Analysis": team_pages,
    "Player Analysis": player_pages,
    "Match Analysis": match_pages,
    "Data Hub": data_hub_pages,
}.items():
    for page in pages:
        if page.url_path not in disabled_search_pages:
            page_search_options[f"{page.title} · {section}"] = page


def open_page_from_search() -> None:
    selected_label = st.session_state.get("platform_page_search")
    if not selected_label:
        return

    selected_page = page_search_options[selected_label]
    st.session_state["platform_page_search"] = None
    st.switch_page(selected_page)


st.sidebar.selectbox(
    "Search platform",
    options=list(page_search_options),
    index=None,
    placeholder="Search pages and tools...",
    key="platform_page_search",
    on_change=open_page_from_search,
    filter_mode="fuzzy",
    width="stretch",
)

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
