# =============================================================================
# PLAYER PROFILES - role-aware player profile and radar
# =============================================================================
import pandas as pd
import streamlit as st

from utils import player_analysis as pa
from utils import ui


def _selected_from_state() -> str | None:
    query_player = st.query_params.get("player")
    if query_player:
        return str(query_player)
    selected = st.session_state.get("selected_player")
    return str(selected) if selected else None


def _player_sort_name(row: pd.Series) -> str:
    last_name = row.get("Last Name")
    if pd.notna(last_name) and str(last_name).strip():
        return str(last_name).strip().casefold()
    player_name = str(row.get("Player", "")).strip()
    return player_name.split()[-1].casefold() if player_name else ""


def _sorted_player_options(players: pd.DataFrame, sort_mode: str) -> list[int]:
    sortable = players.copy()
    sortable["_sort_last"] = sortable.apply(_player_sort_name, axis=1)
    sortable["_sort_player"] = sortable["Player"].astype(str).str.casefold()
    sortable["_sort_team"] = sortable["Team"].astype(str).str.casefold() if "Team" in sortable else ""
    if sort_mode == "Club A-Z":
        sort_cols = ["_sort_team", "_sort_last", "_sort_player"]
    else:
        sort_cols = ["_sort_last", "_sort_player", "_sort_team"]
    return sortable.sort_values(sort_cols, kind="mergesort").index.tolist()


def _profile_css() -> None:
    st.markdown(
        """
        <style>
        .profile-shell {
            background: #ffffff;
            border: 1px solid #d8dde6;
            border-radius: 8px;
            padding: 20px 22px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            margin: 8px 0 18px;
        }

        .profile-topline {
            color: #667085;
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .profile-name-row {
            align-items: flex-start;
            display: flex;
            gap: 18px;
            justify-content: space-between;
        }

        .profile-name {
            color: #111111;
            font-size: clamp(1.8rem, 3vw, 2.45rem);
            font-weight: 850;
            line-height: 1.05;
            margin: 0 0 10px;
        }

        .profile-meta {
            color: #475467;
            display: flex;
            flex-wrap: wrap;
            gap: 6px 10px;
            font-size: 0.82rem;
            line-height: 1.35;
            max-width: 100%;
        }

        .profile-meta span {
            border-right: 1px solid #d8dde6;
            padding-right: 10px;
            overflow-wrap: anywhere;
        }

        .profile-meta span:last-child {
            border-right: 0;
        }

        .profile-score {
            background: #111111;
            border-radius: 8px;
            color: #ffffff;
            min-width: 120px;
            padding: 12px 14px;
            text-align: center;
        }

        .profile-score-label {
            color: rgba(255, 255, 255, 0.72);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .profile-score-value {
            font-size: 1.82rem;
            font-weight: 850;
            line-height: 1;
            margin-top: 6px;
        }

        .profile-score-role {
            color: rgba(255, 255, 255, 0.78);
            font-size: 0.8rem;
            margin-top: 5px;
        }

        .profile-tile-grid {
            display: grid;
            gap: 10px;
            grid-template-columns: repeat(auto-fit, minmax(136px, 1fr));
            margin-top: 18px;
        }

        .profile-tile {
            background: #f8fafc;
            border: 1px solid #e6edf5;
            border-radius: 8px;
            min-height: 74px;
            padding: 10px 12px;
        }

        .profile-tile-label {
            color: #667085;
            font-size: 0.62rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            line-height: 1.18;
            text-transform: uppercase;
            overflow-wrap: anywhere;
        }

        .profile-tile-value {
            color: #111111;
            font-size: 1.04rem;
            font-weight: 820;
            line-height: 1.18;
            margin-top: 4px;
            overflow-wrap: anywhere;
        }

        .profile-note {
            color: #667085;
            font-size: 0.84rem;
            line-height: 1.45;
            margin-top: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _tile(label: str, value: object) -> str:
    return (
        '<div class="profile-tile">'
        f'<div class="profile-tile-label">{ui.esc(label)}</div>'
        f'<div class="profile-tile-value">{ui.esc(value)}</div>'
        "</div>"
    )


def _profile_header(row: pd.Series, role: str, score: object, metric_rows: pd.DataFrame) -> None:
    score_value = "N/A" if pd.isna(score) else f"{float(score):.0f}"
    strongest = metric_rows.sort_values("Role Percentile", ascending=False).iloc[0] if not metric_rows.empty else None
    strongest_text = strongest["Radar Label"] if strongest is not None else "N/A"
    meta_items = [
        f"Team: {row.get('Team', 'Unknown')}",
        f"Position: {row.get('_Position Display', 'Unknown')}",
        f"Role: {role}",
        f"Age: {pa.format_age(row.get('Birthdate'))}",
        f"Nationality: {row.get('Nationality', 'Unknown')}",
        f"Foot: {row.get('Foot', 'Unknown')}",
        f"Competition: {row.get('Competition', 'Unknown')}",
    ]
    meta_html = "".join(f"<span>{ui.esc(item)}</span>" for item in meta_items if item and "nan" not in str(item).lower())
    goals = pd.to_numeric(pd.Series([row.get("Goals /90")]), errors="coerce").iloc[0]
    assists = pd.to_numeric(pd.Series([row.get("Assists /90")]), errors="coerce").iloc[0]
    goal_contribution = (0 if pd.isna(goals) else goals) + (0 if pd.isna(assists) else assists)
    tiles = [
        _tile("Minutes", pa.metric_value(row.get("Minutes"), "Minutes")),
        _tile("Best profile area", strongest_text),
    ]
    if role == "Goalkeeper":
        tiles.extend(
            [
                _tile("Goals prevented /90", pa.metric_value(row.get("Goals Prevented /90"), "Goals Prevented /90")),
                _tile("Save actions /90", pa.metric_value(row.get("Save Actions /90"), "Save Actions /90")),
                _tile("Pass completion", pa.metric_value(row.get("Pass %"), "Pass %")),
                _tile("Pass progression /90", pa.metric_value(row.get("Pass Progression /90"), "Pass Progression /90")),
            ]
        )
    else:
        tiles.extend(
            [
                _tile("Goals + assists /90", pa.metric_value(goal_contribution, "Goals /90")),
                _tile("Pass completion", pa.metric_value(row.get("Pass %"), "Pass %")),
                _tile("Progression /90", pa.metric_value(row.get("Bypassed Opponents /90"), "Bypassed Opponents /90")),
                _tile("Ball wins /90", pa.metric_value(row.get("Ball Wins /90"), "Ball Wins /90")),
            ]
        )
    note = (
        "Goalkeeper score combines shot-stopping and distribution percentile axes against goalkeeper peers."
        if role == "Goalkeeper"
        else "Role score is the average of the selected role-specific percentile axes. Percentiles compare the player to the closest available positional peer group, with an all-outfield fallback where needed."
    )
    html = (
        '<div class="profile-shell">'
        '<div class="profile-name-row">'
        "<div>"
        '<div class="profile-topline">01 / Scouting Details</div>'
        f'<h2 class="profile-name">{ui.esc(row.get("Player", "Unknown player"))}</h2>'
        f'<div class="profile-meta">{meta_html}</div>'
        "</div>"
        '<div class="profile-score">'
        '<div class="profile-score-label">Role Score</div>'
        f'<div class="profile-score-value">{ui.esc(score_value)}</div>'
        f'<div class="profile-score-role">{ui.esc(role)}</div>'
        "</div>"
        "</div>"
        f'<div class="profile-tile-grid">{"".join(tiles)}</div>'
        f'<div class="profile-note">{ui.esc(note)}</div>'
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _metric_table(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    if df.empty:
        st.caption("No metrics are available for this selection.")
        return
    out = df.copy()
    out["Role Percentile"] = pd.to_numeric(out["Role Percentile"], errors="coerce").round(0)
    out["Overall Percentile"] = pd.to_numeric(out["Overall Percentile"], errors="coerce").round(0)
    out["Role Rank"] = out["Role Rank"].apply(lambda value: "N/A" if pd.isna(value) else f"{int(value)}")
    display_cols = columns or ["Category", "Metric", "Display Value", "Role Percentile", "Overall Percentile", "Role Rank"]
    st.dataframe(
        out[display_cols],
        width="stretch",
        hide_index=True,
        column_config={
            "Display Value": st.column_config.TextColumn("Value"),
            "Role Percentile": st.column_config.ProgressColumn("Role pct", min_value=0, max_value=100, format="%.0f"),
            "Overall Percentile": st.column_config.ProgressColumn("Overall pct", min_value=0, max_value=100, format="%.0f"),
        },
    )


pa.page_header(
    "Player Profiles",
    "Role-aware player profile with headline details, clean percentile radar, standard stats and category breakdowns.",
)
_profile_css()

season = pa.select_season(key="player_profiles_season")
players = pa.load_player_data(season)
if players.empty:
    st.warning("No players are available for the selected season.")
    st.stop()

players = pa.add_position_groups(players)
preferred = _selected_from_state()


def _selector_label(idx: int, sort_mode: str) -> str:
    player = players.loc[idx]
    if sort_mode == "Club A-Z":
        return f"{player.get('Team', 'Unknown')} | {player.get('Player', 'Unknown')} | {player.get('_Position Display', 'Unknown')}"
    return f"{player.get('Player', 'Unknown')} | {player.get('Team', 'Unknown')} | {player.get('_Position Display', 'Unknown')}"


pa.section_heading("Profile selector")
selector_cols = st.columns([2.2, 0.8, 0.8])
sort_mode = selector_cols[1].selectbox(
    "Sort by",
    ["Player A-Z", "Club A-Z"],
    index=0,
    key="player_profile_sort",
)
options = _sorted_player_options(players, sort_mode)
default_index = 0
if preferred:
    found = players.index[players["Player"].astype(str) == preferred].tolist()
    if found and found[0] in options:
        default_index = options.index(found[0])
selected_index = selector_cols[0].selectbox(
    "Player", options, index=default_index, format_func=lambda idx: _selector_label(idx, sort_mode)
)
selected_player = str(players.loc[selected_index, "Player"])
st.session_state["selected_player"] = selected_player
min_minutes = selector_cols[2].number_input("Minimum minutes", min_value=0, value=0, step=250)

filtered_players = players[pd.to_numeric(players["Minutes"], errors="coerce").fillna(0) >= min_minutes].copy()
if selected_index not in filtered_players.index:
    filtered_players = pd.concat([filtered_players, players.loc[[selected_index]]]).drop_duplicates()

context = pa.player_profile_context(filtered_players, selected_player)
row = context["row"]
metric_rows = context["metrics"]
score = context["score"]

_profile_header(row, str(context["role"]), score, metric_rows)

pa.section_heading("Profile shape")
chart_col, table_col = st.columns([1.06, 1])
with chart_col:
    radar_fig = pa.player_profile_radar(metric_rows, selected_player, str(context["role"]), score)
    if radar_fig.data:
        st.plotly_chart(
            radar_fig,
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
            key=f"profile_radar_{selected_index}_{pa.safe_key(selected_player)}_{pa.safe_key(context['role'])}",
        )
    else:
        st.info("Radar data is not available for this player and role selection.")

with table_col:
    st.markdown("**Standard profile stats**")
    _metric_table(metric_rows, ["Metric", "Display Value", "Role Percentile", "Overall Percentile", "Role Rank"])

strengths, watchouts = pa.profile_strengths(metric_rows)
summary_cols = st.columns(2)
with summary_cols[0]:
    pa.section_heading("Strongest areas")
    _metric_table(strengths, ["Metric", "Display Value", "Role Percentile"])
with summary_cols[1]:
    pa.section_heading("Watch areas")
    _metric_table(watchouts, ["Metric", "Display Value", "Role Percentile"])

pa.section_heading("Detailed breakdown")
category_rows = pa.profile_category_rows(filtered_players, selected_player)
if category_rows.empty:
    st.caption("No detailed category rows are available for this player.")
else:
    category_order = pa.profile_categories_for_role(str(context["role"]))
    categories = [category for category in category_order if category in category_rows["Category"].unique()]
    tabs = st.tabs(categories)
    for tab, category in zip(tabs, categories):
        with tab:
            cat_df = category_rows[category_rows["Category"] == category].sort_values("Role Percentile", ascending=False)
            _metric_table(cat_df, ["Metric", "Display Value", "Role Percentile", "Overall Percentile", "Role Rank"])

pa.section_heading("Similar players")
similar_pool = context["peers"].copy()
if selected_index not in similar_pool.index:
    similar_pool = filtered_players.copy()
similar_metrics = metric_rows["Metric"].dropna().astype(str).tolist() if "Metric" in metric_rows.columns else []
similar = pa.similarity_table(similar_pool, selected_player, top_n=8, metrics=similar_metrics)
if similar.empty:
    st.info("At least two players and one metric are needed to calculate similar players.")
else:
    display_cols = [
        col
        for col in ["Player", "Team", "_Position Display", "Role Group", "Minutes", *similar_metrics, "Similarity"]
        if col in similar.columns
    ]
    st.dataframe(similar[display_cols], width="stretch", hide_index=True)

pa.section_heading("Profile navigation")
n1, n2, n3 = st.columns(3)
with n1:
    st.page_link("views/player_search.py", label="Back to Player Search")
with n2:
    st.page_link("views/player_comparison.py", label="Open Player Comparison")
with n3:
    st.page_link("views/similar_player_search.py", label="Open Similar Player Search")
