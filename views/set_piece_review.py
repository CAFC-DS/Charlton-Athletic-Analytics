# =============================================================================
# POST-MATCH SET PIECE REVIEW - fixture report for both teams
# =============================================================================
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import data
from utils import match_analysis as ma
from utils import set_piece_analysis as spa
from utils import team_badges, ui


SOURCE = (
    "CAFC_DB Impect provider events supply set-piece IDs, main-event markers, delivery coordinates, sequence phases, "
    "shots, goals and shot xG. First contacts and sequence outcomes are transparently derived by the app where needed. "
    "Match values are compared with each team's own full-season per-match baseline; no iteration-average table is used."
)


def _badge(team_name: str) -> str:
    uri = team_badges.badge_data_uri(team_name)
    if not uri:
        return '<span class="spr-fallback">◆</span>'
    return f'<img class="spr-badge" src="{uri}" alt="{ui.esc(team_name)} badge">'


def _scoreboard(match_row: pd.Series) -> None:
    home = str(match_row.get("Home", "Home"))
    away = str(match_row.get("Away", "Away"))
    home_goals = pd.to_numeric(pd.Series([match_row.get("Home Goals")]), errors="coerce").iloc[0]
    away_goals = pd.to_numeric(pd.Series([match_row.get("Away Goals")]), errors="coerce").iloc[0]
    home_score = "–" if pd.isna(home_goals) else f"{home_goals:.0f}"
    away_score = "–" if pd.isna(away_goals) else f"{away_goals:.0f}"
    date_value = pd.to_datetime(match_row.get("Date"), errors="coerce")
    date_text = date_value.strftime("%d %B %Y") if pd.notna(date_value) else "Date unavailable"
    competition = str(match_row.get("Competition", ""))
    st.html(
        f"""
        <div class="spr-scoreboard">
            <div class="spr-meta">{ui.esc(competition)} &middot; {ui.esc(date_text)}</div>
            <div class="spr-score-row">
                <div class="spr-team spr-team-home">{_badge(home)}<span>{ui.esc(home)}</span></div>
                <div class="spr-score">{home_score}<span>&ndash;</span>{away_score}</div>
                <div class="spr-team spr-team-away"><span>{ui.esc(away)}</span>{_badge(away)}</div>
            </div>
        </div>
        """
    )


def _metric_summary(match_sequences: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    rows = []
    for team_name in teams:
        team_rows = match_sequences[
            match_sequences["Team"].astype(str).eq(str(team_name))
            & match_sequences["Set Piece Type"].astype(str).isin(spa.ANALYSED_TYPES)
        ]
        rows.append(
            {
                "Team": team_name,
                "Corners": int(team_rows["Set Piece Type"].eq("Corner").sum()),
                "Indirect Free Kicks": int(team_rows["Set Piece Type"].eq("Indirect Free Kick").sum()),
                "Direct Free Kicks": int(team_rows["Set Piece Type"].eq("Direct Free Kick").sum()),
                "Throw-Ins": int(team_rows["Set Piece Type"].eq("Throw-In").sum()),
                "Set-Play Shots": int(pd.to_numeric(team_rows["Shots"], errors="coerce").fillna(0).sum()),
                "Set-Play Goals": int(pd.to_numeric(team_rows["Goals"], errors="coerce").fillna(0).sum()),
                "Set-Play xG": round(float(pd.to_numeric(team_rows["xG"], errors="coerce").fillna(0).sum()), 3),
            }
        )
    return pd.DataFrame(rows)


def _category_review(
    match_sequences: pd.DataFrame,
    season_sequences: pd.DataFrame,
    season_matches: pd.DataFrame,
    detailed_events: pd.DataFrame,
    teams: list[str],
    set_piece_types: set[str],
    label: str,
    map_kind: str = "delivery",
) -> None:
    ma.section_heading(f"{label} Match Output vs Season Baseline")
    st.caption(
        "The dashed 100 line is each team's own full-season per-match average. For example, 140 means the match value "
        "was 40% above that team's usual rate; it is not a league percentile."
    )
    st.plotly_chart(
        spa.match_vs_season_chart(
            match_sequences,
            season_sequences,
            season_matches,
            teams,
            set_piece_types,
            f"{label}: Match Output Indexed to Own Season Average",
        ),
        width="stretch",
    )

    ma.section_heading(f"{label} First Contacts and Outcomes")
    st.dataframe(
        spa.match_first_contact_table(match_sequences, teams, set_piece_types),
        width="stretch",
        hide_index=True,
    )

    ma.section_heading(f"{label} Trajectories")
    map_columns = st.columns(2)
    for column, team_name in zip(map_columns, teams, strict=False):
        with column:
            if map_kind == "throw":
                figure = spa.throw_in_map(match_sequences, f"{team_name}: {label}", team_name=team_name)
            else:
                figure = spa.delivery_map(
                    match_sequences,
                    f"{team_name}: {label}",
                    set_piece_types,
                    team_name=team_name,
                )
            st.plotly_chart(figure, width="stretch")

    ma.section_heading(f"{label} Shot Locations")
    shot_columns = st.columns(2)
    for column, team_name in zip(shot_columns, teams, strict=False):
        with column:
            st.plotly_chart(
                spa.shot_map(
                    detailed_events,
                    match_sequences,
                    team_name,
                    f"{team_name}: Shots from {label}",
                    set_piece_types=set_piece_types,
                ),
                width="stretch",
            )


ma.page_header(
    "Post-Match Set Piece Review",
    "Compare both teams' restart plans, first contacts, shot value and match output against their own season standards.",
    SOURCE,
    (
        "The review combines provider-coded restart identifiers with sequence outcomes derived by the app. Swing direction, blocking, marking scheme and "
        "movement patterns require video or tracking data and are not inferred."
    ),
)

st.markdown(
    """
    <style>
    .spr-scoreboard {background:linear-gradient(135deg,#111111 0%,#241113 60%,#9c0214 130%);border-top:5px solid #c30017;border-radius:8px;color:#fff;margin:8px 0 20px;padding:20px 28px 24px;}
    .spr-meta {color:rgba(255,255,255,.72);font-size:.82rem;font-weight:750;letter-spacing:.08em;text-align:center;text-transform:uppercase;}
    .spr-score-row {align-items:center;display:grid;gap:24px;grid-template-columns:1fr auto 1fr;margin-top:16px;}
    .spr-team {align-items:center;display:flex;font-size:clamp(1rem,1.7vw,1.35rem);font-weight:800;gap:14px;min-width:0;}
    .spr-team-home {justify-content:flex-end;text-align:right}.spr-team-away {justify-content:flex-start;text-align:left}
    .spr-badge {height:58px;object-fit:contain;width:58px}.spr-fallback {color:#c30017;font-size:2rem}
    .spr-score {align-items:center;display:flex;font-size:clamp(2rem,4vw,3.3rem);font-weight:350;gap:12px;letter-spacing:-.04em;white-space:nowrap}.spr-score span {color:rgba(255,255,255,.5)}
    @media(max-width:700px){.spr-scoreboard{padding:16px}.spr-score-row{gap:10px}.spr-team{font-size:.86rem;gap:7px}.spr-badge{height:38px;width:38px}}
    </style>
    """,
    unsafe_allow_html=True,
)

available_seasons = data.list_seasons().get("matches", [])
event_seasons = [season for season in available_seasons if str(season).replace("2025", "25") == "25/26"]
if not event_seasons:
    st.warning("No event-level season is available for this report.")
    st.stop()

season = st.selectbox("Match Season", event_seasons, key="set_piece_review_season")
matches = ma.load_matches(season)
if matches.empty:
    st.warning("No match data is available for this season.")
    st.stop()

match_row = ma.match_selector(matches, key="set_piece_review_match")
teams = ma.fixture_teams(match_row)
if len(teams) != 2:
    st.warning("Both fixture teams are required for the comparison report.")
    st.stop()

_scoreboard(match_row)

with st.spinner("Building match set-piece sequences and season baselines..."):
    season_sequences = data.load_set_piece_sequences(season)
    detailed_events = data.load_set_piece_events(season=season, match_id=match_row.get("MatchId"), limit=12000)

if season_sequences.empty:
    st.warning("No provider-defined set-piece sequences are available for this season.")
    st.stop()
match_sequences = spa.filter_sequences_to_matches(season_sequences, [match_row.get("MatchId")])

ma.section_heading("Fixture Set-Play Summary")
st.dataframe(_metric_summary(match_sequences, teams), width="stretch", hide_index=True)
st.caption(
    "A direct free kick can have SHOT as its main event, so it is separated from indirect deliveries rather than "
    "being missed by a simple FREE_KICK action count. Goal kicks are excluded."
)

corner_tab, free_kick_tab, throw_tab = st.tabs(["Corners", "Free Kicks", "Throw-Ins"])

with corner_tab:
    _category_review(
        match_sequences,
        season_sequences,
        matches,
        detailed_events,
        teams,
        {"Corner"},
        "Corners",
    )

with free_kick_tab:
    indirect_tab, direct_tab = st.tabs(["Indirect Deliveries", "Direct Attempts"])
    with indirect_tab:
        _category_review(
            match_sequences,
            season_sequences,
            matches,
            detailed_events,
            teams,
            {"Indirect Free Kick"},
            "Indirect Free Kicks",
        )
    with direct_tab:
        _category_review(
            match_sequences,
            season_sequences,
            matches,
            detailed_events,
            teams,
            {"Direct Free Kick"},
            "Direct Free Kicks",
        )

with throw_tab:
    _category_review(
        match_sequences,
        season_sequences,
        matches,
        detailed_events,
        teams,
        {"Throw-In"},
        "Throw-Ins",
        map_kind="throw",
    )

ma.section_heading("Restart-by-Restart Detail")
with st.expander("Show Match Set-Piece Sequence Data"):
    st.dataframe(spa.match_sequence_table(match_sequences), width="stretch", hide_index=True)
