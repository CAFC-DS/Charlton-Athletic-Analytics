# =============================================================================
# DATA HUB · DATA QUALITY CHECKS — the questions an analyst would actually ask
# =============================================================================
# Not a generic null/dtype dump — named, domain-specific checks (valid
# ranges, duplicates, missing identifiers, cross-dataset consistency) with a
# clear pass/review status and, for anything flagged, the offending rows.
# =============================================================================
import pandas as pd
import streamlit as st
from utils import data, ui

st.title("✅ Data Quality Checks")
st.caption(
    "Checks a football analyst would want verified before trusting this data — "
    "valid ranges, duplicates, missing identifiers and consistency across datasets."
)

ui.data_refresh_control()
st.divider()

if not data.USE_MOCK_DATA:
    source_status = data.data_source_preflight()
    source_status["Status"] = source_status["Available"].map(
        {True: "✅ Available", False: "❌ Unavailable"}
    )
    st.subheader("Production source availability")
    st.caption(
        "These probes use fully qualified CAFC_DB relations; they do not rely on the empty PUBLIC schema."
    )
    st.dataframe(
        source_status[["Capability", "Status", "Source"]],
        width="stretch",
        hide_index=True,
    )
    if not source_status["Available"].all():
        st.error("One or more CAFC_DB source layers is unavailable to the current Snowflake role.")
    st.divider()

# Every dataset below takes an optional season; without one, each loader
# silently resolves to whichever season sorts last (the newest -- often the
# one with the fewest matches played so far), with no way to check a
# completed prior season instead.
season_options = sorted(data.list_seasons().get("players", []))
if not season_options:
    st.warning("No seasons are available to check.")
    st.stop()
preferred_season = data.preferred_season(season_options)
season = st.selectbox("Season", season_options, index=season_options.index(preferred_season))

players = data.load_players(season=season)
teams = data.load_teams(season=season)
matches = data.load_matches(season=season)

METRIC_COLS = ["Goals /90", "Assists /90", "Bypassed Opponents /90", "Passes to Final 3rd /90"]


def check(name: str, ok: bool, detail: str, offending: pd.DataFrame | None = None) -> dict:
    return {"Check": name, "Status": "✅ Pass" if ok else "⚠️ Review", "Detail": detail, "_rows": offending}


def render_checks(title: str, checks: list[dict]) -> None:
    st.subheader(title)
    summary = pd.DataFrame([{"Check": c["Check"], "Status": c["Status"], "Detail": c["Detail"]} for c in checks])
    st.dataframe(summary, width="stretch", hide_index=True)
    for c in checks:
        if c["_rows"] is not None and len(c["_rows"]):
            with st.expander(f"Show affected rows — {c['Check']}"):
                st.dataframe(c["_rows"], width="stretch", hide_index=True)


# ---- Players -----------------------------------------------------------------
missing_id = players[players["Player"].isna() | players["Team"].isna()]
bad_pass_pct = players[(players["Pass %"] < 0) | (players["Pass %"] > 100)]
negative_stats = players[(players[METRIC_COLS] < 0).any(axis=1)]
dupes = players[players.duplicated(subset=["Player", "Team", "Season"], keep=False)]
implausible_minutes = players[(players["Minutes"] < 0) | (players["Minutes"] > 4500)]
# Range checks alone pass trivially on an all-NaN column (NaN < 0 and NaN >
# 4500 both evaluate False), so a dtype bug that silently nulled out a whole
# column would slip through as "Pass" without this explicit check.
missing_minutes = players[players["Minutes"].isna()]
missing_metrics = players[players[METRIC_COLS].isna().all(axis=1)]

render_checks(f"Players ({len(players)} rows)", [
    check("Player and team identified for every row", len(missing_id) == 0,
          f"{len(missing_id)} row(s) missing a player or team name.", missing_id),
    check("Pass % is a valid percentage (0-100)", len(bad_pass_pct) == 0,
          f"{len(bad_pass_pct)} player(s) outside the valid range.", bad_pass_pct),
    check("No negative stats", len(negative_stats) == 0,
          f"{len(negative_stats)} player(s) with a negative Goals/Assists/Bypassed Opponents/Passes value.", negative_stats),
    check("No duplicate player entries", len(dupes) == 0,
          f"{len(dupes)} row(s) sharing the same player, team and season.", dupes),
    check("Minutes played within a plausible season range (0-4,500)", len(implausible_minutes) == 0,
          f"{len(implausible_minutes)} player(s) outside that range.", implausible_minutes),
    check("Minutes is populated (not silently blank)", len(missing_minutes) == 0,
          f"{len(missing_minutes)} player(s) with no Minutes value at all -- distinct from 0 minutes played.", missing_minutes),
    check("At least one per-90 metric is populated per player", len(missing_metrics) == 0,
          f"{len(missing_metrics)} player(s) with every tracked per-90 metric blank -- likely a join/merge gap rather than a genuinely metric-free player.", missing_metrics),
])
st.divider()

# ---- Teams ---------------------------------------------------------------
missing_team = teams[teams["Team"].isna()]
bad_pass_pct_t = teams[(teams["Pass %"] < 0) | (teams["Pass %"] > 100)]
negative_stats_t = teams[(teams[METRIC_COLS] < 0).any(axis=1)]
expected_teams = 24

render_checks(f"Teams ({len(teams)} rows)", [
    check("Team name present for every row", len(missing_team) == 0,
          f"{len(missing_team)} row(s) missing a team name.", missing_team),
    check("Pass % is a valid percentage (0-100)", len(bad_pass_pct_t) == 0,
          f"{len(bad_pass_pct_t)} team(s) outside the valid range.", bad_pass_pct_t),
    check("No negative stats", len(negative_stats_t) == 0,
          f"{len(negative_stats_t)} team(s) with a negative metric.", negative_stats_t),
    check(f"Squad count matches expected league size ({expected_teams})", len(teams) == expected_teams,
          f"{len(teams)} teams loaded."),
])
st.divider()

# ---- Matches -------------------------------------------------------------
missing_scores = matches[matches["Home Goals"].isna() | matches["Away Goals"].isna()]
if "MatchId" in matches:
    dup_matches = matches[matches["MatchId"].duplicated(keep=False)]
    duplicate_detail = f"{len(dup_matches)} duplicate match row(s)."
else:
    dup_matches = pd.DataFrame()
    duplicate_detail = "MatchId is not available in this dataset, so duplicate ID checks are skipped."
negative_goals = matches[(matches["Home Goals"] < 0) | (matches["Away Goals"] < 0)]

# Compare only seasons present in both current loader outputs.
shared_seasons = set(matches["Season"].dropna()).intersection(teams["Season"].dropna())
shared_matches = matches[matches["Season"].isin(shared_seasons)]
shared_match_team_names = set(shared_matches["Home"]).union(shared_matches["Away"])
known_team_names = set(teams["Team"])
unknown_shared = shared_match_team_names - known_team_names

unverified_venue = matches[~matches["Venue Verified"]]

render_checks(f"Matches ({len(matches)} rows)", [
    check("Every match has both team scores recorded", len(missing_scores) == 0,
          f"{len(missing_scores)} match(es) missing a score.", missing_scores),
    check("No duplicate matches", len(dup_matches) == 0,
          duplicate_detail, dup_matches),
    check("No negative goals", len(negative_goals) == 0,
          f"{len(negative_goals)} match(es) with a negative score.", negative_goals),
    check("Match teams also appear in the same-season Teams dataset", len(unknown_shared) == 0,
          f"{len(unknown_shared)} team name(s) in shared seasons aren't in Teams."),
    check("Home/Away is a verified fact, not an inferred ordering", len(unverified_venue) == 0,
          f"{len(unverified_venue)} match(es) use an unverified Home/Away ordering.", unverified_venue),
])
